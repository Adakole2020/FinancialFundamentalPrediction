import io
import math
import os
import warnings

import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2
import random
import torch
import wandb
from einops import rearrange

from spacetimeformer.eval_stats import mape
from spacetimeformer.distribution import SkewNormal


def _assert_squeeze(x):
    assert len(x.shape) == 2
    return x.squeeze(-1)


def plot(x_c, y_c, x_t, y_t, idx, title, preds, shape=None, pad_val=None, conf=None):
    """
    Plot the context, true values, and forecasted values.

    Args:
        x_c (array-like): The x-axis values for the context.
        y_c (array-like): The y-axis values for the context.
        x_t (array-like): The x-axis values for the target.
        y_t (array-like): The y-axis values for the target.
        idx (int): The index of the target value to plot.
        title (str): The title of the plot.
        preds (array-like): The forecasted values.
        shape (float, optional): The shape value to display in the title. Defaults to None.
        pad_val (float, optional): The padding value to exclude from the plot. Defaults to None.
        conf (array-like, optional): The confidence interval values. Defaults to None.

    Returns:
        array-like: The image of the plot.
    """
    y_c = y_c[..., 0]
    y_t = _assert_squeeze(y_t)
    preds = _assert_squeeze(preds)
    
    if pad_val is not None:
        y_c = y_c[y_c != pad_val]
        yt_mask = y_t != pad_val
        y_t = y_t[yt_mask]
        preds = preds[yt_mask]

    fig, ax = plt.subplots(figsize=(10, 6))
    xaxis_c = np.arange(len(y_c))
    xaxis_t = np.arange(len(y_c), len(y_c) + len(y_t))
    context = pd.DataFrame({"xaxis_c": xaxis_c, "y_c": y_c})
    target = pd.DataFrame({"xaxis_t": xaxis_t, "y_t": y_t, "pred": preds})
    sns.lineplot(data=context, x="xaxis_c", y="y_c", label="Context", linewidth=2)
    ax.scatter(
        x=target["xaxis_t"], y=target["y_t"], c="grey", label="True", linewidth=1.0
    )
    sns.lineplot(data=target, x="xaxis_t", y="pred", label="Forecast", linewidth=2, alpha=0.7)
    if conf is not None:
        ax.fill_between(
            xaxis_t, (conf[0][..., idx]), (conf[1][..., idx]), color="orange", alpha=0.1, label = "95% CI"
        )
    ax.legend(loc="upper left", prop={"size": 10})
    ax.set_xticks(range(0, len(y_c) + len(y_t), 4))
    ax.set_xlabel("")
    ax.set_ylabel("")
    
    if shape is not None:
        title = f"{title} - Shape: {shape.mean():.2f} +/- {shape.std():.2f}"
    ax.set_title(title)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=128)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    plt.close(fig)
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def show_image(data, title, figsize=(7,7), x_tick_spacing=None, y_tick_spacing=None, cmap="Blues"):
    """
    Display an image using matplotlib.

    Args:
        data (numpy.ndarray): The image data.
        title (str): The title of the image.
        figsize (tuple, optional): The size of the figure. Defaults to (7, 7).
        x_tick_spacing (int, optional): The spacing between x-axis ticks. Defaults to None.
        y_tick_spacing (int, optional): The spacing between y-axis ticks. Defaults to None.
        cmap (str, optional): The colormap to use. Defaults to "Blues".

    Returns:
        numpy.ndarray: The image as a NumPy array.

    """
    fig, ax = plt.subplots(figsize=figsize)

    plt.imshow(data, cmap=cmap)
    if x_tick_spacing:
        plt.xticks(np.arange(0, data.shape[-1] + 1, x_tick_spacing), [])
        
    if y_tick_spacing:
        plt.yticks(np.arange(0, data.shape[0] + 1, y_tick_spacing), [])

    plt.title(title, fontsize=8)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=256, bbox_inches="tight")
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    plt.close(fig)
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class PredictionPlotterCallback(pl.Callback):
    """
    Callback for plotting predictions during training.

    Args:
        test_batches (list): List of test data batches.
        var_idxs (list, optional): List of variable indices to plot. Defaults to None.
        var_names (list, optional): List of variable names corresponding to var_idxs. Defaults to None.
        pad_val (float, optional): Padding value for plotting. Defaults to None.
        total_samples (int, optional): Total number of samples to plot. Defaults to 4.
        identifiers (list, optional): List of identifiers for each sample. Defaults to None.
        log_to_wandb (bool, optional): Whether to log images to Weights & Biases. Defaults to True.
    """

    def __init__(self,
        test_batches,
        var_idxs=None,
        var_names=None,
        pad_val=None,
        total_samples=4,
        identifiers=None,
        log_to_wandb=True,
    ):
        self.test_data = test_batches
        self.total_samples = total_samples
        self.pad_val = pad_val
        self.log_to_wandb = log_to_wandb
        
        if var_idxs is None and var_names is None:
            d_yt = self.test_data[-1].shape[-1]
            var_idxs = list(range(d_yt))
            var_names = [f"y{i}" for i in var_idxs]
        
        self.var_idxs = var_idxs
        self.var_names = var_names
        self.identifiers = identifiers
        self.imgs = None

    def on_train_epoch_end(self, trainer, model):
        """
        Method called at the end of each training epoch.

        Args:
            trainer (pytorch_lightning.Trainer): The trainer object.
            model (torch.nn.Module): The model being trained.
        """
        idxs = [random.sample(range(self.test_data[0].shape[0]), k=self.total_samples)]
        x_c, y_c, x_t, y_t = [i[idxs].detach().to(model.device) for i in self.test_data]
        with torch.no_grad():
            preds, *_ = model(x_c, y_c, x_t, y_t, **model.eval_step_forward_kwargs)
            if isinstance(preds, SkewNormal):
                samples = preds.sample((200,)).permute(1, 2, 0, 3)
                L_CI = torch.quantile(samples, 0.025, dim=-2)
                R_CI = torch.quantile(samples, 0.975, dim=-2)
                shape = preds.alpha
                preds = preds.mean
                conf = [L_CI, R_CI]
            else:
                conf = None
        imgs = []
        for i in range(preds.shape[0]):
            if self.identifiers is not None:
                identifier = " [" + self.identifiers[idxs[0][i]] + "]"
            else:
                identifier = ""
            for var_idx, var_name in zip(self.var_idxs, self.var_names):
                img = plot(
                    x_c[i].cpu().numpy(),
                    y_c[i].cpu().numpy(),
                    x_t[i].cpu().numpy(),
                    y_t[i].cpu().numpy(),
                    idx=var_idx,
                    title=f"{var_name}{identifier}",
                    preds=preds[i].cpu().numpy(),
                    shape=shape[i].cpu().numpy() if shape is not None else None,
                    pad_val=self.pad_val,
                    conf=[conf[0][i], conf[1][i]] if conf is not None else None,
                )
                if img is not None:
                    if self.log_to_wandb:
                        img = wandb.Image(img)
                    imgs.append(img)

        if self.log_to_wandb:
            trainer.logger.experiment.log(
                {
                    "test/prediction_plots": imgs,
                    "global_step": trainer.global_step,
                }
            )
        else:
            self.imgs = imgs


class AttentionMatrixCallback(pl.Callback):
    """
    Callback class for computing and logging attention matrices during training.

    Args:
        test_batches (tuple): Tuple of test data batches.
        layer (int): Layer index for which attention matrices should be computed.
        total_samples (int): Number of samples to compute attention matrices for.
        raw_data_dir (str): Directory path to raw data.

    Attributes:
        test_data (tuple): Tuple of test data batches.
        total_samples (int): Number of samples to compute attention matrices for.
        layer (int): Layer index for which attention matrices should be computed.
        raw_data_dir (str): Directory path to raw data.
    """

    def __init__(self, test_batches, layer=0, total_samples=32, raw_data_dir=None):
        self.test_data = test_batches
        self.total_samples = total_samples
        self.layer = layer
        self.raw_data_dir = raw_data_dir

    def _get_attns(self, model):
        """
        Compute attention matrices for the given model.

        Args:
            model: The model for which attention matrices should be computed.

        Returns:
            Tuple: Tuple containing the encoder attention matrix and decoder attention matrix.
        """
        idxs = [random.sample(range(self.test_data[0].shape[0]), k=self.total_samples)]
        x_c, y_c, x_t, y_t = [i[idxs].detach().to(model.device) for i in self.test_data]
        enc_attns, dec_attns = None, None
        # save memory by doing inference 1 example at a time
        for i in range(self.total_samples):
            x_ci = x_c[i].unsqueeze(0)
            y_ci = y_c[i].unsqueeze(0)
            x_ti = x_t[i].unsqueeze(0)
            y_ti = y_t[i].unsqueeze(0)
            
            with torch.no_grad():
                *_, (enc_self_attn, dec_cross_attn) = model(
                    x_ci, y_ci, x_ti, y_ti, output_attn=True
                )
            if enc_attns is None:
                enc_attns = [[a] for a in enc_self_attn]
            else:
                for cum_attn, attn in zip(enc_attns, enc_self_attn):
                    cum_attn.append(attn)
            if dec_attns is None:
                dec_attns = [[a] for a in dec_cross_attn]
            else:
                for cum_attn, attn in zip(dec_attns, dec_cross_attn):
                    cum_attn.append(attn)

        # re-concat over batch dim, avg over batch dim
        if enc_attns:
            enc_attns = [torch.cat(a, dim=0) for a in enc_attns][self.layer].mean(0)
        else:
            enc_attns = None
        if dec_attns:
            dec_attns = [torch.cat(a, dim=0) for a in dec_attns][self.layer].mean(0)
        else:
            dec_attns = None
        return enc_attns, dec_attns

    def _make_imgs(self, attns, img_title_prefix):
        """
        Create images from attention matrices.

        Args:
            attns: Attention matrices.
            img_title_prefix (str): Prefix for the image titles.

        Returns:
            list: List of images created from attention matrices.
        """
        heads = [i for i in range(attns.shape[0])] + ["avg", "sum"]
        imgs = []
        for head in heads:
            if head == "avg":
                a_head = attns.mean(0)
            elif head == "sum":
                a_head = attns.sum(0)
            else:
                a_head = attns[head]

            a_head /= torch.max(a_head, dim=-1)[0].unsqueeze(1)
            if img_title_prefix.startswith("Self"):
                imgs.append(
                    wandb.Image(
                        show_image(
                            a_head.cpu().numpy(),
                            f"{img_title_prefix} Head {str(head)}",
                            figsize=(4, 4),
                            x_tick_spacing=self.test_data[0].shape[-2],
                            y_tick_spacing=self.test_data[0].shape[-2],
                            cmap="Blues",
                        )
                    )
                )
            else:
                imgs.append(
                    wandb.Image(
                        show_image(
                            a_head.cpu().numpy(),
                            f"{img_title_prefix} Head {str(head)}",
                            figsize=(6,6),
                            x_tick_spacing=self.test_data[-1].shape[-2],
                            y_tick_spacing=self.test_data[0].shape[-2],
                            cmap="Blues",
                        )
                    )
                )
        return imgs
    
    
    def _pos_sim_scores(self, embedding, seq_len, device):
        """
        Compute position embedding similarity scores.

        Args:
            embedding: The embedding layer.
            seq_len (int): Length of the sequence.
            device: The device on which the computation should be performed.

        Returns:
            ndarray: Array of position embedding similarity scores.
        """
        if embedding.position_emb == "t2v":
            inp = torch.arange(seq_len).float().to(device).view(1, -1, 1)
            encoder_embs = embedding.local_emb(inp)[0, :, 1:]
        elif embedding.position_emb == "abs":
            encoder_embs = embedding.local_emb(torch.arange(seq_len).to(device).long())
        cos_sim = torch.nn.CosineSimilarity(dim=0)
        scores = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            for j in range(0, i + 1):
                sim = cos_sim(encoder_embs[i], encoder_embs[j])
                scores[i, j] = sim
                scores[j, i] = sim
        return scores

    
    def on_train_epoch_end(self, trainer, model):
        """
        Callback method called at the end of each training epoch.

        Args:
            trainer: The PyTorch Lightning trainer.
            model: The model being trained.
        """
        self_attns, cross_attns = self._get_attns(model)

        if self_attns is not None:
            self_attn_imgs = self._make_imgs(
                self_attns, f"Self Attn, Layer {self.layer},"
            )
            trainer.logger.experiment.log(
                {"test/self_attn": self_attn_imgs, "global_step": trainer.global_step}
            )
        if cross_attns is not None:
            cross_attn_imgs = self._make_imgs(
                cross_attns, f"Cross Attn, Layer {self.layer},"
            )
            trainer.logger.experiment.log(
                {"test/cross_attn": cross_attn_imgs, "global_step": trainer.global_step}
            )
        
        enc_emb_sim = self._pos_sim_scores(
            model.spacetimeformer.enc_embedding,
            seq_len=self.test_data[1].shape[1],
            device=model.device,
        )
        dec_emb_sim = self._pos_sim_scores(
            model.spacetimeformer.dec_embedding,
            seq_len=self.test_data[3].shape[1],
            device=model.device,
        )
        emb_sim_imgs = [
            wandb.Image(
                show_image(
                    enc_emb_sim,
                    f"Encoder Position Emb. Similarity",
                    figsize=(4, 4),
                    x_tick_spacing=enc_emb_sim.shape[-1],
                    y_tick_spacing=enc_emb_sim.shape[-1],
                    cmap="Greens",
                )
            ),
            wandb.Image(
                show_image(
                    dec_emb_sim,
                    f"Decoder Position Emb. Similarity",
                    figsize=(4, 4),
                    x_tick_spacing=dec_emb_sim.shape[-1],
                    y_tick_spacing=dec_emb_sim.shape[-1],
                    cmap="Greens",
                )
            ),
        ]
        trainer.logger.experiment.log(
            {"test/pos_embs": emb_sim_imgs, "global_step": trainer.global_step}
        )
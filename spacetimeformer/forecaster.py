from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

import spacetimeformer as stf
from spacetimeformer.distribution import SkewNormal


class Forecaster(pl.LightningModule, ABC):
    def __init__(
        self,
        d_x: int,
        d_yc: int,
        d_yt: int,
        learning_rate: float = 1e-3,
        l2_coeff: float = 0,
        loss: str = "mse",
        linear_window: int = 0,
        linear_shared_weights: bool = False,
        use_revin: bool = False,
        use_seasonal_decomp: bool = False,
        verbose: int = True,
    ):
        """
        Forecaster class for time series forecasting using SpaceTimeFormer.

        Args:
            d_x (int): Dimensionality of the input features.
            d_yc (int): Dimensionality of the conditioning features.
            d_yt (int): Dimensionality of the target features.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
            l2_coeff (float, optional): L2 regularization coefficient. Defaults to 0.
            loss (str, optional): Loss function to use. Can be "mse", "mae", "smape", or "nll". Defaults to "mse".
            linear_window (int, optional): Window size for the linear model. Defaults to 0.
            linear_shared_weights (bool, optional): Whether to use shared weights in the linear model. Defaults to False.
            use_revin (bool, optional): Whether to use RevIN for conditioning features. Defaults to False.
            use_seasonal_decomp (bool, optional): Whether to use seasonal decomposition for conditioning features. Defaults to False.
            verbose (int, optional): Verbosity level. Defaults to True.
        """
        super().__init__()
        torch.set_flush_denormal(True)
        
        self.validation_step_outputs = [] # https://github.com/Lightning-AI/pytorch-lightning/pull/16520
        
        qprint = lambda _msg_: print(_msg_) if verbose else None
        qprint("Forecaster")
        qprint(f"\tL2: {l2_coeff}")
        qprint(f"\tLinear Window: {linear_window}")
        qprint(f"\tLinear Shared Weights: {linear_shared_weights}")
        qprint(f"\tRevIN: {use_revin}")
        qprint(f"\tDecomposition: {use_seasonal_decomp}")

        self._inv_scaler = lambda x: x
        self.l2_coeff = l2_coeff
        self.learning_rate = learning_rate
        self.time_masked_idx = None
        self.null_value = None
        self.loss = loss

        if linear_window:
            self.linear_model = stf.linear_model.LinearModel(
                linear_window, shared_weights=linear_shared_weights, d_yt=d_yt
            )
        else:
            self.linear_model = lambda x, *args, **kwargs: 0.0

        self.use_revin = use_revin
        if use_revin:
            assert d_yc == d_yt, "TODO: figure out exo case for revin"
            self.revin = stf.revin.RevIN(num_features=d_yc)
        else:
            self.revin = lambda x, *args, **kwargs: x

        self.use_seasonal_decomp = use_seasonal_decomp
        if use_seasonal_decomp:
            self.seasonal_decomp = stf.revin.SeriesDecomposition(kernel_size=25)
        else:
            self.seasonal_decomp = lambda x: (x, x.clone())

        self.d_x = d_x
        self.d_yc = d_yc
        self.d_yt = d_yt
        self.save_hyperparameters()

    def set_null_value(self, val: float) -> None:
        """
        Set the null value for the target features.

        Args:
            val (float): Null value.
        """
        self.null_value = val

    def set_inv_scaler(self, scaler) -> None:
        """
        Set the inverse scaler function.

        Args:
            scaler: Inverse scaler function.
        """
        self._inv_scaler = scaler

    def set_scaler(self, scaler) -> None:
        """
        Set the scaler function.

        Args:
            scaler: Scaler function.
        """
        self._scaler = scaler

    @property
    @abstractmethod
    def train_step_forward_kwargs(self):
        """
        Get the forward kwargs for the training step.

        Returns:
            dict: Forward kwargs for the training step.
        """
        return {}

    @property
    @abstractmethod
    def eval_step_forward_kwargs(self):
        """
        Get the forward kwargs for the evaluation step.

        Returns:
            dict: Forward kwargs for the evaluation step.
        """
        return {}

    def loss_fn(
        self, true: torch.Tensor, preds, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the loss function.

        Args:
            true (torch.Tensor): True target values.
            preds: Predicted target values.
            mask (torch.Tensor): Mask indicating valid values.

        Returns:
            torch.Tensor: Loss value.
        """
        if self.loss == "mse":
            if isinstance(preds, SkewNormal):
                preds = preds.mean
            return F.mse_loss(mask * true, mask * preds)
        elif self.loss == "mae":
            if isinstance(preds, SkewNormal):
                preds = preds.mean
            return torch.abs((true - preds) * mask).mean()
        elif self.loss == "smape":
            if isinstance(preds, SkewNormal):
                preds = preds.mean
            num = 2.0 * abs(preds - true)
            den = abs(preds.detach()) + abs(true) + 1e-5
            return 100.0 * (mask * (num / den)).sum() / max(mask.sum(), 1)
        elif self.loss == "nll":
            assert isinstance(preds, SkewNormal), "NLL Loss only works with a Distribution but the code is optimized for normal"
            log_prob = preds.log_prob(true)
            return -(mask * log_prob).sum(-1).sum(-1).mean()
        else:
            raise ValueError(f"Unrecognized Loss Function : {self.loss}")

    def forecasting_loss(
        self, outputs, y_t: torch.Tensor, time_mask: int
    ) -> Tuple[torch.Tensor]:
        """
        Compute the forecasting loss.

        Args:
            outputs: Predicted target values.
            y_t (torch.Tensor): True target values.
            time_mask (int): Time mask.

        Returns:
            Tuple[torch.Tensor]: Forecasting loss and mask.
        """
        if self.null_value is not None:
            null_mask_mat = y_t != self.null_value
        else:
            null_mask_mat = torch.ones_like(y_t)
            
        # genuine NaN failsafe
        null_mask_mat *= ~torch.isnan(y_t)

        time_mask_mat = y_t > -float("inf")
        if time_mask is not None:
            time_mask_mat[:, time_mask:] = False

        full_mask = time_mask_mat * null_mask_mat
        forecasting_loss = self.loss_fn(y_t, outputs, full_mask)

        return forecasting_loss, full_mask

    def compute_loss(
        self,
        batch: Tuple[torch.Tensor],
        time_mask: int = None,
        forward_kwargs: dict = {},
    ) -> Tuple[torch.Tensor]:
        """
        Compute the loss for a batch of data.

        Args:
            batch (Tuple[torch.Tensor]): Batch of data.
            time_mask (int, optional): Time mask. Defaults to None.
            forward_kwargs (dict, optional): Forward kwargs. Defaults to {}.

        Returns:
            Tuple[torch.Tensor]: Loss value, outputs, and mask.
        """
        x_c, y_c, x_t, y_t = batch
        outputs, *_ = self(x_c, y_c, x_t, y_t, **forward_kwargs)
        loss, mask = self.forecasting_loss(
            outputs=outputs, y_t=y_t, time_mask=time_mask
        )
        return loss, outputs, mask

    def predict(
        self,
        x_c: torch.Tensor,
        y_c: torch.Tensor,
        x_t: torch.Tensor,
        sample_preds: bool = False,
    ) -> torch.Tensor:
        """
        Make predictions for a given input.

        Args:
            x_c (torch.Tensor): Input features for conditioning.
            y_c (torch.Tensor): Target features for conditioning.
            x_t (torch.Tensor): Input features for prediction.
            sample_preds (bool, optional): Whether to sample predictions from a distribution. Defaults to False.

        Returns:
            torch.Tensor: Predicted target values.
        """
        og_device = y_c.device
        # move to model device
        x_c = x_c.to(self.device).float()
        x_t = x_t.to(self.device).float()
        # move y_c to cpu if it isn't already there, scale, and then move back to the model device
        y_c = torch.from_numpy(self._scaler(y_c.cpu().numpy())).to(self.device).float()
        # create dummy y_t of zeros
        y_t = (
            torch.zeros((x_t.shape[0], x_t.shape[1], y_c.shape[2]))
            .to(self.device)
            .float()
        )

        with torch.no_grad():
            # gradient-free prediction
            normalized_preds, *_ = self.forward(
                x_c, y_c, x_t, y_t, **self.eval_step_forward_kwargs
            )

        # handle case that the output is a distribution (spacetimeformer)
        if isinstance(normalized_preds, SkewNormal):
            if sample_preds:
                normalized_preds = normalized_preds.sample((100,))
            else:
                normalized_preds = normalized_preds.mean

        # preds --> cpu --> inverse scale to original units --> original device of y_c
        preds = (
            torch.from_numpy(self._inv_scaler(normalized_preds.cpu().numpy()))
            .to(og_device)
            .float()
        )
        return preds

    @abstractmethod
    def forward_model_pass(
        self,
        x_c: torch.Tensor,
        y_c: torch.Tensor,
        x_t: torch.Tensor,
        y_t: torch.Tensor,
        **forward_kwargs,
    ) -> Tuple[torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            x_c (torch.Tensor): Input features for conditioning.
            y_c (torch.Tensor): Target features for conditioning.
            x_t (torch.Tensor): Input features for prediction.
            y_t (torch.Tensor): True target values.

        Returns:
            Tuple[torch.Tensor]: Predicted target values and any additional outputs.
        """
        return NotImplemented

    def forward(
        self,
        x_c: torch.Tensor,
        y_c: torch.Tensor,
        x_t: torch.Tensor,
        y_t: torch.Tensor,
        **forward_kwargs,
    ) -> Tuple[torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            x_c (torch.Tensor): Input features for conditioning.
            y_c (torch.Tensor): Target features for conditioning.
            x_t (torch.Tensor): Input features for prediction.
            y_t (torch.Tensor): True target values.
            forward_kwargs (dict): Forward kwargs.

        Returns:
            Tuple[torch.Tensor]: Predicted target values and any additional outputs.
        """
        x_c, y_c, x_t, y_t = self.nan_to_num(x_c, y_c, x_t, y_t)
        _, pred_len, d_yt = y_t.shape

        y_c = self.revin(y_c, mode="norm")  # does nothing if use_revin = False

        seasonal_yc, trend_yc = self.seasonal_decomp(
            y_c
        )  # both are the original if use_seasonal_decomp = False
        
        preds, *extra = self.forward_model_pass(x_c, seasonal_yc, x_t, y_t, **forward_kwargs)
        baseline = self.linear_model(trend_yc, pred_len=pred_len, d_yt=d_yt)
        
        if isinstance(preds, SkewNormal):
            preds.loc = self.revin(preds.loc + baseline, mode="denorm")
            output = preds
        else:
            output = self.revin(preds + baseline, mode="denorm")
            
        if extra:
            return (output,) + tuple(extra)
        return (output,)

    def _compute_stats(self, pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor):
        """
        Compute evaluation statistics.

        Args:
            pred (torch.Tensor): Predicted target values.
            true (torch.Tensor): True target values.
            mask (torch.Tensor): Mask indicating valid values.

        Returns:
            dict: Evaluation statistics.
        """
        pred = pred * mask
        true = torch.nan_to_num(true) * mask

        adj = mask.cpu().numpy() + 1e-5
        pred = pred.detach().cpu().numpy()
        true = true.detach().cpu().numpy()
        scaled_pred = pred
        scaled_true = true
        stats = {
            "mape": stf.eval_stats.mape(scaled_true, scaled_pred) / adj,
            "mae": stf.eval_stats.mae(scaled_true, scaled_pred) / adj,
            "mse": stf.eval_stats.mse(scaled_true, scaled_pred) / adj,
            "rse": stf.eval_stats.rrse(scaled_true, scaled_pred) / adj,
            "smape": stf.eval_stats.smape(scaled_true, scaled_pred) / adj,
        }
        return stats

    def step(self, batch: Tuple[torch.Tensor], train: bool = False):
        """
        Perform a forward pass and compute loss and evaluation statistics.

        Args:
            batch (Tuple[torch.Tensor]): Batch of data.
            train (bool, optional): Whether it's a training step. Defaults to False.

        Returns:
            dict: Evaluation statistics.
        """
        kwargs = (
            self.train_step_forward_kwargs if train else self.eval_step_forward_kwargs
        )
        time_mask = self.time_masked_idx if train else None

        # first forward-backward pass
        loss, output, mask = self.compute_loss(
            batch=batch,
            time_mask=time_mask,
            forward_kwargs=kwargs,
        )
        *_, y_t = batch
        stats = self._compute_stats(mask * output, mask * y_t)
        stats["loss"] = loss
        return stats

    def training_step(self, batch, batch_idx):
        """
        Training step.

        Args:
            batch: Batch of data.
            batch_idx: Batch index.

        Returns:
            dict: Evaluation statistics.
        """
        stats = self.step(batch, train=True)
        self._log_stats("train", stats)
        return stats

    def validation_step(self, batch, batch_idx):
        """
        Validation step.

        Args:
            batch: Batch of data.
            batch_idx: Batch index.

        Returns:
            dict: Evaluation statistics.
        """
        stats = self.step(batch, train=False)
        self.validation_step_outputs.append(stats)
        self._log_stats("val", stats)
        return stats

    def test_step(self, batch, batch_idx):
        """
        Test step.

        Args:
            batch: Batch of data.
            batch_idx: Batch index.

        Returns:
            dict: Evaluation statistics.
        """
        stats = self.step(batch, train=True)
        self._log_stats("test", stats)
        return stats

    def _log_stats(self, section, outs):
        """
        Log evaluation statistics.

        Args:
            section (str): Section name.
            outs (dict): Evaluation statistics.
        """
        for key in outs.keys():
            stat = outs[key]
            if isinstance(stat, np.ndarray) or isinstance(stat, torch.Tensor):
                stat = stat.mean()
            if key == "loss":
                self.log(f"{section}/{key}", stat, sync_dist=True, prog_bar=True)
            else:
                self.log(f"{section}/{key}", stat, sync_dist=True)

    def training_step_end(self, outs):
        return {"loss": outs["loss"].mean()}

    def validation_step_end(self, outs):
        return {"loss": outs["loss"].mean()}

    def test_step_end(self, outs):
        return {"loss": outs["loss"].mean()}

    def predict_step(self, batch, batch_idx):
        return self(*batch, **self.eval_step_forward_kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.l2_coeff
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=3,
            factor=0.2,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }

    @classmethod
    def add_cli(self, parser):
        parser.add_argument("--gpus", type=int, nargs="+")
        parser.add_argument("--l2_coeff", type=float, default=1e-6)
        parser.add_argument("--learning_rate", type=float, default=1e-4)
        parser.add_argument("--grad_clip_norm", type=float, default=0)
        parser.add_argument("--linear_window", type=int, default=0)
        parser.add_argument("--use_revin", action="store_true")
        parser.add_argument(
            "--loss", type=str, default="nll", choices=["mse", "mae", "nll", "smape", "rse"]
        )
        parser.add_argument("--linear_shared_weights", action="store_true")
        parser.add_argument("--use_seasonal_decomp", action="store_true")
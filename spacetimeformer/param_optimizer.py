from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import numpy as np

from spacetimeformer.spacetimeformer_model import Spacetimeformer_Forecaster
from spacetimeformer.data import FundamentalsDataModule, FundamentalsCSVSeries
from spacetimeformer.plot import PredictionPlotterCallback, AttentionMatrixCallback
from argparse import ArgumentParser
import sys

import pytorch_lightning as pl
from pytorch_lightning.profilers import PyTorchProfiler 
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint, EarlyStopping
import torch

pl.seed_everything(123, workers=True)
rstate = np.random.default_rng(123)

import warnings
import wandb
import os
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=DeprecationWarning)


class SpaceTimeformerProgressBar(TQDMProgressBar):
    def get_metrics(self, *args, **kwargs):
        # don't show the version number
        items = super().get_metrics(*args, **kwargs)
        items.pop("v_num", None)
        return items


def create_parser():
    """
    Create an argument parser for command-line arguments.

    Returns:
        argparse.ArgumentParser: The argument parser object.
    """
    parser = ArgumentParser()

    FundamentalsCSVSeries.add_cli(parser)  # Add command-line arguments for FundamentalsCSVSeries
    FundamentalsDataModule.add_cli(parser)  # Add command-line arguments for FundamentalsDataModule

    Spacetimeformer_Forecaster.add_cli(parser)  # Add command-line arguments for Spacetimeformer_Forecaster

    parser.add_argument("--accumulate", type=int, default=1)  # Add an argument for accumulation
    parser.add_argument("--patience", type=int, default=5)  # Add an argument for patience
    parser.add_argument(
        "--execution_type", type=str, default="train", choices=["train", "test"]
    )  # Add an argument for execution type
    
    if len(sys.argv) > 3 and sys.argv[3] == "-h":
        parser.print_help()
        sys.exit(0)

    return parser

def trainer(args):
    """
    Trains a Spacetimeformer model for financial fundamentals prediction.

    Args:
        args: A dictionary containing the arguments for training.

    Returns:
        None
    """
    load_dotenv()
    
    log_dir = os.getenv("STF_LOG_DIR")  # Get the log directory from environment variables
    if log_dir is None:
        log_dir = "./data/STF_LOG_DIR"
        print(
            "Using default wandb log dir path of ./data/STF_LOG_DIR. This can be adjusted with the environment variable `STF_LOG_DIR`"
        )
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    project = os.getenv("STF_WANDB_PROJ")  # Get the project name from environment variables
    assert (
        project is not None
    ), "Please set environment variables `STF_WANDB_PROJ` with \n\
        your wandb project title."
    experiment = wandb.init(
        project=project,
        config={
        "architecture": "Spacetimeformer",
        "epochs": 15,
        "method": args.embed_method,
        },
        dir=log_dir,
    )
    config = wandb.config
    wandb.run.name = "Financial_Fundamentals_Train"
    wandb.run.save()
    logger = pl.loggers.WandbLogger(
        experiment=experiment,
        save_dir=log_dir,
    )
    forecaster = Spacetimeformer_Forecaster(d_x=2, d_yc=35, d_yt = 1, categorical_dict_sizes = [11,25,74,163], 
    max_seq_len=args.context_points + args.target_points,
    start_token_len=0,
    attn_factor=0,
    d_model=96,
    d_queries_keys=32,
    d_values=32,
    n_heads=8,
    e_layers=2,
    d_layers=2,
    d_ff=800,
    dropout_emb=0.2,
    dropout_attn_out=0,
    dropout_attn_matrix=0,
    dropout_qkv=0,
    dropout_ff=0.2,
    pos_emb_type="t2v",
    use_final_norm=True,
    global_self_attn="performer",
    local_self_attn="performer" if args.embed_method == "spatio-temporal" else "none",
    global_cross_attn="performer",
    local_cross_attn="performer" if args.embed_method == "spatio-temporal" else "none",
    performer_kernel="softmax",
    performer_redraw_interval=150,
    attn_time_windows=1,
    use_shifted_time_windows=args.use_shifted_time_windows,
    norm="batch",
    activation="gelu",
    init_lr=1e-8,
    base_lr=5e-4,
    warmup_steps=20,
    decay_factor=0.9, # https://stackoverflow.com/questions/67746083/setting-a-minimum-learning-rate-on-reduce-on-plateau
    initial_downsample_convs=args.initial_downsample_convs,
    intermediate_downsample_convs=args.intermediate_downsample_convs,
    embed_method=args.embed_method,
    l2_coeff=1e-6,
    loss=args.loss,
    class_loss_imp=0.05,
    recon_loss_imp=0.1,
    time_emb_dim=8,
    categorical_embedding_dim=8,
    null_value=-1e2,
    pad_value=None,
    linear_window=4,
    use_revin=False,
    linear_shared_weights=True,
    use_seasonal_decomp=True,
    recon_mask_skip_all=args.recon_mask_skip_all,
    recon_mask_max_seq_len=args.recon_mask_max_seq_len,
    recon_mask_drop_seq=args.recon_mask_drop_seq,
    recon_mask_drop_standard=args.recon_mask_drop_standard,
    recon_mask_drop_full=args.recon_mask_drop_full,
    distribution_output=args.distribution_output,)
    data_module = FundamentalsDataModule(
        dataset_kwargs={
            "context_length": args.context_points,
            "prediction_length": args.target_points,
            "test_split":0.15,
            "val_split":0.15,
        },
        batch_size=16,
        workers=2,
    )
    inv_scaler = data_module.series.reverse_scaling
    scaler = data_module.series.apply_scaling
    forecaster.set_inv_scaler(inv_scaler)
    forecaster.set_scaler(scaler)
    plot_var_names = data_module.series.target_cols
    plot_var_idxs = [i for i in range(len(plot_var_names))]

    # Callbacks
    callbacks = [SpaceTimeformerProgressBar()]
    test_samples = next(iter(data_module.test_dataloader()))
    test_identifiers = data_module.series.test_group_identifiers[:16]

    callbacks.append(
        PredictionPlotterCallback(
            test_samples,
            var_idxs=plot_var_idxs,
            var_names=plot_var_names,
            pad_val=None,
            identifiers=test_identifiers,
            total_samples=8,
        )
    )
        

    callbacks.append(
        AttentionMatrixCallback(
            test_samples,
            layer=0,
            total_samples=8,
        )
    )   
    
    callbacks.append(
        ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            auto_insert_metric_name=True,
        )
    )
    
    callbacks.append(
            EarlyStopping(
                monitor="val/loss",
                patience=3,
            )
        )
    
    
    trainer = pl.Trainer(
        max_epochs=15,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=args.accumulate,
        sync_batchnorm=True,
    )
    # Train
    trainer.fit(forecaster, datamodule=data_module)

    # Test
    trainer.test(datamodule=data_module)
    
    # Predict (only here as a demo and test)
    # forecaster.to("cuda")
    # xc, yc, xt, _ = test_samples
    # yt_pred = forecaster.predict(xc, yc, xt)
    experiment.finish()
    
def test_model(args):
    """
    Function to test the Spacetimeformer model.

    Args:
        args: An object containing the arguments for the model.

    Returns:
        None
    """
    forecaster = Spacetimeformer_Forecaster(d_x=2, d_yc=35, d_yt = 1, categorical_dict_sizes = [11,25,74,163], 
    max_seq_len=args.context_points + args.target_points,
    start_token_len=0,
    attn_factor=0,
    d_model=96,
    d_queries_keys=32,
    d_values=32,
    n_heads=8,
    e_layers=2,
    d_layers=2,
    d_ff=800,
    dropout_emb=0.2,
    dropout_attn_out=0,
    dropout_attn_matrix=0,
    dropout_qkv=0,
    dropout_ff=0.2,
    pos_emb_type="t2v",
    use_final_norm=True,
    global_self_attn="performer",
    local_self_attn="performer",
    global_cross_attn="performer",
    local_cross_attn="performer",
    performer_kernel="softmax",
    performer_redraw_interval=150,
    attn_time_windows=1,
    use_shifted_time_windows=args.use_shifted_time_windows,
    norm="batch",
    activation="gelu",
    init_lr=1e-8,
    base_lr=5e-4,
    warmup_steps=20,
    decay_factor=0.9, # https://stackoverflow.com/questions/67746083/setting-a-minimum-learning-rate-on-reduce-on-plateau
    initial_downsample_convs=args.initial_downsample_convs,
    intermediate_downsample_convs=args.intermediate_downsample_convs,
    embed_method=args.embed_method,
    l2_coeff=1e-6,
    loss=args.loss,
    class_loss_imp=0.1,
    recon_loss_imp=0.1,
    time_emb_dim=8,
    categorical_embedding_dim=8,
    null_value=-1e2,
    pad_value=None,
    linear_window=4,
    use_revin=False,
    linear_shared_weights=True,
    use_seasonal_decomp=True,
    recon_mask_skip_all=args.recon_mask_skip_all,
    recon_mask_max_seq_len=args.recon_mask_max_seq_len,
    recon_mask_drop_seq=args.recon_mask_drop_seq,
    recon_mask_drop_standard=args.recon_mask_drop_standard,
    recon_mask_drop_full=args.recon_mask_drop_full,
    distribution_output="skewnormal",)
    data_module = FundamentalsDataModule(
        dataset_kwargs={
            "context_length": args.context_points,
            "prediction_length": args.target_points,
            "test_split":0.6,
            "val_split":0.1,
        },
        batch_size=16,
        workers=2,
    )
    inv_scaler = data_module.series.reverse_scaling
    scaler = data_module.series.apply_scaling
    forecaster.set_inv_scaler(inv_scaler)
    forecaster.set_scaler(scaler)
    
    profiler = PyTorchProfiler(dirpath="profiling", filename="output", export_to_chrome=False, profile_memory=True, record_shapes=True)
    
    # Callbacks
    callbacks = [SpaceTimeformerProgressBar()]
    
    trainer = pl.Trainer(
        max_epochs=1,
        # gpus=18,
        logger=None,
        callbacks=callbacks,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=args.accumulate,
        sync_batchnorm=True,
        profiler=profiler,
        detect_anomaly=True,
    )
    # Train
    trainer.fit(forecaster, datamodule=data_module)

    # Test
    trainer.validate(datamodule=data_module)


# def param_optimizer(args):
#     # Define the search space for each hyperparameter
#     search_space = {
#         "linear_window": hp.choice("linear_window", np.arange(0, 10, dtype=int)),
#         "linear_shared_weights": hp.choice("linear_shared_weights", [True, False]),
#         "d_model": hp.choice("d_model", [32, 48, 64, 96, 128, 256, 512]),
#         "d_qk": hp.choice("d_qk", [32, 48, 64, 96, 128, 256, 512]),
#         "d_v": hp.choice("d_v", [32, 48, 64, 96, 128, 256, 512]),
#         "d_ff": hp.quniform("d_ff", 200, 1600, 100),
#     }


#     # Define an objective function for hyperparameter optimization
#     def objective(search_params):
#         params = {**vars(args), **search_params}
#         forecaster = Spacetimeformer_Forecaster(d_x=2, d_yc=35, d_yt = 1, categorical_dict_sizes = [11,25,74,163], 
#         max_seq_len=params["context_points"] + params["target_points"],
#         start_token_len=0,
#         attn_factor=0,
#         d_model=int(params["d_model"]),
#         d_queries_keys=int(params["d_qk"]),
#         d_values=int(params["d_v"]),
#         n_heads=8,
#         e_layers=2,
#         d_layers=2,
#         d_ff=int(params["d_ff"]),
#         dropout_emb=0.2,
#         dropout_attn_out=0,
#         dropout_attn_matrix=0,
#         dropout_qkv=0,
#         dropout_ff=0.2,
#         pos_emb_type="t2v",
#         use_final_norm=True,
#         global_self_attn="performer",
#         local_self_attn="performer",
#         global_cross_attn="performer",
#         local_cross_attn="performer",
#         performer_kernel="softmax",
#         performer_redraw_interval=150,
#         attn_time_windows=1,
#         use_shifted_time_windows=params["use_shifted_time_windows"],
#         norm="batch",
#         activation="gelu",
#         init_lr=1e-8,
#         base_lr=5e-4,
#         warmup_steps=20,
#         decay_factor=0.9, # https://stackoverflow.com/questions/67746083/setting-a-minimum-learning-rate-on-reduce-on-plateau
#         initial_downsample_convs=params["initial_downsample_convs"],
#         intermediate_downsample_convs=params["intermediate_downsample_convs"],
#         embed_method=params["embed_method"],
#         l2_coeff=1e-6,
#         loss=params["loss"],
#         class_loss_imp=0.1,
#         recon_loss_imp=0.1,
#         time_emb_dim=8,
#         categorical_embedding_dim=8,
#         null_value=-1e2,
#         pad_value=None,
#         linear_window=int(params["linear_window"]),
#         use_revin=False,
#         linear_shared_weights=params["linear_shared_weights"],
#         use_seasonal_decomp=True,
#         recon_mask_skip_all=params["recon_mask_skip_all"],
#         recon_mask_max_seq_len=params["recon_mask_max_seq_len"],
#         recon_mask_drop_seq=params["recon_mask_drop_seq"],
#         recon_mask_drop_standard=params["recon_mask_drop_standard"],
#         recon_mask_drop_full=params["recon_mask_drop_full"],)
#         data_module = FundamentalsDataModule(
#             dataset_kwargs={
#                 "context_length": params["context_points"],
#                 "prediction_length": params["target_points"],
#                 "test_split":0.8,
#                 "val_split":0.1,
#             },
#             batch_size=8,
#             workers=8,
#         )
#         inv_scaler = data_module.series.reverse_scaling
#         scaler = data_module.series.apply_scaling
#         forecaster.set_inv_scaler(inv_scaler)
#         forecaster.set_scaler(scaler)
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         forecaster = forecaster.to(device)
            
#         trainer = pl.Trainer(
#             max_epochs=5,
#             # gpus=18,
#             logger=None,
#             accelerator=device.type,
#             gradient_clip_val=0.5,
#             gradient_clip_algorithm="norm",
#             accumulate_grad_batches=params["accumulate"],
#             sync_batchnorm=True,
#             profiler="advanced",
#             detect_anomaly=True,
#         )
#         # Train
#         trainer.fit(forecaster, datamodule=data_module)

#         # Test
#         results = trainer.validation(datamodule=data_module)[0]
        
#         return {**results, 'status': STATUS_OK}

#     search_space["embed_method"] = hp.choice("embed_method", [args.embed_method])

#     # Initialize a Trials object to keep track of results
#     trials = Trials()

#     # Perform hyperparameter optimization
#     best_hyperparams = fmin(
#         fn=objective,
#         space=search_space,
#         algo=tpe.suggest,
#         max_evals=20,  # Adjust the number of evaluations as needed
#         trials=trials,
#         rstate=rstate,
#     )
#     print("Best hyperparameters:", best_hyperparams)
    


    
if __name__ == "__main__":
    # CLI
    parser = create_parser()
    args = parser.parse_args()
    if args.execution_type == "train":
        trainer(args)
    else:
        test_model(args)
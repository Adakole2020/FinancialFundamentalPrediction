from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import numpy as np

from spacetimeformer.spacetimeformer_model import Spacetimeformer_Forecaster
from spacetimeformer.data import FundamentalsDataModule, FundamentalsCSVSeries
from argparse import ArgumentParser
import sys

import pytorch_lightning as pl

pl.seed_everything(123, workers=True)
rstate = np.random.default_rng(123)

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

def create_parser():
    parser = ArgumentParser()

    FundamentalsCSVSeries.add_cli(parser)
    FundamentalsDataModule.add_cli(parser)

    Spacetimeformer_Forecaster.add_cli(parser)

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--patience", type=int, default=5)

    if len(sys.argv) > 3 and sys.argv[3] == "-h":
        parser.print_help()
        sys.exit(0)

    return parser

def param_optimizer(args):
    # Define the search space for each hyperparameter
    search_space = {
        "time_mask_start": hp.choice("time_mask_start", np.arange(1, 100, dtype=int)),
        "time_mask_end": hp.choice("time_mask_end", np.arange(1, 100, dtype=int)),
        "time_mask_anneal_steps": hp.quniform("time_mask_anneal_steps", 100, 5000, 100),
        "time_mask_loss": hp.choice("time_mask_loss", [True, False]),
        "l2_coeff": hp.loguniform("l2_coeff", -10, 0),
        "learning_rate": hp.loguniform("learning_rate", -7, -2),
        "grad_clip_norm": hp.choice("grad_clip_norm", [0, hp.loguniform("grad_clip_norm_val", -3, 3)]),
        "linear_window": hp.choice("linear_window", np.arange(0, 10, dtype=int)),
        "linear_shared_weights": hp.choice("linear_shared_weights", [True, False]),
        "use_seasonal_decomp": hp.choice("use_seasonal_decomp", [True, False]),
        "batch_size": hp.choice("batch_size", [1, 2, 4, 8, 16, 32, 64, 128]),
        "d_model": hp.choice("d_model", [32, 48, 64, 96, 128, 256, 512]),
        "d_qk": hp.choice("d_qk", [32, 48, 64, 96, 128, 256, 512]),
        "d_v": hp.choice("d_v", [32, 48, 64, 96, 128, 256, 512]),
        "n_heads": hp.choice("n_heads", np.arange(2, 16, dtype=int)),
        "enc_layers": hp.choice("enc_layers", np.arange(1, 12, dtype=int)),
        "dec_layers": hp.choice("dec_layers", np.arange(1, 12, dtype=int)),
        "d_ff": hp.quniform("d_ff", 200, 1600, 100),
        "dropout_emb": hp.uniform("dropout_emb", 0, 0.5),
        "dropout_attn_matrix": hp.uniform("dropout_attn_matrix", 0, 0.5),
        "dropout_qkv": hp.uniform("dropout_qkv", 0, 0.5),
        "dropout_ff": hp.uniform("dropout_ff", 0, 0.5),
        "dropout_attn_out": hp.uniform("dropout_attn_out", 0, 0.5),
        "global_self_attn": hp.choice("global_self_attn", ["full", "prob", "performer", "none"]),
        "global_cross_attn": hp.choice("global_cross_attn", ["full", "performer", "none"]),
        "local_self_attn": hp.choice("local_self_attn", ["full", "prob", "performer", "none"]),
        "local_cross_attn": hp.choice("local_cross_attn", ["full", "performer", "none"]),
        "activation": hp.choice("activation", ["relu", "gelu"]),
        "norm": hp.choice("norm", ["layer", "batch", "scale", "power", "none"]),
        "init_lr": hp.loguniform("init_lr", -20, -9),
        "base_lr": hp.loguniform("base_lr", -7, -3),
        "warmup_steps": hp.quniform("warmup_steps", 0, 10000, 100),
        "decay_factor": hp.uniform("decay_factor", 0.1, 0.9),
        "recon_loss_imp": hp.uniform("recon_loss_imp", 0, 1),
        "time_emb_dim": hp.choice("time_emb_dim", [2, 4, 6, 8, 10]),
        "categorical_embedding_dim": hp.choice("categorical_embedding_dim", [2, 4, 8, 16, 32]),
        "performer_kernel": hp.choice("performer_kernel", ["softmax", "relu"]),
        "performer_redraw_interval": hp.quniform("performer_redraw_interval", 50, 500, 25),
        "pos_emb_type": hp.choice("pos_emb_type", ["abs", "t2v"]),
        "no_final_norm": hp.choice("no_final_norm", [True, False]),
    }


    # Define an objective function for hyperparameter optimization
    def objective(search_params):
        params = {**vars(args), **search_params}
        forecaster = Spacetimeformer_Forecaster(d_x=2, d_yc=35, d_yt = 1, categorical_dict_sizes = [11,25,74,163], 
        max_seq_len=params["context_points"] + params["target_points"],
        start_token_len=0,
        attn_factor=params["attn_factor"],
        d_model=int(params["d_model"]),
        d_queries_keys=int(params["d_qk"]),
        d_values=int(params["d_v"]),
        n_heads=int(params["n_heads"]),
        e_layers=int(params["enc_layers"]),
        d_layers=int(params["dec_layers"]),
        d_ff=int(params["d_ff"]),
        dropout_emb=params["dropout_emb"],
        dropout_attn_out=params["dropout_attn_out"],
        dropout_attn_matrix=params["dropout_attn_matrix"],
        dropout_qkv=params["dropout_qkv"],
        dropout_ff=params["dropout_ff"],
        pos_emb_type=params["pos_emb_type"],
        use_final_norm=not params["no_final_norm"],
        global_self_attn=params["global_self_attn"],
        local_self_attn=params["local_self_attn"],
        global_cross_attn=params["global_cross_attn"],
        local_cross_attn=params["local_cross_attn"],
        performer_kernel=params["performer_kernel"],
        performer_redraw_interval=params["performer_redraw_interval"],
        attn_time_windows=params["attn_time_windows"],
        use_shifted_time_windows=params["use_shifted_time_windows"],
        norm=params["norm"],
        activation=params["activation"],
        init_lr=params["init_lr"],
        base_lr=params["base_lr"],
        warmup_steps=int(params["warmup_steps"]),
        decay_factor=params["decay_factor"],
        initial_downsample_convs=params["initial_downsample_convs"],
        intermediate_downsample_convs=params["intermediate_downsample_convs"],
        embed_method=params["embed_method"],
        l2_coeff=params["l2_coeff"],
        loss=params["loss"],
        class_loss_imp=params["class_loss_imp"],
        recon_loss_imp=params["recon_loss_imp"],
        time_emb_dim=params["time_emb_dim"],
        categorical_embedding_dim=params["categorical_embedding_dim"],
        null_value=np.NAN,
        pad_value=None,
        linear_window=int(params["linear_window"]),
        use_revin=params["use_revin"],
        linear_shared_weights=params["linear_shared_weights"],
        use_seasonal_decomp=params["use_seasonal_decomp"],
        recon_mask_skip_all=params["recon_mask_skip_all"],
        recon_mask_max_seq_len=params["recon_mask_max_seq_len"],
        recon_mask_drop_seq=params["recon_mask_drop_seq"],
        recon_mask_drop_standard=params["recon_mask_drop_standard"],
        recon_mask_drop_full=params["recon_mask_drop_full"],)
        data_module = FundamentalsDataModule(
            dataset_kwargs={
                "context_length": params["context_points"],
                "prediction_length": params["target_points"],
            },
            batch_size=params["batch_size"],
            workers=params["workers"],
        )
        inv_scaler = data_module.series.reverse_scaling
        scaler = data_module.series.apply_scaling
        forecaster.set_inv_scaler(inv_scaler)
        forecaster.set_scaler(scaler)
            
        trainer = pl.Trainer(
            max_epochs=20,
            # gpus=18,
            logger=None,
            accelerator='cpu',
            gradient_clip_val=params["grad_clip_norm"],
            gradient_clip_algorithm="norm",
            overfit_batches=20 if params["debug"] else 0,
            accumulate_grad_batches=params["accumulate"],
            sync_batchnorm=True,
        )
        # Train
        trainer.fit(forecaster, datamodule=data_module)

        # Test
        results = trainer.test(datamodule=data_module, ckpt_path="best")[0]
        
        return {**results, 'status': STATUS_OK}

    search_space["embed_method"] = hp.choice("embed_method", [args.embed_method])

    # Initialize a Trials object to keep track of results
    trials = Trials()

    # Perform hyperparameter optimization
    best_hyperparams = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=100,  # Adjust the number of evaluations as needed
        trials=trials,
        rstate=rstate,
    )
    print("Best hyperparameters:", best_hyperparams)
    


    
if __name__ == "__main__":
    # CLI
    parser = create_parser()
    args = parser.parse_args()

    param_optimizer(args)
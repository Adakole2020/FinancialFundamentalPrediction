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
        "linear_window": hp.choice("linear_window", np.arange(0, 10, dtype=int)),
        "linear_shared_weights": hp.choice("linear_shared_weights", [True, False]),
        "use_seasonal_decomp": hp.choice("use_seasonal_decomp", [True, False]),
        "d_model": hp.choice("d_model", [32, 48, 64, 96, 128, 256, 512]),
        "d_qk": hp.choice("d_qk", [32, 48, 64, 96, 128, 256, 512]),
        "d_v": hp.choice("d_v", [32, 48, 64, 96, 128, 256, 512]),
        "n_heads": hp.choice("n_heads", np.arange(2, 16, dtype=int)),
        "enc_layers": hp.choice("enc_layers", np.arange(1, 12, dtype=int)),
        "dec_layers": hp.choice("dec_layers", np.arange(1, 12, dtype=int)),
        "d_ff": hp.quniform("d_ff", 200, 1600, 100),
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
        use_shifted_time_windows=params["use_shifted_time_windows"],
        norm="batch",
        activation="gelu",
        init_lr=1e-8,
        base_lr=5e-4,
        warmup_steps=20,
        decay_factor=0.9, # https://stackoverflow.com/questions/67746083/setting-a-minimum-learning-rate-on-reduce-on-plateau
        initial_downsample_convs=params["initial_downsample_convs"],
        intermediate_downsample_convs=params["intermediate_downsample_convs"],
        embed_method=params["embed_method"],
        l2_coeff=1e-6,
        loss=params["loss"],
        class_loss_imp=0.1,
        recon_loss_imp=0.1,
        time_emb_dim=8,
        categorical_embedding_dim=8,
        null_value=-1e2,
        pad_value=None,
        linear_window=int(params["linear_window"]),
        use_revin=False,
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
            batch_size=32,
            workers=8,
        )
        inv_scaler = data_module.series.reverse_scaling
        scaler = data_module.series.apply_scaling
        forecaster.set_inv_scaler(inv_scaler)
        forecaster.set_scaler(scaler)
            
        trainer = pl.Trainer(
            max_epochs=5,
            # gpus=18,
            logger=None,
            accelerator='cpu',
            gradient_clip_val=0.5,
            gradient_clip_algorithm="norm",
            overfit_batches=20 if params["debug"] else 0,
            accumulate_grad_batches=params["accumulate"],
            sync_batchnorm=True,
        )
        # Train
        trainer.fit(forecaster, datamodule=data_module)

        # Test
        results = trainer.test(datamodule=data_module)[0]
        
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
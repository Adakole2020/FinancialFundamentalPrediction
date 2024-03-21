from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import numpy as np

from spacetimeformer.train import create_parser
from spacetimeformer.spacetimeformer_model import Spacetimeformer_Forecaster
from spacetimeformer.data import FundamentalsDataModule

def param_optimizer(args):
    # Define the search space for each hyperparameter
    search_space = {
        "time_mask_start": hp.quniform("time_mask_start", 1, 100, 1),
        "time_mask_end": hp.quniform("time_mask_end", 1, 100, 1),
        "time_mask_anneal_steps": hp.quniform("time_mask_anneal_steps", 100, 5000, 100),
        "time_mask_loss": hp.choice("time_mask_loss", [True, False]),
        "l2_coeff": hp.loguniform("l2_coeff", -10, 0),
        "learning_rate": hp.loguniform("learning_rate", -7, -2),
        "grad_clip_norm": hp.choice("grad_clip_norm", [0, hp.loguniform("grad_clip_norm_val", -3, 3)]),
        "linear_window": hp.quniform("linear_window", 0, 10, 1),
        "linear_shared_weights": hp.choice("linear_shared_weights", [True, False]),
        "use_seasonal_decomp": hp.choice("use_seasonal_decomp", [True, False]),
        "batch_size": hp.choice("batch_size", [1, 2, 4, 8, 16, 32, 64, 128]),
        "start_token_len": hp.quniform("start_token_len", 0, 10, 1),
        "d_model": hp.choice("d_model", [32, 48, 64, 96, 128, 256, 512]),
        "d_qk": hp.choice("d_qk", [32, 48, 64, 96, 128, 256, 512]),
        "d_v": hp.choice("d_v", [32, 48, 64, 96, 128, 256, 512]),
        "n_heads": hp.quniform("n_heads", 2, 16, 1),
        "enc_layers": hp.quniform("enc_layers", 1, 12, 1),
        "dec_layers": hp.quniform("dec_layers", 1, 12, 1),
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
        "initial_downsample_convs": hp.quniform("initial_downsample_convs", 0, 5, 1),
        "recon_loss_imp": hp.uniform("recon_loss_imp", 0, 1),
        "intermediate_downsample_convs": hp.quniform("intermediate_downsample_convs", 0, 5, 1),
        "time_emb_dim": hp.choice("time_emb_dim", [2, 4, 6, 8, 10]),
        "categorical_embedding_dim": hp.choice("categorical_embedding_dim", [2, 4, 8, 16, 32]),
        "performer_kernel": hp.choice("performer_kernel", ["softmax", "relu"]),
        "performer_redraw_interval": hp.quniform("performer_redraw_interval", 50, 500, 25),
        "pos_emb_type": hp.choice("pos_emb_type", ["abs", "t2v"]),
        "no_final_norm": hp.choice("no_final_norm", [True, False]),
        "recon_mask_skip_all": hp.uniform("recon_mask_skip_all", 0, 1),
        "recon_mask_max_seq_len": hp.quniform("recon_mask_max_seq_len", 1, 10, 1),
        "recon_mask_drop_seq": hp.uniform("recon_mask_drop_seq", 0, 0.5),
        "recon_mask_drop_standard": hp.uniform("recon_mask_drop_standard", 0, 0.5),
        "recon_mask_drop_full": hp.uniform("recon_mask_drop_full", 0, 0.5),
    }


    # Define an objective function for hyperparameter optimization
    def objective(search_params):
        params = {**args, **search_params}
        if hasattr(config, "context_points") and hasattr(config, "target_points"):
            max_seq_len = params.context_points + params.target_points
        elif hasattr(config, "max_len"):
            max_seq_len = params.max_len
        else:
            raise ValueError("Undefined max_seq_len")
        forecaster = Spacetimeformer_Forecaster(x_dim=2, yc_dim=35, yt_dim = 1, categorical_dict_sizes = [11,25,74,163], **params)
        data_module = FundamentalsDataModule(
            dataset_kwargs={
                "context_length": params.context_points,
                "prediction_length": params.target_points,
            },
            batch_size=config.batch_size,
        )
        inv_scaler = dset.reverse_scaling
        scaler = dset.apply_scaling
        forecaster.set_inv_scaler(inv_scaler)
        forecaster.set_scaler(scaler)
        forecaster.set_null_values(np.NAN)
            
        trainer = pl.Trainer(
            max_epochs=20,
            gpus=2,
            logger=None,
            accelerator="dp",
            gradient_clip_val=params.grad_clip_norm,
            gradient_clip_algorithm="norm",
            overfit_batches=20 if params.debug else 0,
            accumulate_grad_batches=params.accumulate,
            sync_batchnorm=True,
            limit_val_batches=params.limit_val_batches,
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
        max_evals=10,  # Adjust the number of evaluations as needed
        trials=trials
    )
    print("Best hyperparameters:", best_hyperparams)
    
    
if __name__ == "__main__":
    # CLI
    parser = create_parser()
    args = parser.parse_args()

    param_optimizer(args)
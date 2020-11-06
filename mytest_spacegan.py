from spacegan_wrapper import SpaceganWrapper
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from spacegan_utils import gaussian, rmse, mie

# problem configuration
prob_config = {
    "epochs": 3000,
    "batch_size": 100,
    "device": torch.device("cuda"),
    "noise_type": gaussian,  # type of noise and dimension used
    "noise_params": None,  # other params for noise (loc, scale, etc.) pass as a dict
    "scale_x": StandardScaler(),  # a sklearn.preprocessing scaling method
    "scale_y": StandardScaler(),  # a sklearn.preprocessing scaling method
    "print_results": False,
    # additional Generator params
    "gen_opt": torch.optim.SGD,
    "gen_opt_params": {"lr": 0.01},
    # additional Discriminator params
    "disc_opt": torch.optim.SGD,
    "disc_opt_params": {"lr": 0.01},
    # loss function
    "adversarial_loss": torch.nn.BCELoss()
}

# checkpointing configuration
check_config = {
    "check_interval": 1000,  # for model checkpointing
    "generate_image": False,
    "n_samples": 50,
    "perf_metrics": {"RMSE": rmse,
                     "MIE": mie,
                     },
    "pf_metrics_setting": {
        "RMSE": {"metric_level": "agg_metrics",
             "rank_function": np.argmin,
             "agg_function": lambda x: np.array(x)
             },
        "MIE": {"metric_level": "agg_metrics",
                "rank_function": np.argmin,
                "agg_function": lambda x: np.array(x)
               },
    },
    "agg_funcs": {"avg": np.mean,
                  "std": np.std
                 },
    "sample_metrics": False,
    "agg_metrics": True
}

spgan_wrapper = SpaceganWrapper(prob_config, check_config, model_save_prefix='saved_models/noaa/')
spgan_wrapper.build_dataset("data/raw_data.csv", rows=101, neighbours=50, output_vars="salinity")
spgan_wrapper.build_gan()
spgan_wrapper.fit()
spgan_wrapper.select_best_generators()

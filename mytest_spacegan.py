from spacegan_wrapper import SpaceganWrapper
from spacegan_utils import gaussian, rmse, mie
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import pdb
import pandas as pd
from tsgan_metrics import TsganMetrics 
from tsgan_wrapper import TsganWrapper


def one_run(cond_vars_arr, output_vars_arr):
    # problem configuration
    prob_config = {
        "epochs": 1000,
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

    spgan_wrapper = SpaceganWrapper(prob_config, check_config, model_save_prefix='saved_models/noaa/',fig_save_prefix='img/noaa/')
    df = spgan_wrapper.build_dataset("data/raw_data.csv", rows=101, neighbours=5, output_vars=output_vars_arr,cond_vars=cond_vars_arr)
    # plot(self, img_name, title, generated_seq, dataframe=None):

    # build_dataset need to return dataframe for this to work
    spgan_wrapper.plot(img_name='p1_me.png',title="Observed",generated_seq=df[["salinity"]].values.astype(float))

    spgan_wrapper.build_gan()
    spgan_wrapper.fit()
    spgan_wrapper.select_best_generators()
    ax1_checkmetrics = spgan_wrapper.load_model("grid_checkmetrics.pkl.gz") # or should select_best_generators return this dataframe?

    #load mie selection results
    gan_samples_df = spgan_wrapper.load_model("grid_MIE.pkl.gz")
    gen_seq = gan_samples_df[["sample_" + str(x) for x in range(20)]].mean(axis=1) # 50 instead of 20 ??
    # spgan_wrapper.plot(img_name='p2_me.png',title="SpaceGAN - Best MIE",generated_seq=gen_seq,dataframe=ax1_checkmetrics)

    #load rmse selection results
    gan_samples_df = spgan_wrapper.load_model("grid_RMSE.pkl.gz")
    gen_seq = gan_samples_df[["sample_" + str(x) for x in range(20)]].mean(axis=1) 
    # spgan_wrapper.plot(img_name='p3_me.png',title="SpaceGAN - Best RMSE",generated_seq=gen_seq,dataframe=ax1_checkmetrics)

    return df, gen_seq  # evaluated using RMSE
    # iteration = 1000
    # gan_samples_df = spgan_wrapper.generate(iteration) # this returns the dataframe from the given iteration
    # gen_seq = gan_samples_df[["sample_" + str(x) for x in range(1)]].mean(axis=1)  # why is range just 1 ?
    # spgan_wrapper.plot(
    #     img_name='p4_me.png',
    #     title="SpaceGAN - Iteration " + str(iteration),
    #     generated_seq=gen_seq
    # )

    # what is the point of p5 ?? it's exactly the same with p4

# Should have a cleaner and more general way to do this
df, gen_seq_5 = one_run(cond_vars_arr=['unix_time', 'depth', 'conductivity', 'density', 'temperature'],output_vars_arr=['salinity'])
_, gen_seq_4 = one_run(cond_vars_arr=['unix_time', 'depth', 'conductivity', 'density', 'salinity'], output_vars_arr=['temperature'])
_, gen_seq_3 = one_run(cond_vars_arr=['unix_time', 'depth', 'conductivity', 'temperature', 'salinity'], output_vars_arr=['density'])
_, gen_seq_2 = one_run(cond_vars_arr=['unix_time', 'depth', 'density', 'temperature', 'salinity'], output_vars_arr=['conductivity'])
_, gen_seq_1 = one_run(cond_vars_arr=['unix_time', 'conductivity', 'density', 'temperature', 'salinity'],output_vars_arr=['depth'])
_, gen_seq_0 = one_run(cond_vars_arr=['depth', 'conductivity', 'density', 'temperature', 'salinity'],output_vars_arr=['unix_time'])


synthetic_df = pd.concat([gen_seq_0,gen_seq_1,gen_seq_2,gen_seq_3,gen_seq_4,gen_seq_5], axis=1)

dataX = df.to_numpy()
dataX_hat = synthetic_df.to_numpy()

t0 = TsganWrapper(dataX)
dataX = t0.build_dataset()
t1 = TsganWrapper(dataX_hat)
dataX_hat = t1.build_dataset()

pdb.set_trace()
tsgan_metrics = TsganMetrics(1)
tsgan_metrics.compute_discriminative(dataX, dataX_hat)
tsgan_metrics.compute_predictive(dataX, dataX_hat)
results = tsgan_metrics.mean_std()
print(results)


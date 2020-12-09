import logging
import typing
import numpy as np

from examples.example_utils import delete_experiments_folder
from smallab.experiment_types.experiment import Experiment
from smallab.runner.runner import ExperimentRunner
from smallab.specification_generator import SpecificationGenerator
from smallab.runner_implementations.main_process_runner import MainRunner

from spacegan_wrapper import SpaceganWrapper
from tsgan_wrapper import TsganWrapper
from tsgan_metrics import TsganMetrics 
from visualizers import Visualizers 
import pandas as pd

class TsganExperiment(Experiment):
    def one_run(self, cond_vars_arr, output_vars_arr):
        csv_files = ["data/mexico/traj_35.csv","data/mexico/traj_1.csv"]
        spgan_wrapper = SpaceganWrapper(output_vars=output_vars_arr,cond_vars=cond_vars_arr) #prob_config, check_config
        df = spgan_wrapper.build_dataset(csv_files)
        # plot(self, img_name, title, generated_seq, dataframe=None):

        # build_dataset need to return dataframe for this to work
        spgan_wrapper.plot(img_name='p1_me.png',title="Observed",generated_seq=df[["salinity"]].values.astype(float))

        spgan_wrapper.build_gan()
        spgan_wrapper.fit()
        spgan_wrapper.select_best_generators()
        ax1_checkmetrics = spgan_wrapper.load_model("grid_checkmetrics.pkl.gz") # or should select_best_generators return this dataframe?

        # #load mie selection results
        gan_samples_df = spgan_wrapper.load_model("grid_MIE.pkl.gz")
        gen_seq = gan_samples_df[["sample_" + str(x) for x in range(50)]].mean(axis=1) # 50 instead of 20 ??
        spgan_wrapper.plot(img_name='p2_me.png',title="SpaceGAN - Best MIE",generated_seq=gen_seq,dataframe=ax1_checkmetrics)

        # #load rmse selection results
        gan_samples_df = spgan_wrapper.load_model("grid_RMSE.pkl.gz")
        gen_seq = gan_samples_df[["sample_" + str(x) for x in range(50)]].mean(axis=1) 
        spgan_wrapper.plot(img_name='p3_me.png',title="SpaceGAN - Best RMSE",generated_seq=gen_seq,dataframe=ax1_checkmetrics)

        return df, gen_seq  # evaluated using RMSE


    def main(self, specification:typing.Dict) -> typing.Dict:
        # pass
        param_arr = ['unix_time', 'depth', 'conductivity', 'density', 'temperature','salinity']
        synthetic_df = None
        for i in range(len(param_arr)):
            cond_vars = param_arr.copy()
            cond_vars.remove(param_arr[i])
            df, gen_seq = self.one_run(cond_vars_arr=cond_vars, output_vars_arr=[param_arr[i]])
            if(i==0):
                synthetic_df = pd.concat([gen_seq,df.iloc[:,1],df.iloc[:,2]],axis=1)
            else:
                synthetic_df = pd.concat([synthetic_df, gen_seq], axis=1)

        # dataX = df.to_numpy() #need to change this to unseen data
        logging.getLogger(self.get_logger_name()).info("Finished space gan training.")

        dataX_hat = synthetic_df.to_numpy()
        unseen_wrapper = SpaceganWrapper()
        unseen_df = unseen_wrapper.build_dataset(csv_files=['data/mexico/traj_13.csv'])
        dataX = unseen_df.to_numpy()

        dataX_hat = dataX_hat[:min(len(dataX_hat), len(dataX))]
        dataX = dataX[:min(len(dataX_hat), len(dataX))]


        t0 = TsganWrapper(dataX=dataX) #does this work?
        dataX = t0.build_dataset()
        t1 = TsganWrapper(dataX=dataX_hat)
        dataX_hat = t1.build_dataset()

        # dataX = dataX[:min(len(dataX_hat), len(dataX))]
        # dataX_hat = dataX_hat[:min(len(dataX_hat), len(dataX))]

        tsgan_metrics = TsganMetrics(1)
        tsgan_metrics.compute_discriminative(dataX, dataX_hat)
        tsgan_metrics.compute_predictive(dataX, dataX_hat)
        results = tsgan_metrics.mean_std()

        return {"discriminative score mean": results[0], "predictive score mean": results[2]}


    

generation_specification = {  # just trying out random values for testing
    "total_iterations": [1], 
    "sub_iterations": [2],
    "data_size": [965],
    "max_seq_length": [12],
    "iterations": [10001],
    "batch_size": [128]
}


# Call the generate method. Will create the cross product.
specifications = SpecificationGenerator().generate(generation_specification)
print(specifications)

expt = TsganExperiment()
name = "spacegan" #+expt.get_hash()
runner = ExperimentRunner()
runner.run(name, specifications, expt, specification_runner=MainRunner(), propagate_exceptions=True)


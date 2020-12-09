import logging
import typing
import numpy as np

from examples.example_utils import delete_experiments_folder
from smallab.experiment_types.experiment import Experiment
from smallab.runner.runner import ExperimentRunner
from smallab.specification_generator import SpecificationGenerator
from smallab.runner_implementations.main_process_runner import MainRunner

from tsgan_wrapper import TsganWrapper
from tsgan_metrics import TsganMetrics 
from visualizers import Visualizers 

# Same experiment as before
class TsganExperiment(Experiment):
    def main(self, specification:typing.Dict) -> typing.Dict:

        # data loading: using cashed data from csv file
        # x = np.loadtxt('data/conv_loc_time_new.csv', delimiter=',') #, skiprows=1
        # x = np.delete(x, 0, axis=1)
        # x = x[:specification["data_size"],:]
        train_traj = ['data/mexico/traj_35.csv','data/mexico/traj_1.csv']
        wrapper = TsganWrapper()
        seq_length = min(specification["max_seq_length"], int(specification["data_size"]/3))
        dataX = wrapper.build_dataset(csv_files=train_traj, seq_length)
        
        # loading unseen data: to test discriminative score and predivtive score
        # unseen = np.loadtxt('data/unseen_dataX_new.csv', delimiter=',')
        # unseen = np.delete(unseen, 0, axis=1)
        unseen_wrap = TsganWrapper()
        dataX_disctest = unseen_wrap.build_dataset(csv_files=['data/mexico/traj_13.csv'],seq_length)
        dataX_disctest = dataX_disctest[:len(dataX), :]

        logging.getLogger(self.get_logger_name()).info("Dataset is ready.")

        wrapper.set_tgan_parameters('iterations', specification['iterations'])
        wrapper.set_tgan_parameters('batch_size', specification['batch_size'])
        wrapper.set_tgan_parameters('hidden_dim', len(dataX[0][0,:]) * 4)
        wrapper.set_tgan_parameters('z_dim', len(dataX[0][0,:]))

        metrics = TsganMetrics(specification['sub_iterations'])
        prefix_str = self.get_logger_name()

        for it in range(specification['total_iterations']):
            wrapper.fit(logger=self.get_logger_name(), filename=prefix_str)
            dataX_hat = wrapper.generate()
            logging.getLogger(self.get_logger_name()).info("Computing discriminative score...")
            metrics.compute_discriminative(dataX_disctest, dataX_hat)
            logging.getLogger(self.get_logger_name()).info("Discriminative score complete.")
            logging.getLogger(self.get_logger_name()).info("Computing predictive score...")
            metrics.compute_predictive(dataX_disctest, dataX_hat)
            logging.getLogger(self.get_logger_name()).info("Predictive score complete.")

        results = metrics.mean_std()

        logging.getLogger(self.get_logger_name()).info('Discriminative Score - Mean: ' + str(results[0]) + ', Std: ' + str(results[1]))
        logging.getLogger(self.get_logger_name()).info('Predictive Score - Mean: ' + str(results[2]) + ', Std: ' + str(results[3]))

        visualizer = Visualizers(dataX, dataX_hat)
        visualizer.PCA(prefix_str+'pca.png')
        visualizer.tSNE(prefix_str+'tsne.png')
        logging.getLogger(self.get_logger_name()).info("Visualization complete.")

        return {"discriminative score mean": results[0], "predictive score mean": results[2]}

    def get_hash(self):
        return self.get_logger_name()


# In the generation specification keys that have lists as their values will be cross producted with other list valued keys to create many specifications
# in this instance there will be 8 * 3 = 24 specifications
generation_specification = {  # just trying out random values for testing
    "total_iterations": [1], 
    "sub_iterations": [2],
    "data_size": [300],
    "max_seq_length": [12],
    "iterations": [10001],
    "batch_size": [128],
    # "module_name": ['gru', 'lstm', 'lstmLN']
}

# Call the generate method. Will create the cross product.
specifications = SpecificationGenerator().generate(generation_specification)
print(specifications)

expt = TsganExperiment()
name = "tsgan_unseen_metrics" #+expt.get_hash()
runner = ExperimentRunner()
runner.run(name, specifications, expt, specification_runner=MainRunner(), propagate_exceptions=True)


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

        # data loading
        # need to change to use data_loader later
        # currently using cashed data from csv file
        x = np.loadtxt('converted_loc.csv', delimiter=',', skiprows=1)
        x = np.delete(x, 0, axis=1)
        x = x[:specification["data_size"],:] # trying on 500 data
        wrapper = TsganWrapper(x)
        seq_length = max(specification["max_seq_length"], int(specification["data_size"]/3))
        wrapper.build_dataset(seq_length)
        dataX = wrapper.dataX
        logging.getLogger(self.get_logger_name()).info("Dataset is ready.")
        # print('Dataset is ready.')

        wrapper.set_tgan_parameters('iterations', specification['iterations'])
        wrapper.fit(self.get_logger_name(), '500data_model')
        dataX_hat = wrapper.generate()

        visualizer = Visualizers(dataX, dataX_hat)
        visualizer.PCA('500data_pca.png')
        visualizer.tSNE('500data_tsne.png')
        logging.getLogger(self.get_logger_name()).info("Visualization complete.")

        metrics = TsganMetrics(dataX, dataX_hat, specification['sub_iterations'])
        metrics.compute_discriminative()
        metrics.compute_predictive()
        results = metrics.mean_std()

        logging.getLogger(self.get_logger_name()).info('Discriminative Score - Mean: ' + str(results[0]) + ', Std: ' + str(results[1]))
        logging.getLogger(self.get_logger_name()).info('Predictive Score - Mean: ' + str(results[2]) + ', Std: ' + str(results[3]))
        return {"discriminative score mean": results[0], "predictive score mean": results[2]}


# In the generation specification keys that have lists as their values will be cross producted with other list valued keys to create many specifications
# in this instance there will be 8 * 3 = 24 specifications
generation_specification = {  # just trying out random values for testing
    # "total_iterations": [1], 
    "sub_iterations": [3],
    "data_size": [500],
    "max_seq_length": [40],
    "iterations": [10]  # trying on very little iterations
    # "batch_size": [32, 64, 128],
    # "module_name": ['gru', 'lstm', 'lstmLN']
}

# Call the generate method. Will create the cross product.
specifications = SpecificationGenerator().generate(generation_specification)
print(specifications)

name = "tsgan_500data_experiment"
runner = ExperimentRunner()
runner.run(name, specifications, TsganExperiment(), specification_runner=MainRunner(), propagate_exceptions=True)


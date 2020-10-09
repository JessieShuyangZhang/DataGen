import typing
import numpy as np # ??

from examples.example_utils import delete_experiments_folder
from smallab.experiment_types.experiment import Experiment
from smallab.runner.runner import ExperimentRunner
from smallab.specification_generator import SpecificationGenerator

from tsgan_wrapper import TsganWrapper

# Same experiment as before
class TsganExperiment(Experiment):
    def main(self, specification:typing.Dict) -> typing.Dict:

        # data loading
        # need to change to use data_loader later
        # currently using cashed data from csv file
        x = np.loadtxt('converted_loc.csv', delimiter=',', skiprows=1)
        x = np.delete(x, 0, axis=1)
        wrapper = TsganWrapper(x)
        wrapper.build_dataset()
        dataX = wrapper.dataX
        print('Dataset is ready.')

        wrapper.set_tgan_parameters('iterations', specification['iterations'])
        results = wrapper.run_tgan(specification['total_iterations'], specification['sub_iterations'])
        return {"discriminative score mean": results[0], "predictive score mean": results[2]}


# In the generation specification keys that have lists as their values will be cross producted with other list valued keys to create many specifications
# in this instance there will be 8 * 3 = 24 specifications
generation_specification = {  # just trying out random values for testing
    "total_iterations": [2], 
    "sub_iterations": [3],
    "iterations": [10000], #, 10000, 15000
    # "batch_size": [32, 64, 128],
    # "module_name": ['gru', 'lstm', 'lstmLN']
}

# Call the generate method. Will create the cross product.
specifications = SpecificationGenerator().generate(generation_specification)
print(specifications)

name = "tsgan_experiment"
runner = ExperimentRunner()
runner.run(name, specifications, TsganExperiment())

delete_experiments_folder(name)

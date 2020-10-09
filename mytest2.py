from tsgan_wrapper import TsganWrapper
import numpy as np

x = np.loadtxt('converted_loc.csv', delimiter=',', skiprows=1)
x = np.delete(x, 0, axis=1)
wrapper = TsganWrapper(x)
wrapper.build_dataset()
dataX = wrapper.dataX
print('Dataset is ready.')

specification = {  # just trying out random values for testing
    "total_iterations": 1, 
    "sub_iterations": 3,
    "iterations": 10 #, 10000, 15000
    # "batch_size": [32, 64, 128],
    # "module_name": ['gru', 'lstm', 'lstmLN']
}

wrapper.set_tgan_parameters('iterations', specification['iterations'])
results = wrapper.run_tgan(specification['total_iterations'], specification['sub_iterations'], 'mytest2_model')
print('Discriminative Score - Mean: ' + str(results[0]) + ', Std: ' + str(results[1]))
print('Predictive Score - Mean: ' + str(results[2]) + ', Std: ' + str(results[3]))









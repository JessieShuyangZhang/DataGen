from tsgan_wrapper import TsganWrapper
from tsgan_metrics import TsganMetrics 
from visualizers import Visualizers 
import numpy as np

x = np.loadtxt('converted_loc.csv', delimiter=',', skiprows=1) # test on very small dataset
x = np.delete(x, 0, axis=1)
x = x[:6,:]

wrapper = TsganWrapper(x)
wrapper.build_dataset(seq_length=3)
dataX = wrapper.dataX
# print(dataX)
print('Dataset is ready.')

specification = {  # just trying out random values for testing
    "total_iterations": 1, 
    "sub_iterations": 3,
    "iterations": 10  # trying on very little iterations
    # "batch_size": [32, 64, 128],
    # "module_name": ['gru', 'lstm', 'lstmLN']
}

wrapper.set_tgan_parameters('iterations', specification['iterations'])
wrapper.fit('mytest2_model')
dataX_hat = wrapper.generate()

visualizer = Visualizers(dataX, dataX_hat)
visualizer.PCA()
visualizer.tSNE()

metrics = TsganMetrics(dataX, dataX_hat, specification['sub_iterations'])
metrics.compute_discriminative()
metrics.compute_predictive()
results = metrics.mean_std()

print('Discriminative Score - Mean: ' + str(results[0]) + ', Std: ' + str(results[1]))
print('Predictive Score - Mean: ' + str(results[2]) + ', Std: ' + str(results[3]))









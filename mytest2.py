from tsgan_wrapper import TsganWrapper
from tsgan_metrics import TsganMetrics 
from visualizers import Visualizers 
import numpy as np
import pdb 

x = np.loadtxt('conv_loc_time_new.csv', delimiter=',') #, skiprows=1   test on very small dataset
x = np.delete(x, 0, axis=1)
x = x[:200,:]

wrapper = TsganWrapper(x)
wrapper.build_dataset() # seq_length=3
dataX = wrapper.dataX
# print(dataX)
print('Dataset is ready.')

# buidling unseen dataX for discriminative score evaluation
unseen = np.loadtxt('unseen_dataX_new.csv', delimiter=',')
unseen = np.delete(unseen, 0, axis=1)
unseen_wrap = TsganWrapper(unseen)
unseen_wrap.build_dataset()
dataX_disctest = unseen_wrap.dataX
dataX_disctest = dataX_disctest[:len(dataX), :]


specification = {  # just trying out random values for testing
    "total_iterations": 1, 
    "sub_iterations": 2,
    "iterations": 10,  # trying on very little iterations
    "batch_size": 128, # 32, 64, 
    # "module_name": ['gru', 'lstm', 'lstmLN']
}

wrapper.set_tgan_parameters('iterations', specification['iterations'])
wrapper.set_tgan_parameters('batch_size', specification['batch_size'])
wrapper.set_tgan_parameters('hidden_dim', len(dataX[0][0,:]) * 4)
wrapper.set_tgan_parameters('z_dim', len(dataX[0][0,:]))
# wrapper.fit(model_saved_name='mytest2_model')
# dataX_hat = wrapper.generate()
prefix_str='mytest2_debug_'

metrics = TsganMetrics(specification['sub_iterations'])

for it in range(specification['total_iterations']):
    # pdb.set_trace()

    wrapper.fit(filename=prefix_str+'model')
    dataX_hat = wrapper.generate()

    # pdb.set_trace()

    print("computing discriminative")
    metrics.compute_discriminative(dataX_disctest, dataX_hat)
    print("computing predictive")
    metrics.compute_predictive(dataX_disctest, dataX_hat)

results = metrics.mean_std()
visualizer = Visualizers(dataX, dataX_hat)
visualizer.PCA(prefix_str+'pca.png')
visualizer.tSNE(prefix_str+'tsne.png')

print('Discriminative Score - Mean: ' + str(results[0]) + ', Std: ' + str(results[1]))
print('Predictive Score - Mean: ' + str(results[2]) + ', Std: ' + str(results[3]))

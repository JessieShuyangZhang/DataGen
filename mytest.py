# try to restore the model: mytest2_model 
# import numpy as np

import tensorflow as tf

import numpy as np
from tsgan_wrapper import TsganWrapper

x = np.loadtxt('converted_loc.csv', delimiter=',', skiprows=1) # test on very small dataset
x = np.delete(x, 0, axis=1)

wrapper = TsganWrapper(x)
wrapper.build_dataset() # seq_length=3
dataX = wrapper.dataX
# print(dataX)
print('Dataset is ready.')

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

prefix_str='delete_this_'

for it in range(specification['total_iterations']):
    wrapper.fit(model_saved_name=prefix_str+'model')
    dataX_hat = wrapper.generate()


sess = tf.Session()
saver = tf.train.import_meta_graph("saved_models/"+prefix_str+"model/"+prefix_str+"model.meta")
saver.restore(sess, tf.train.latest_checkpoint('saved_models/'+prefix_str+"model/"))

graph = tf.get_default_graph()

X = graph.get_tensor_by_name("myinput_x:0")
T = graph.get_tensor_by_name("myinput_t:0")
Z = graph.get_tensor_by_name("myinput_z:0")

X_hat = graph.get_tensor_by_name("recovery_1/fully_connected/Sigmoid:0") # name of the operation

dataT = list()
Max_Seq_Len = 0
for i in range(len(dataX)):
    Max_Seq_Len = max(Max_Seq_Len, len(dataX[i][:,0]))
    dataT.append(len(dataX[i][:,0]))

Z_mb = wrapper.noise_generator(len(dataX), len(dataX[0][0,:]), dataT, Max_Seq_Len)

X_hat_curr = sess.run(X_hat, feed_dict={Z: Z_mb, X: dataX, T: dataT})

dataX_hat = list()
    
for i in range(len(dataX)):
    Temp = X_hat_curr[i,:dataT[i],:]
    dataX_hat.append(Temp)


min_val = np.min(np.min(dataX, axis = 0), axis = 0)    
max_val = np.max(np.max(dataX, axis = 0), axis = 0)
dataX_hat = dataX_hat * max_val
dataX_hat = dataX_hat + min_val
# graph.get_tensors()


# graph.get_operations()  # returns soo many operations
# tf.trainable_variables()

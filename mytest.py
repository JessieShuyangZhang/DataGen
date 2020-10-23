# Don't run this file. It's just a bunch of scratch

# try to restore the model: mytest2_model 
# import numpy as np

import tensorflow as tf
import numpy as np
from tsgan_wrapper import TsganWrapper

# x = np.loadtxt('converted_loc.csv', delimiter=',', skiprows=1) # test on very small dataset
# x = np.delete(x, 0, axis=1)

# wrapper = TsganWrapper(x)
wrapper = TsganWrapper([])
# wrapper.build_dataset()
# dataX = wrapper.dataX
# print('Dataset is ready.')

# specification = { 
#     "total_iterations": 1, 
#     "sub_iterations": 2,
#     "iterations": 10, 
#     "batch_size": 128, 
# }
# wrapper.set_tgan_parameters('iterations', specification['iterations'])
# wrapper.set_tgan_parameters('batch_size', specification['batch_size'])
# wrapper.set_tgan_parameters('hidden_dim', len(dataX[0][0,:]) * 4)
# wrapper.set_tgan_parameters('z_dim', len(dataX[0][0,:]))

prefix_str='delete_this_'

# wrapper.fit(model_saved_name=prefix_str+'model')
# dataX_hat = wrapper.generate()


sess = tf.Session()
saver = tf.train.import_meta_graph("saved_models/"+prefix_str+"model/"+prefix_str+"model.meta")
saver.restore(sess, tf.train.latest_checkpoint('saved_models/'+prefix_str+"model/"))

graph = tf.get_default_graph()
X_hat = graph.get_tensor_by_name("recovery_1/fully_connected/Sigmoid:0") # name of the operation
X = graph.get_tensor_by_name("myinput_x:0")
T = graph.get_tensor_by_name("myinput_t:0")
Z = graph.get_tensor_by_name("myinput_z:0")

dataX = graph.get_tensor_by_name("dataX:0").eval(session=sess)
No = graph.get_tensor_by_name("dataX_len:0").eval(session=sess)
dataT = graph.get_tensor_by_name("dataT:0").eval(session=sess)
z_dim = graph.get_tensor_by_name("z_dimension:0").eval(session=sess)
Max_Seq_Len = graph.get_tensor_by_name("Max_Seq_Len:0").eval(session=sess)

Z_mb = wrapper.noise_generator(No, z_dim, dataT, Max_Seq_Len)
X_hat_curr = sess.run(X_hat, feed_dict={Z: Z_mb, X: dataX, T: dataT})

# construct dataX_hat from raw synthetic data X_hat_curr
dataX_hat = list()
for i in range(No):
    Temp = X_hat_curr[i,:dataT[i],:]
    dataX_hat.append(Temp)

min_val = np.min(np.min(dataX, axis = 0), axis = 0)    
max_val = np.max(np.max(dataX, axis = 0), axis = 0)
dataX_hat = dataX_hat * max_val
dataX_hat = dataX_hat + min_val










# testing workability of discriminative_score_metrics

import sys
sys.path.append('metrics')
from discriminative_score_metrics import discriminative_score_metrics
import numpy as np
# dataX_hat=np.array([[[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]],[[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]]])
dataX_hat=np.array([[[1,1,0,1,0],[1,1,0,1,1],[0,0,0,1,0]],[[1,1,0,1,1],[0,0,0,0,0],[1,1,0,1,1]],[[1,1,1,1,1],[0,0,0,0,0],[1,0,0,1,1]]])

# dataX=np.array([[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]])
dataX=np.array([[[1,1,0,1,0],[1,1,1,1,1],[0,0,0,1,0]],[[1,1,1,1,1],[0,0,0,0,0],[1,1,0,1,1]],[[1,1,1,1,1],[0,0,0,0,0],[1,0,0,1,1]]])
disc_score=discriminative_score_metrics(dataX, dataX_hat)

print(disc_score)








# load data once more
from data_loader import DataLoader
dl = DataLoader(csv_filename='conv_loc_time.csv')
dl.load_raw_data()
dl.convert_location()
dl.convert_time()
dl.output_csv()

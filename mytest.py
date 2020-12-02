# Don't run this file. It's just a bunch of scratch

# try to restore the model: mytest2_model 
# import numpy as np

import tensorflow as tf
import numpy as np
from tsgan_wrapper import TsganWrapper

# x = np.loadtxt('data/converted_loc.csv', delimiter=',', skiprows=1) # test on very small dataset
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
dl = DataLoader(csv_filename='raw_data.csv')
dl.del_position_key()
dl.output_csv()

# load data once more
from data_loader import DataLoader
dl = DataLoader()
robot_keys_mexico = dl.get_robot_keys_at_location()
robot_key = robot_keys_mexico[0]
dl.load_raw_data(robot_key)
dl.output_csv("mexico/key_1.csv")
# for i in range(1,5):
#     robot_key = robot_keys_mexico[i]
#     dl.load_raw_data(robot_key)
#     dl.output_csv('mexico/key_'+str(robot_key)+'.csv')



from data_loader import DataLoader
dl = DataLoader()
robot_starting = dl.load_raw_data()


'''
robot keys of Gulf of Mexico
[1, 2, 13, 29, 35, 36, 38, 39, 51, 52, 54, 55, 58, 64, 65, 66, 67, 70, 72, 78, 90, 159, 193, 194, 202, 204, 205, 
206, 207, 208, 209, 210, 211, 213, 214, 215, 216, 217, 218, 237, 245, 247, 248, 249, 250, 251, 252, 253, 254, 255, 
256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 292, 530, 531, 532, 547, 548, 549, 550, 551, 554, 555, 
556, 557, 558, 560, 562, 564, 614, 615, 620, 621, 622, 623, 624, 625, 627, 628, 629, 630, 631, 632, 633, 634, 635, 
753, 755, 757, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 777, 778, 779, 780, 781, 782, 783, 
786, 849, 905, 906, 911, 934]
'''

import sqlite3 as sq3 
connector = sq3.connect('../../noaa-data.db')
cursor=connector.cursor()
import json
with open('start_position.json') as json_file:
    robot_starting = json.load(json_file)

robot_starting={int(k):int(v) for k,v in robot_starting.items()}
# try robot_key==13
key = 13
start_position = robot_starting[13]
end_position = robot_starting[29]-1
fetch_data_cmd = "SELECT x.type_of_data, x.value FROM (sensor_data inner join position on position.position_key==sensor_data.position_key) as x WHERE x.position_key=? LIMIT 4"
for position in range(start_position,end_position,100):
    cursor.execute(fetch_data_cmd,(position,))
    sensor_data = cursor.fetchall() # still slow when there is missing data
    print(position)
    print(sensor_data)


# position_key=1392161 only has temperature and pressure(irrelavent)

















# restore saved model to compute metrics

from tsgan_wrapper import TsganWrapper
from tsgan_metrics import TsganMetrics 
from visualizers import Visualizers 
# import numpy as np
# import pdb 

loadmodel = TsganWrapper([])
dataX = []
dataX_hat = []
model_name = 'smallab.princess-twenty-two-bravo'
dataX, dataX_hat = loadmodel.load_model(model_name)

metrics = TsganMetrics(2)
metrics.compute_discriminative(dataX, dataX_hat)
metrics.compute_predictive(dataX, dataX_hat)
results = metrics.mean_std()
print('Discriminative Score - Mean: ' + str(results[0]) + ', Std: ' + str(results[1]))
print('Predictive Score - Mean: ' + str(results[2]) + ', Std: ' + str(results[3]))

visualizer = Visualizers(dataX, dataX_hat)
visualizer.PCA(model_name+'_pca.png')
visualizer.tSNE(model_name+'_tsne.png')
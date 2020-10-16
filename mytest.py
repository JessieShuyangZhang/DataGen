# try to restore the model: mytest2_model 

import tensorflow as tf
import numpy as np

sess = tf.Session()
saver = tf.train.import_meta_graph("saved_models/mytest2_model.meta")
saver.restore(sess, tf.train.latest_checkpoint('saved_models/'))

graph = tf.get_default_graph()

# graph.get_tensors()
X = graph.get_tensor_by_name("myinput_x:0")
T = graph.get_tensor_by_name("myinput_t:0")
Z = graph.get_tensor_by_name("myinput_z:0")
graph.get_operations()  # returns soo many operations
tf.trainable_variables()

for v in graph.as_graph_def().node: # same as graph.get_operations() but only getting the names
    print(v.name)
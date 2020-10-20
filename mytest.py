# try to restore the model: mytest2_model 
# import numpy as np

import tensorflow as tf
import sys

old_stdout = sys.stdout
log_file = open("tf_operation_names.log","w")
sys.stdout = log_file

sess = tf.Session()
saver = tf.train.import_meta_graph("saved_models/debug_model/debug_model.meta")
saver.restore(sess, tf.train.latest_checkpoint('saved_models/'))

graph = tf.get_default_graph()

# X = graph.get_tensor_by_name("myinput_x:0")
# T = graph.get_tensor_by_name("myinput_t:0")
# Z = graph.get_tensor_by_name("myinput_z:0")

# graph.get_tensors()


# graph.get_operations()  # returns soo many operations
# tf.trainable_variables()

for v in graph.as_graph_def().node: # same as graph.get_operations() but only getting the names
    print(v.name)


sys.stdout = old_stdout

log_file.close()
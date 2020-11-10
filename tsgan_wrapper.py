import numpy as np
from tgan import tgan
import tensorflow as tf
from datagen_wrapper import DataGenWrapper
import pdb

class TsganWrapper(DataGenWrapper):
    '''
    data preprocessing, random generator for tgan, run/train tgan
    try feeding the pretrained(saved) model some noisy data and generate synthetic data
    '''
    def __init__(self, raw_data):
        self.raw_data = raw_data.copy()
        self.dataX = [] # original data
        self.dataX_hat = [] # synthetic data
        self.parameters = dict() # network parameters
        # parameters: a dictionary with keys: hidden_dim,num_layers,iterations,batch_size,module_name,z_dim
        # arbitrary initial values
        self.parameters['hidden_dim'] = 32 
        self.parameters['num_layers'] = 3
        self.parameters['iterations'] = 10 # was 50000 took super long
        self.parameters['batch_size'] = 32 # changed this to fit smaller datasize 
        self.parameters['module_name'] = 'gru' # Other optios: 'lstm' or 'lstmLN'
        self.parameters['z_dim'] = 8 

    def build_dataset(self, seq_length=12):  # can tune seq_length
        pdb.set_trace()
        tempX = []
        # Cut data by sequence length
        for i in range(0, len(self.raw_data) - seq_length):
            _x = (self.raw_data[i:i + seq_length]).copy()
            t0 = _x[0][0]
            for i in range(len(_x)): # make the first row in the sequence be t0 = 0
                _x[i][0] -= t0
            tempX.append(_x)
        # Mix Data (to make it similar to i.i.d)
        idx = np.random.permutation(len(tempX))
        self.dataX = []
        for i in range(len(tempX)):
            self.dataX.append(tempX[idx[i]])

        # normalize dataX
        self.dataX, minval, maxval = self.MinMaxScaler(self.dataX)
        return self.dataX
        

    def set_tgan_parameters(self, param_name, value):  # can tune network parameter through param_name
        if(param_name in self.parameters):
            self.parameters[param_name] = value
    
    #%% Random vector generation
    def noise_generator(self, batch_size, z_dim, T_mb, Max_Seq_Len):      
        Z_mb = list()
        for i in range(batch_size):            
            Temp = np.zeros([Max_Seq_Len, z_dim])            
            Temp_Z = np.random.uniform(0., 1, [T_mb[i], z_dim])        
            Temp[:T_mb[i],:] = Temp_Z            
            Z_mb.append(Temp_Z)      
        return Z_mb 

    def fit(self, filename, logger=''):
        self.dataX_hat = tgan(self.dataX, self.parameters, self.noise_generator, logger, filename)
        print('Finish Synthetic Data Generation')

    def generate(self):
        return self.dataX_hat
    
    def load_model(self, filename):
        sess = tf.Session()
        saver = tf.train.import_meta_graph('saved_models/'+filename+'/'+filename+'.meta')
        saver.restore(sess, tf.train.latest_checkpoint('saved_models/'+filename+'/'))
        graph = tf.get_default_graph()
        X_hat = graph.get_tensor_by_name("recovery_1/fully_connected/Sigmoid:0") # name of operation
        X = graph.get_tensor_by_name("myinput_x:0")
        T = graph.get_tensor_by_name("myinput_t:0")
        Z = graph.get_tensor_by_name("myinput_z:0")

        dataX = graph.get_tensor_by_name("dataX:0")
        dataX = dataX.eval(session=sess)
        No = graph.get_tensor_by_name("dataX_len:0").eval(session=sess)
        dataT = graph.get_tensor_by_name("dataT:0").eval(session=sess)
        z_dim = graph.get_tensor_by_name("z_dimension:0").eval(session=sess)
        Max_Seq_Len = graph.get_tensor_by_name("Max_Seq_Len:0").eval(session=sess)

        # generate synthetic data from noise input
        Z_mb = self.noise_generator(No, z_dim, dataT, Max_Seq_Len)
        X_hat_curr = sess.run(X_hat, feed_dict={Z: Z_mb, X: dataX, T: dataT})

        # construct dataX_hat from raw synthetic data X_hat_curr
        dataX_hat = list()
        for i in range(No):
            Temp = X_hat_curr[i,:dataT[i],:]
            dataX_hat.append(Temp)

        return dataX, dataX_hat
        

    def MinMaxScaler(self, dataX):
        min_val = np.min(np.min(dataX, axis = 0), axis = 0)
        dataX = dataX - min_val        
        max_val = np.max(np.max(dataX, axis = 0), axis = 0)
        dataX = dataX / (max_val + 1e-7)        
        return dataX, min_val, max_val


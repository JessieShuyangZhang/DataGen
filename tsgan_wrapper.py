import numpy as np
from tgan import tgan

class TsganWrapper:
    '''
    data preprocessing, random generator for tgan, run/train tgan
    try feeding the pretrained(saved) model some noisy data and generate synthetic data
    '''
    def __init__(self, raw_data):
        self.raw_data = raw_data
        self.dataX = []
        self.parameters = dict()
        # arbitrary initial values
        self.parameters['hidden_dim'] = 32 #arbitrary
        self.parameters['num_layers'] = 3
        self.parameters['iterations'] = 10 # was 50000 took super long
        self.parameters['batch_size'] = 128
        self.parameters['module_name'] = 'gru'   # Other options: 'lstm' or 'lstmLN'
        self.parameters['z_dim'] = 8 #arbitrary
        # parameters: a dictionary with keys: hidden_dim,num_layers,iterations,batch_size,module_name,z_dim

    # def MinMaxScaler(self): # normalize the data
    #     numerator = self.raw_data - np.min(self.raw_data, 0)
    #     denominator = np.max(self.raw_data, 0) - np.min(self.raw_data, 0)
    #     self.raw_data = numerator / (denominator + 1e-7)

    def build_dataset(self, seq_length=40):
        # self.MinMaxScaler()
        self.dataX = []
        for i in range(len(self.raw_data) - seq_length):
            _x = self.raw_data[i:i+seq_length]
            self.dataX.append(_x)
        # Mix Data (to make it similar to i.i.d)
        np.random.shuffle(self.dataX)

    def set_tgan_parameters(self, param_name, value):
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

    def run_tgan(self):
        return tgan(self.dataX, self.parameters, self.noise_generator)




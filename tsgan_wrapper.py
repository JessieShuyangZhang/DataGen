import numpy as np
import sys
from tgan import tgan

sys.path.append('metrics')
from discriminative_score_metrics import discriminative_score_metrics
from visualization_metrics import PCA_Analysis, tSNE_Analysis
from predictive_score_metrics import predictive_score_metrics

# not sure if those imports will work...


class TsganWrapper:
    '''
    data preprocessing, random generator for tgan, run/train tgan
    try feeding the pretrained(saved) model some noisy data and generate synthetic data
    '''
    def __init__(self, raw_data):
        self.raw_data = raw_data
        self.dataX = [] # original data
        self.dataX_hat = [] # synthetic data
        self.discriminative_score = list()
        self.predictive_score = list()
        self.parameters = dict() # network parameters
        # arbitrary initial values
        self.parameters['hidden_dim'] = 32 
        self.parameters['num_layers'] = 3
        self.parameters['iterations'] = 10 # was 50000 took super long
        self.parameters['batch_size'] = 128
        self.parameters['module_name'] = 'gru'   # Other options: 'lstm' or 'lstmLN'
        self.parameters['z_dim'] = 8 
        # parameters: a dictionary with keys: hidden_dim,num_layers,iterations,batch_size,module_name,z_dim

    def build_dataset(self, seq_length=40):  # can tune seq_length
        self.dataX = []
        for i in range(len(self.raw_data) - seq_length):
            _x = self.raw_data[i:i+seq_length]
            self.dataX.append(_x)
        # Mix Data (to make it similar to i.i.d)
        np.random.shuffle(self.dataX)

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

    def run_tgan(self, total_iterations, sub_iterations, model_saved_name):
        # self.dataX_hat = tgan(self.dataX, self.parameters, self.noise_generator)
        
        for it in range(total_iterations):
            # Synthetic Data Generation
            self.dataX_hat = tgan(self.dataX, self.parameters, self.noise_generator, model_saved_name)
            print('Finish Synthetic Data Generation')

            #%% Performance Metrics

            # 1. Discriminative Score            
            Acc = list()
            for tt in range(sub_iterations):
                Acc.append(discriminative_score_metrics(self.dataX, self.dataX_hat))
            self.discriminative_score.append(np.mean(Acc))

            # 2. Predictive Performance
            MAE_All = list()
            for tt in range(sub_iterations):
                MAE_All.append(predictive_score_metrics(self.dataX, self.dataX_hat))
            self.predictive_score.append(np.mean(MAE_All))
            
        print('Finish TGAN iterations')
        self.visualizations()
        return self.results_mean_std()
        # return dataX_hat


    # metrics
    def visualizations(self):
        PCA_Analysis(self.dataX, self.dataX_hat)
        tSNE_Analysis(self.dataX, self.dataX_hat)

    def results_mean_std(self):        
        disc_mean = np.round(np.mean(self.discriminative_score),4)
        disc_std = np.round(np.std(self.discriminative_score),4)
        pred_mean = np.round(np.mean(self.discriminative_score),4)
        pred_std = np.round(np.std(self.discriminative_score),4)
        return disc_mean, disc_std, pred_mean, pred_std
        # print('Discriminative Score - Mean: ' + str(np.round(np.mean(self.discriminative_score),4)) + ', Std: ' + str(np.round(np.std(self.discriminative_score),4)))
        # print('Predictive Score - Mean: ' + str(np.round(np.mean(self.predictive_score),4)) + ', Std: ' + str(np.round(np.std(self.predictive_score),4)))



#%% Necessary Packages

import numpy as np
import sys

#%% Functions
# 1. Models
# from tgan import tgan

# 2. Data Loading
# from data_loading import google_data_loading, sine_data_generation
from tsgan_wrapper import TsganWrapper

# 3. Metrics
sys.path.append('metrics')
from discriminative_score_metrics import discriminative_score_metrics
from visualization_metrics import PCA_Analysis, tSNE_Analysis
from predictive_score_metrics import predictive_score_metrics

#%% Main Parameters
# Experiments iterations
Iteration = 1 # was 2
Sub_Iteration = 3

#%% Data Loading
# using cashed dataset with seq_length=40
x = np.loadtxt('converted_loc.csv', delimiter=',', skiprows=1)
x = np.delete(x, 0, axis=1)
wrapper = TsganWrapper(x)
wrapper.build_dataset()
dataX = wrapper.dataX
print('Dataset is ready.')

#%% Newtork Parameters
parameters = dict()
parameters['hidden_dim'] = len(dataX[0][0,:]) * 4
parameters['num_layers'] = 3
parameters['iterations'] = 10 # was 50000 took super long
parameters['batch_size'] = 128
parameters['module_name'] = 'gru'   # Other options: 'lstm' or 'lstmLN'
parameters['z_dim'] = len(dataX[0][0,:])
print('Parameters are ' + str(parameters))

for k in parameters:
    wrapper.set_tgan_parameters(k, parameters[k])

#%% Experiments
# Output Initialization
Discriminative_Score = list()
Predictive_Score = list()

# Each Iteration
for it in range(Iteration):

    # Synthetic Data Generation
    # dataX_hat = tgan(dataX, parameters, wrapper.noise_generator)
    dataX_hat = wrapper.run_tgan()

    print('Finish Synthetic Data Generation')

    #%% Performance Metrics

    # 1. Discriminative Score
    
    Acc = list()
    for tt in range(Sub_Iteration):
        Temp_Disc = discriminative_score_metrics (dataX, dataX_hat)
        Acc.append(Temp_Disc)

    Discriminative_Score.append(np.mean(Acc))

    # 2. Predictive Performance
    MAE_All = list()
    for tt in range(Sub_Iteration):
        MAE_All.append(predictive_score_metrics (dataX, dataX_hat))

    Predictive_Score.append(np.mean(MAE_All))
    
print('Finish TGAN iterations')


#%% 3. Visualization
PCA_Analysis (dataX, dataX_hat)
tSNE_Analysis (dataX, dataX_hat)

# Print Results
print('Discriminative Score - Mean: ' + str(np.round(np.mean(Discriminative_Score),4)) + ', Std: ' + str(np.round(np.std(Discriminative_Score),4)))
print('Predictive Score - Mean: ' + str(np.round(np.mean(Predictive_Score),4)) + ', Std: ' + str(np.round(np.std(Predictive_Score),4)))

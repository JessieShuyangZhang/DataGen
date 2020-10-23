import numpy as np
import sys
sys.path.append('metrics')
from discriminative_score_metrics import discriminative_score_metrics
from predictive_score_metrics import predictive_score_metrics

class TsganMetrics:
    def __init__(self, sub_iterations):#(self, dataX, dataX_hat, sub_iterations):
        # self.dataX = dataX
        # self.dataX_hat = dataX_hat
        self.sub_iterations = sub_iterations
        self.discriminative_score = list()
        self.predictive_score = list()

    def compute_discriminative(self, dataX, dataX_hat):
        # for tt in range(self.sub_iterations):
        #     self.discriminative_score.append(discriminative_score_metrics(dataX, dataX_hat))
        # print('*acc_array*',self.discriminative_score)
        
        acc = list()
        for tt in range(self.sub_iterations):
            acc.append(discriminative_score_metrics(dataX, dataX_hat))
        self.discriminative_score.append(np.mean(acc))
        

    def compute_predictive(self, dataX, dataX_hat):
        # for tt in range(self.sub_iterations):
        #     self.predictive_score.append(predictive_score_metrics(dataX, dataX_hat))
        # print('*MAE_array*',self.predictive_score)
        
        MAE_ALL = list()
        for tt in range(self.sub_iterations):
            MAE_ALL.append(predictive_score_metrics(dataX, dataX_hat))
        self.predictive_score.append(np.mean(MAE_ALL))
        
    
    def mean_std(self):
        print('*disc_array*',self.discriminative_score)
        print('*pred_array*',self.predictive_score)
        disc_mean = np.round(np.mean(self.discriminative_score),4)
        disc_std = np.round(np.std(self.discriminative_score),4)
        pred_mean = np.round(np.mean(self.predictive_score),4)
        pred_std = np.round(np.std(self.predictive_score),4)
        return disc_mean, disc_std, pred_mean, pred_std
        
    # def mean_std(disc, pred):
    #     disc_mean = np.round(np.mean(disc),4)
    #     disc_std = np.round(np.std(disc),4)
    #     pred_mean = np.round(np.mean(pred),4)
    #     pred_std = np.round(np.std(pred),4)
    #     return disc_mean, disc_std, pred_mean, pred_std 



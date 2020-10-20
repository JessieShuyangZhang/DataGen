import numpy as np
import sys
sys.path.append('metrics')
from discriminative_score_metrics import discriminative_score_metrics
from predictive_score_metrics import predictive_score_metrics

class TsganMetrics:
    def __init__(self, dataX, dataX_hat, sub_iterations):
        self.dataX = dataX
        self.dataX_hat = dataX_hat
        self.sub_iterations = sub_iterations
        self.discriminative_score = list()
        self.predictive_score = list()

    def compute_discriminative(self):
        for tt in range(self.sub_iterations):
            self.discriminative_score.append(discriminative_score_metrics(self.dataX, self.dataX_hat))
        print('*acc_array*',self.discriminative_score)

    def compute_predictive(self):
        for tt in range(self.sub_iterations):
            self.predictive_score.append(predictive_score_metrics(self.dataX, self.dataX_hat))
        print('*MIE_array*',self.predictive_score)
    
    def mean_std(self):
        disc_mean = np.round(np.mean(self.discriminative_score),4)
        disc_std = np.round(np.std(self.discriminative_score),4)
        pred_mean = np.round(np.mean(self.discriminative_score),4)
        pred_std = np.round(np.std(self.discriminative_score),4)
        return disc_mean, disc_std, pred_mean, pred_std
        
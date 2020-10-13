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
        self.discriminative_score = []
        self.predictive_score = []

    def compute_discriminative(self):
        acc = list()
        for tt in range(self.sub_iterations):
            acc.append(discriminative_score_metrics(self.dataX, self.dataX_hat))
        self.discriminative_score.append(np.mean(acc))

    def compute_predictive(self):
        MAE_All = list()
        for tt in range(self.sub_iterations):
            MAE_All.append(predictive_score_metrics(self.dataX, self.dataX_hat))
        self.predictive_score.append(np.mean(MAE_All))
    
    def mean_std(self):
        disc_mean = np.round(np.mean(self.discriminative_score),4)
        disc_std = np.round(np.std(self.discriminative_score),4)
        pred_mean = np.round(np.mean(self.discriminative_score),4)
        pred_std = np.round(np.std(self.discriminative_score),4)
        return disc_mean, disc_std, pred_mean, pred_std
        
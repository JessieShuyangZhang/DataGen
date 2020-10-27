import numpy as np
import sys
sys.path.append('metrics')
from discriminative_score_metrics import discriminative_score_metrics
from predictive_score_metrics import predictive_score_metrics

class TsganMetrics:
    def __init__(self, sub_iterations):
        self.sub_iterations = sub_iterations
        self.discriminative_score = list()
        self.predictive_score = list()

    def compute_discriminative(self, dataX, dataX_hat):
        assert (
            len(dataX) == len(dataX_hat)
        ), f"original and synthetic data have different length"

        acc = list()
        for tt in range(self.sub_iterations):
            self.discriminative_score.append(discriminative_score_metrics(dataX, dataX_hat))
        

    def compute_predictive(self, dataX, dataX_hat):
        assert (
            len(dataX) == len(dataX_hat)
        ), f"original and synthetic data have different length"

        for tt in range(self.sub_iterations):
            self.predictive_score.append(predictive_score_metrics(dataX, dataX_hat))
        
    
    def mean_std(self):
        print('*disc_array*',self.discriminative_score)
        print('*pred_array*',self.predictive_score)
        disc_mean = np.round(np.mean(self.discriminative_score),4)
        disc_std = np.round(np.std(self.discriminative_score),4)
        pred_mean = np.round(np.mean(self.predictive_score),4)
        pred_std = np.round(np.std(self.predictive_score),4)
        return disc_mean, disc_std, pred_mean, pred_std


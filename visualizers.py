import sys
sys.path.append('metrics')
from visualization_metrics import PCA_Analysis, tSNE_Analysis

class Visualizers:
    def __init__(self, dataX, dataX_hat):
        self.dataX = dataX
        self.dataX_hat = dataX_hat

    def PCA(self, output_file='PCA.png'):
        PCA_Analysis(self.dataX, self.dataX_hat, output_file)

    def tSNE(self, output_file='tSNE.png'):
        tSNE_Analysis(self.dataX, self.dataX_hat, output_file)
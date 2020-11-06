import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
# import warnings
# warnings.simplefilter("ignore")
import pdb
import sys
sys.path.append('src')
from spacegan_method import SpaceGAN
from spacegan_selection import get_spacegan_config, compute_metrics
from spacegan_utils import gaussian, rmse, mie, get_neighbours_featurize
from spacegan_config import Generator, Discriminator
from datagen_wrapper import DataGenWrapper

# fig_save_prefix = 'img/'

class SpaceganWrapper(DataGenWrapper):
    def __init__(self, prob_config, check_config, model_save_prefix='saved_models/noaa/'):
        self.model_save_prefix = model_save_prefix  # path to save the model
        self.prob_config = prob_config
        self.check_config = check_config
        self.df = None
        self.neighbours = None
        self.coord_vars = []
        self.cond_vars = []
        self.cont_vars = []
        self.output_vars = []        
        self.neighbour_list = [] # list of labels after augmentation by neighbours (eg. 'nn_houseValue_0', 'nn_houseValue_1'...'nn_latitude_8', 'nn_latitude_9')

        self.original = []
        self.synthetic = []

        # for building GAN
        self.target = []
        self.cond_input = []
        self.coord_input = []
        self.disc_method = None
        self.gen_method = None


    def build_dataset(self, filename, neighbours, output_vars, rows=2): # construct a dataframe from a csv file with a header of var names
        self.df = pd.read_csv(filename,nrows=rows) #don't know if this syntax works
        self.coord_vars = ["longitude", "latitude"] #Define spatial coordinates
        # cond_vars and cont_vars are hardcoded for now...can be made tunable later
        self.cond_vars = ['unix_time', 'depth', 'conductivity', 'density', 'temperature'] + self.coord_vars #Define the predictor variables
        self.cont_vars = ['unix_time', 'depth', 'conductivity', 'density', 'temperature', 'salinity'] + self.coord_vars #Define which neighbour features to use as context variables
        self.output_vars = [output_vars]
        self.neighbours = neighbours
        self.prob_config["cond_dim"] = len(self.cond_vars) + (self.neighbours * len(self.cont_vars))
        self.prob_config["output_dim"] = len(self.output_vars)  # size of output
        self.prob_config["noise_dim"] = self.prob_config["cond_dim"]  # size of noise
        

    def build_gan(self):
        # neighbours
        self.df, self.neighbour_list = get_neighbours_featurize(self.df, self.coord_vars, self.cont_vars, self.neighbours)

        # data structures
        self.target = self.df[self.output_vars].values
        self.cond_input = self.df[self.cond_vars + self.neighbour_list].values
        self.coord_input = self.df[self.coord_vars].values
        self.prob_config["output_labels"] = self.output_vars  # move to fit, before calling spaceGAN.train
        self.prob_config["input_labels"] = self.cond_vars + self.neighbour_list # move to fit, before calling spaceGAN.train

        # pre-instantiation
        self.disc_method = Discriminator(self.prob_config["output_dim"], self.prob_config["cond_dim"])
        self.disc_method.to(self.prob_config["device"])
        self.gen_method = Generator(self.prob_config["cond_dim"], self.prob_config["noise_dim"], self.prob_config["output_dim"])
        self.gen_method.to(self.prob_config["device"])


    def fit(self):
        # training SpaceGAN
        spacegan = SpaceGAN(self.prob_config, self.check_config, self.disc_method, self.gen_method)
        spacegan.train(x_train=self.cond_input, y_train=self.target, coords=self.coord_input)

        # export final model and data
        spacegan.checkpoint_model(spacegan.epochs) 
        self.save_model(spacegan.df_losses, "grid_spaceganlosses.pkl.gz")
    
    def select_best_generators(self):
        # computing metrics
        gan_metrics = compute_metrics(self.target, self.cond_input, self.prob_config, self.check_config, self.coord_input, self.neighbours)

        # selecting and sampling gan
        for criteria in list(self.check_config["perf_metrics"].keys()):   # criteria is either RMSE or MIE
            # find best config
            criteria_info = self.check_config["pf_metrics_setting"][criteria]
            perf_metrics = gan_metrics[criteria_info["metric_level"]]
            perf_values = criteria_info["agg_function"](perf_metrics[[criteria]])
            best_config = perf_metrics.index[criteria_info["rank_function"](perf_values)]  # the training step that has the best generator
            print("best "+criteria+" at iteration ",best_config) # just out of curiosity

            # generate samples of synthetic data    
            gan_samples_df = self.generate(best_config)
            # if curious, print out gan_sample_df and its shape and its header

            # export results
            self.save_model(gan_samples_df, "grid_" + criteria + ".pkl.gz")
        self.save_model(gan_metrics["agg_metrics"], "grid_checkmetrics.pkl.gz")


    def generate(self, iteration): # generating 50 samples and taking the mean
        iter_spacegan = get_spacegan_config(iteration, self.prob_config, self.check_config, self.cond_input, self.target)
        
        # training samples
        gan_samples_df = pd.DataFrame(index=range(self.cond_input.shape[0]), columns=self.cond_vars + self.neighbour_list + self.output_vars)
        gan_samples_df[self.cond_vars + self.neighbour_list] = self.cond_input
        gan_samples_df[self.output_vars] = self.target
        for i in range(self.check_config["n_samples"]):
            gan_samples_df["sample_" + str(i)] = iter_spacegan.predict(self.cond_input)
        return gan_samples_df
    

    def plotting(self, img_name): #haven't decided if i need this & don't know how to write this neatly
        return None


    def load_model(self, model_name): #seems useless tho
        return pd.read_pickle(self.model_save_prefix + model_name)

    def save_model(self, model, model_name): #this also seems useless...don't know if this would work...
        model.to_pickle(self.model_save_prefix + model_name)

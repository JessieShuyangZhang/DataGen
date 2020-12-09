import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
# import warnings
# warnings.simplefilter("ignore")
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import pdb
import sys
sys.path.append('src')
from spacegan_method import SpaceGAN
from spacegan_selection import get_spacegan_config, compute_metrics
from spacegan_utils import gaussian, rmse, mie, get_neighbours_featurize
from spacegan_config import Generator, Discriminator
from datagen_wrapper import DataGenWrapper

class SpaceganWrapper(DataGenWrapper):
    def __init__(self, output_vars=[],cond_vars=[], model_save_prefix='saved_models/noaa/', fig_save_prefix='img/noaa'): #prob_config, check_config, 
        self.model_save_prefix = model_save_prefix  # path to save the model
        self.fig_save_prefix = fig_save_prefix
        self.prob_config = {
            "epochs": 1000,
            "batch_size": 100,
            "device": torch.device("cuda"),
            "noise_type": gaussian,  # type of noise and dimension used
            "noise_params": None,  # other params for noise (loc, scale, etc.) pass as a dict
            "scale_x": StandardScaler(),  # a sklearn.preprocessing scaling method
            "scale_y": StandardScaler(),  # a sklearn.preprocessing scaling method
            "print_results": False,
            "neighbours": 50,
            # additional Generator params
            "gen_opt": torch.optim.SGD,
            "gen_opt_params": {"lr": 0.01},
            # additional Discriminator params
            "disc_opt": torch.optim.SGD,
            "disc_opt_params": {"lr": 0.01},
            # loss function
            "adversarial_loss": torch.nn.BCELoss()
        }
        self.check_config = {
            "check_interval": 1000,  # for model checkpointing
            "generate_image": False,
            "n_samples": 50,
            "perf_metrics": {"RMSE": rmse,
                            "MIE": mie,
                            },
            "pf_metrics_setting": {
                "RMSE": {"metric_level": "agg_metrics",
                    "rank_function": np.argmin,
                    "agg_function": lambda x: np.array(x)
                    },
                "MIE": {"metric_level": "agg_metrics",
                        "rank_function": np.argmin,
                        "agg_function": lambda x: np.array(x)
                    },
            },
            "agg_funcs": {"avg": np.mean,
                        "std": np.std
                        },
            "sample_metrics": False,
            "agg_metrics": True
        }
        self.df = None
        self.neighbours = self.prob_config["neighbours"]
        self.coord_vars = ["longitude", "latitude"]
        self.cond_vars = cond_vars + self.coord_vars
        self.cont_vars=['unix_time', 'depth', 'conductivity', 'density', 'temperature', 'salinity'] + self.coord_vars #fixed? always all of the variables  
        self.output_vars = output_vars   
        self.neighbour_list = [] # list of labels after augmentation by neighbours (eg. 'nn_houseValue_0', 'nn_houseValue_1'...'nn_latitude_8', 'nn_latitude_9')

        self.original = []
        self.synthetic = []

        # for building GAN
        self.target = []
        self.cond_input = []
        self.coord_input = []
        self.disc_method = None
        self.gen_method = None


    def build_dataset(self, csv_files, rows=None): # construct a dataframe from a csv file with a header of var names
        # csv_files is an array of files to be loaded 
        self.df = pd.read_csv(csv_files[0])
        for i in range(1, len(csv_files)):
            df2 = pd.read_csv(csv_files[0])
            self.df = pd.concat([self.df, df2],ignore_index = True)
        # pdb.set_trace()
        if('position_key' in self.df.columns): # remove the column of position_key
            self.df.drop('position_key', axis=1, inplace=True)
        # self.coord_vars = ["longitude", "latitude"] #Define spatial coordinates
        # self.cond_vars = cond_vars + self.coord_vars #Define the predictor variables
        # self.cont_vars_arr=['unix_time', 'depth', 'conductivity', 'density', 'temperature', 'salinity'] + self.coord_vars #fixed? always all of the variables
        # self.output_vars = output_vars
        # self.neighbours = neighbours
        self.prob_config["cond_dim"] = len(self.cond_vars) + (self.neighbours * len(self.cont_vars))
        self.prob_config["output_dim"] = len(self.output_vars)  # size of output
        self.prob_config["noise_dim"] = self.prob_config["cond_dim"]  # size of noise
        # pdb.set_trace()
        return self.df

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
        min_rmse_val = None
        min_mie_val = None
        # selecting and sampling gan
        for criteria in list(self.check_config["perf_metrics"].keys()):   # criteria is either RMSE or MIE
            # find best config
            criteria_info = self.check_config["pf_metrics_setting"][criteria]
            perf_metrics = gan_metrics[criteria_info["metric_level"]]
            perf_values = criteria_info["agg_function"](perf_metrics[[criteria]])
            # print("perf_values for "+criteria,perf_values) # just out of curiosity
            if(criteria == "RMSE"):
                min_rmse_val = perf_values[criteria_info["rank_function"](perf_values)]
                print("min perf_value for "+criteria, min_rmse_val)
            else:
                min_mie_val = perf_values[criteria_info["rank_function"](perf_values)]
                print("min perf_value for "+criteria, min_mie_val)

            best_config = perf_metrics.index[criteria_info["rank_function"](perf_values)]  # the training step that has the best generator
            print("best "+criteria+" at iteration ",best_config) # just out of curiosity

            # generate samples of synthetic data    
            gan_samples_df = self.generate_samples(best_config)
            # pdb.set_trace()
            # if curious, print out gan_sample_df and its shape and its header

            # export results
            self.save_model(gan_samples_df, "grid_" + criteria + ".pkl.gz")
        self.save_model(gan_metrics["agg_metrics"], "grid_checkmetrics.pkl.gz")


    def generate_samples(self, iteration): # generating 50 samples and taking the mean
        iter_spacegan = get_spacegan_config(iteration, self.prob_config, self.check_config, self.cond_input, self.target)
        
        # training samples
        gan_samples_df = pd.DataFrame(index=range(self.cond_input.shape[0]), columns=self.cond_vars + self.neighbour_list + self.output_vars)
        gan_samples_df[self.cond_vars + self.neighbour_list] = self.cond_input
        gan_samples_df[self.output_vars] = self.target
        for i in range(self.check_config["n_samples"]):
            gan_samples_df["sample_" + str(i)] = iter_spacegan.predict(self.cond_input)
        return gan_samples_df
    

    def plot(self, img_name, title, generated_seq, dataframe=None): #haven't decided if i need this & don't know how to write this neatly

        # plotting observed house value distrubution at lon-lat location
        if(dataframe is not None):
            fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(15, 5))
            dataframe.plot(ax=ax1)
            #set title for ax1 ?
        else:
            fig, ax2 = plt.subplots(1, 1, figsize=(7, 5))
            
        gen_seq = generated_seq
        norm_gan_mean = (gen_seq - min(gen_seq)) / (max(gen_seq) - min(gen_seq))
        colors = cm.rainbow(norm_gan_mean)

        # plotting
        for lat, long, c in zip(self.df["latitude"], self.df["longitude"], colors):
            ax2.scatter(lat, long, color=c, s=5)          # s denotes marker size
        ax2.set_xlabel(r'$c^{(1)}$', fontsize=14)
        ax2.set_ylabel(r'$c^{(2)}$', fontsize=14)
        ax2.set_title(title)
        fig.savefig(self.fig_save_prefix+img_name)


    def load_model(self, model_name): #seems useless tho
        return pd.read_pickle(self.model_save_prefix + model_name)

    def save_model(self, model, model_name): #this also seems useless...don't know if this would work...
        model.to_pickle(self.model_save_prefix + model_name)

    def set_prob_config_param(param, value):
        self.prob_config[param] = value
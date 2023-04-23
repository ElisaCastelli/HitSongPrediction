import os
import time
import numpy as np
import pandas as pd
#from utils import global_variables as gv
#from utils.logger import init_logger
from functools import reduce
from sklearn.model_selection import StratifiedKFold,KFold, train_test_split
from models.my_model import *
from models.model_selection import select_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from models.model_selection import *

class PopularityPrediction:
    def __init__(self):
        self.df_final = pd.read_csv("./input/df_final.csv")
        self.X = None
        self.y = None
        self.X_test = None
        self.y_test = None
        self.X_sc = None
        self.y_sc = None
        self.scaler = None
        self.task = 'regression'
        self.target = 'popularity'
        self.model_args = select_model(model_val=1)
        #self.mode = None
        self.dl_model = None
        self.setup_model()

    def setup_model(self):
        ok = True
        try:
            self.dl_model = MyModel(model_name = self.model_args['model_name'],
                                              model_dir = self.model_args['model_dir'],
                                              model_subDir = self.model_args['model_subDir'],
                                              input_dim = self.model_args['input_dim'],
                                              output_dim = self.model_args['output_dim'],
                                              optimizer=self.model_args['optimizer'],
                                              metrics=self.model_args['metrics'],
                                              loss=self.model_args['loss'],
                                              add_earlyStopping=self.model_args['earlyStop'],
                                              weights_path=self.model_args['weights'],
                                              neuron_parameters=self.model_args['neurons'],
                                              layers=self.model_args['n_layers'],
                                              initialization=self.model_args['weights_init'],
                                              level_dropout=self.model_args['dropout'],
                                              problem=self.task)
        except Exception as e:
            #self.logger.error(e)
            print(e)
            ok = False
        return ok

    def data_preprocessing(self, compressed=True, compressed_method='AE'):
        ok = True
        try: 
            tracks_id = self.df_final[['track_id']]
            tracks_id.to_csv(os.path.join('./input', 'tracks_id_' + self.task + '.csv'))
            #self.logger.info('Pre-processing data for %s problem ...', self.task)
            drop_cols = ['album_id', 'analysis_url', 'artists_id',
                                'available_markets', 'country', 'disc_number', 'id',
                                'lyrics','name', 'playlist','preview_url',
                                'track_href', 'track_name_prev', 'track_number',
                                'uri','time_signature','href', 'track_id', 'type']
            check_cols = [col for col in drop_cols if col in list(self.df_final.columns)]

            self.df_final.drop(check_cols, axis=1, inplace=True)

            self.y = self.df_final.loc[:, self.target]
            self.X = self.df_final.drop(self.target, axis=1)

            self.scaler = MinMaxScaler(feature_range=(0, 1))
            self.X_sc = self.scaler.fit_transform(self.X)
            self.y_sc = self.scaler.fit_transform(np.array(self.y).reshape(-1, 1))
            self.y_sc = pd.DataFrame(self.y_sc, columns = [self.target])
            self.X_sc = pd.DataFrame(self.X_sc, columns = self.X.columns) #convert to dataframe

            if compressed:
                if compressed_method =='AE':
                    self.x_compressed = self.apply_feature_compression()
                    # Scale feature vector from autoencoder
                    
                    self.x_compressed = self.scaler.fit_transform(self.x_compressed)
                    cols = ['feat_compressed_' + str(i+1) for i in range(self.x_compressed.shape[1])]
                    self.X_sc = pd.DataFrame(self.x_compressed, columns=cols)

        except Exception as e:
            ok = False
            print(e)
            #self.logger.error(e)
        return ok

    def feature_compression(self):
        x_compressed = None
        try:
            encoder = self.dl_model.train_autoencoder(x_train=self.X_sc, epochs=50,
                                                      compress=self.compress)
            x_compressed = encoder.predict(self.X_sc)
        except Exception as e:
            #self.logger.error(e)
            print(e)

        return x_compressed

    def train_model(self):
        pass

    def predict(self):
        pass

    def evaluate(self):
        pass






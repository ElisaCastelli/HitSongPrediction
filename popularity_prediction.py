import os
import time
import numpy as np
import pandas as pd
# from utils import global_variables as gv
# from utils.logger import init_logger
from functools import reduce
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from models.my_model import *
from models.model_selection import select_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from models.model_selection import *


class PopularityPrediction:
    def __init__(self):
        self.df_final = pd.read_csv("./input/df_final_year.csv")
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
        # self.mode = None

        self.compress = 'auto'
        self.dl_model = None
        self.model_history_cv = {}
        self.output = {}
        self.setup_model()

    def setup_model(self):
        ok = True
        try:
            self.dl_model = MyModel(model_name=self.model_args['model_name'],
                                    model_dir=self.model_args['model_dir'],
                                    model_subDir=self.model_args['model_subDir'],
                                    input_dim=self.model_args['input_dim'],
                                    output_dim=self.model_args['output_dim'],
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
            # self.logger.error(e)
            print(e)
            ok = False
        return ok

    def data_preprocessing(self, compressed=True, compressed_method='AE'):
        ok = True
        try:
            tracks_id = self.df_final[['track_id']]
            tracks_id.to_csv(os.path.join('./input', 'tracks_id_' + self.task + '.csv'))
            # self.logger.info('Pre-processing data for %s problem ...', self.task)
            drop_cols = ['album_id', 'analysis_url', 'artists_id',
                         'available_markets', 'country', 'disc_number', 'genres', 'id',
                         'lyrics', 'name', 'playlist', 'preview_url',
                         'track_href', 'track_name_prev', 'track_number',
                         'uri', 'time_signature', 'href', 'track_id', 'type']
            check_cols = [col for col in drop_cols if col in list(self.df_final.columns)]

            self.df_final.drop(check_cols, axis=1, inplace=True)

            self.y = self.df_final.loc[:, self.target]
            self.X = self.df_final.drop(self.target, axis=1)

            self.scaler = MinMaxScaler(feature_range=(0, 1))
            self.X_sc = self.scaler.fit_transform(self.X)
            self.y_sc = self.scaler.fit_transform(np.array(self.y).reshape(-1, 1))
            self.y_sc = pd.DataFrame(self.y_sc, columns=[self.target])
            self.X_sc = pd.DataFrame(self.X_sc, columns=self.X.columns)  # convert to dataframe

            if compressed:
                if compressed_method == 'AE':
                    self.x_compressed = self.feature_compression()
                    # Scale feature vector from autoencoder

                    self.x_compressed = self.scaler.fit_transform(self.x_compressed)
                    cols = ['feat_compressed_' + str(i + 1) for i in range(self.x_compressed.shape[1])]
                    self.X_sc = pd.DataFrame(self.x_compressed, columns=cols)

        except Exception as e:
            ok = False
            print(e)
            # self.logger.error(e)
        return ok

    def feature_compression(self):
        x_compressed = None
        try:
            encoder = self.dl_model.train_autoencoder(x_train=self.X_sc, epochs=50,
                                                      compress=self.compress)
            x_compressed = encoder.predict(self.X_sc)
        except Exception as e:
            # self.logger.error(e)
            print(e)

        return x_compressed

    def train_model(self, n_splits=5, shuffle_data=True):
        ok = True
        try:
            # --------------------------------------------
            tracks_id = pd.read_csv(os.path.join('./input', 'tracks_id_' + self.task + '.csv'),
                                    index_col=0)

            # Separate into Train and Test
            x_total = pd.concat([self.X_sc, self.y_sc, tracks_id], axis=1)
            train, test = train_test_split(x_total, test_size=0.10)

            self.y_train = train.loc[:, self.target]
            self.x_train = train.drop([self.target], axis=1, inplace=False)
            self.y_test = test.loc[:, self.target]
            self.x_test = test.drop([self.target], axis=1, inplace=False)

            # Save test
            self.x_test.to_csv(os.path.join('./input', 'x_test_' + self.task + '.csv'))
            self.y_test.to_csv(os.path.join('./input', 'y_test_' + self.task + '.csv'))

            # Remove ids
            self.x_train.drop(['track_id'], axis=1, inplace=True)
            self.x_test.drop(['track_id'], axis=1, inplace=True)
            # ----------------------------------------------

            # 1) Choose Cross-validation method
            if self.task == 'classification':
                skf = StratifiedKFold(n_splits=n_splits)
                splits = skf.split(self.x_train, pd.DataFrame(self.y_train, columns=[self.target]))
            else:
                skf = KFold(n_splits=n_splits, shuffle=shuffle_data)
                splits = skf.split(self.x_train.index)

            if self.dl_model is None:
                self.set_up_model()

            # Change input Dimension and build model
            self.dl_model.input_dim = self.x_train.shape[1]
            done = self.dl_model.build_model()

            if done:
                start = time.time()
                for index, (train_indices, val_indices) in enumerate(splits):
                    #
                    # self.logger.info("Training on fold %s/%s", index+1, n_splits)

                    # Generate batches from indices
                    x_train, x_val = np.array(self.x_train.iloc[train_indices]).round(3), \
                        np.array(self.x_train.iloc[val_indices]).round(3)
                    y_train, y_val = np.array(self.y_train.iloc[train_indices]).round(3), \
                        np.array(self.y_train.iloc[val_indices]).round(3)

                    # Compile the model
                    self.dl_model.compile_model()

                    # Train the DL model
                    model_history = self.dl_model.train_model(x_train, y_train,
                                                              x_val, y_val,
                                                              epochs=self.model_args['epochs'],
                                                              batch_size=self.model_args['batch_size'],
                                                              monitor_early=self.model_args['early_mon'],
                                                              mode=self.model_args['mode'],
                                                              monitor_checkout=self.model_args['checkout_mon'])

                    # self.logger.info('Saving Model History ... ')
                    history = model_history.history
                    # Save Metrics
                    for metric in list(history.keys()):
                        self.model_history_cv[metric] = history[metric][-1]

                    # Save Model and History
                    self.dl_model.save_model()
                    self.dl_model.save_history(model_history, k=index)
                    ok = True

                end = time.time()

                self.model_history_cv['cpu (ms)'] = end - start

                # Predict Function
                # self.logger.info('Calculating predictions using x_test ... ')
                y_pred = self.dl_model.validate_model(X_test=self.x_test)

                mae_test = mean_absolute_error(y_true=np.array(self.y_test), y_pred=y_pred)
                self.model_history_cv['test_mean_absolute_error'] = mae_test

                # Summary results
                self.output = self.dl_model.get_model_summary(history=self.model_history_cv, task=self.task,
                                                              k_fold=n_splits)
            else:
                ok = False
        except Exception as e:
            print(e)
            # self.logger.error(e)
            ok = False
        return ok

    def run(self):
        output = {''}
        try:
            # 1) Train the Neural Network
            # if self.mode == 'Train':
            # 1) Data Pre-processing
            ok = self.data_preprocessing()
            if ok:
                ok = self.train_model()
                output = self.output
                print(self.model_history_cv)
            # Predict
            else:
                # Load Model and predict
                pass
        except Exception as e:
            # self.logger.error(e)
            print(e)
            output = {''}

        return output


pp = PopularityPrediction()
print(pp.run())

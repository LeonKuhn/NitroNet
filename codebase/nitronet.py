# Copyright (c) 2025 Leon Kuhn
#
# This file is part of a software project licensed under a Custom Non-Commercial License.
#
# You may use, modify, and distribute this code for non-commercial purposes only.
# Use in academic or scientific research requires prior written permission.
#
# For full license details, see the LICENSE file or contact l.kuhn@mpic.de

import torch

from torch import nn
import tomli
import os
import pickle
import numpy as np
import json
from scipy.interpolate import interp1d
import pandas as pd
import sklearn
from sklearn.preprocessing import QuantileTransformer, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import InconsistentVersionWarning
import warnings

warnings.filterwarnings("error")  # in order to be able to catch warnings as errors

class Normalizer(BaseEstimator, TransformerMixin):
    # A normalizer class, which subtracts and multiplies a feature by a hardcoded value.

    def __init__(self, subtract=None, multiply=None):
        self.subtract = subtract
        self.multiply = multiply

    def fit(self, X, y=None):
        if self.subtract is None:
            self.subtract = np.nanquantile(X, 0.01)
        if self.multiply is None:
            self.multiply = 1 / np.nanquantile(X - self.subtract, 0.99)
        return self

    def transform(self, X):
        X_ = X.copy()
        X_ = (X_ - self.subtract) * self.multiply
        return X_

    def inverse_transform(self, X):
        X_ = X.copy()
        X_ = (X_ / self.multiply) + self.subtract
        return X_

class FeedForwardNetwork(nn.Module):
    """
    Simple dense Feed Forward network with variable number of hidden layers L and number of neurons per hidden
    layer n_hidden.
    """

    def __init__(self, hparams, device="cpu"):

        super().__init__()

        # assign hyper parameters
        self.hparams = hparams

        self.n_in = len(self.hparams["X_NAMES"]) - len([x for x in self.hparams["X_NAMES"] if "alt_highres_" in x]) + 1
        self.n_hidden = self.hparams["N_HIDDEN"]
        self.n_out = 1
        self.L = self.hparams["L_HIDDEN"]
        self.activation_fct = self.hparams["ACTIVATION_FNCT"]
        self.drop_out_rate = self.hparams["DROP_OUT_RATE"]
        self.batch_norm = self.hparams["BATCH_NORM"]
        self.X_names = self.hparams["X_NAMES"]
        self.device = device

        self.layers = []

        # add input layer
        self.layers.append(nn.Linear(self.n_in, self.n_hidden, bias=False))

        # add hidden layers
        for i in range(self.L):
            self.layers.append(nn.Linear(self.n_hidden, self.n_hidden))
            if self.batch_norm: self.layers.append(nn.BatchNorm1d(self.n_hidden))
            self.layers.append(self.activation_fct())

            if self.drop_out_rate != 0: self.layers.append(nn.Dropout(p=self.drop_out_rate))

        # add output layer
        self.layers.append(nn.Linear(self.n_hidden, self.n_out))
        self.linear_relu_stack = nn.Sequential(*self.layers)

    def forward(self, x):

        # Pass through model layers
        for l in self.layers:
            x = l(x)
        return x

class ProfileModel:
    """
    The core class of NitroNet.
    The ProfileModel class takes care of
    - instantiating the neural network
    - converting input data in a user-friendly format to suitable input matrices for the neural network
    - transforming input and output data of the neural network
    """

    def parse_hparams_from_json(self, hparams):

        # X-names
        hparams["X_NAMES"] = hparams["X_NAMES"].split(",")
        hparams["X_NAMES"] = [x.replace("'", "").replace("[", "").replace(" ", "").replace("]", "") for x in hparams["X_NAMES"]]

        hparams["Y_NAMES"] = hparams["Y_NAMES"].split(",")
        hparams["Y_NAMES"] = [x.replace("'", "").replace("[", "").replace(" ", "").replace("]", "") for x in hparams["Y_NAMES"]]

        # activation function
        fct_name = hparams["ACTIVATION_FNCT"].replace("<class '", "").replace("'>", "")
        match fct_name:
            case 'torch.nn.modules.activation.PReLU':
                hparams["ACTIVATION_FNCT"] = torch.nn.modules.activation.PReLU
            case _:
                print("Error while parsing the hyperparameter 'ACTIVATION_FNCT'")

        return hparams

    def __init__(self, model_directory, device="cpu"):

        self.usage_params = {}  # e.g. at which altitude ranges to use

        # load torch neural network and put it in evaluation mode
        with open(os.path.join(model_directory, "hparams.json"), "r") as input_file:
            self.hparams = json.load(input_file)

        # when loading the model, the hparam file must be parsed
        self.hparams = self.parse_hparams_from_json(self.hparams)

        self.model_directory = model_directory
        self.name = self.model_directory.split("/")[-2]
        self.nn = FeedForwardNetwork(self.hparams, device=device)
        self.nn = self.nn.to(device)
        self.nn.load_state_dict(torch.load(os.path.join(self.model_directory, "weights.pt"), weights_only=True))
        self.nn.eval()

        # load the LUT for bias correction (whether the bias correction is actually applied is decided later)
        try:
            self.bias_correction_profile = pd.read_csv(os.path.join(self.model_directory, "bias_correction.csv"))[["layer", "altitude", "bias_corr_factor"]]
            self.bias_correction_interpolator = interp1d(self.bias_correction_profile["altitude"], self.bias_correction_profile["bias_corr_factor"], bounds_error=False, fill_value="extrapolate")
        except:
            self.bias_correction_profile, self.bias_correction_interpolator = None, None
            print(f"Warning: No bias correction LUT found for the model {self.name}")

        # load the LUT for neural network error estimation
        try:
            self.nn_error_profile = pd.read_csv(os.path.join(self.model_directory, "error_diagnostic.csv"))[["layer", "altitude", "MAPE"]]
            self.nn_error_profile["MAPE"] = self.nn_error_profile["MAPE"].apply(lambda x: float(x.split(" %")[0]))
            self.nn_error_interpolator = interp1d(self.nn_error_profile["altitude"], self.nn_error_profile["MAPE"], bounds_error=False, fill_value="extrapolate")
        except Exception as e:
            self.nn_error_profile, self.nn_error_interpolator = None, None
            print(f"Warning: No neural network error LUT found for the model {self.name}")

        # load the transformations
        with open(os.path.join(self.model_directory, "transformations/transformations.toml"), mode="rb") as fp:
            transforms_toml = tomli.load(fp)

        def transformers_as_list(transforms_toml, var_names):
            """
            Return the transformers of a list of variable names in a list
            """

            def get_transformer_object(toml_file_obj, name):

                tomli_entry = toml_file_obj[name]
                transformer_type = tomli_entry[0]
                args = tomli_entry[1]

                match transformer_type:
                    case "Normalizer":
                        return Normalizer(subtract=args[0], multiply=args[1])
                    case "lambda":
                        return FunctionTransformer(
                            func=lambda x: eval(args[0].split('lambda x:')[-1]),
                            inverse_func=lambda x: eval(args[1].split('lambda x:')[-1])
                        )
                    case "QuantileTransformer":
                        with open(os.path.join(model_directory, f"transformations/{args[0]}"), 'rb') as f:
                            try:
                                return pickle.load(f)
                            except InconsistentVersionWarning:
                                """
                                InconsistentVersionWarning may be raised, because the QuantileTransformers were saved
                                with sklearn version 1.1.1. In order to ensure consistency across versions, we perform
                                a sanity test by loading the *.test files in the transformations directory
                                """

                                test_data = np.loadtxt(os.path.join(model_directory, f"transformations/{args[0].replace('.qtrans', '.test')}"))

                                with warnings.catch_warnings():

                                    warnings.simplefilter("ignore")
                                    q_tf = pickle.load(f)

                                    assert np.array_equal(q_tf.transform(test_data[:, 0].reshape(-1, 1)), test_data[:, 1].reshape(-1, 1)), f"sklearn (here: {sklearn.__version__}) appears to be unable to correctly unpickle the QuantileTransformers of NitroNet (produced with version 1.1.1)."

                                    return q_tf
                    case "identity":
                        return FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x)

            transformer_list = []
            for xn in var_names:
                try:
                    # for non-vertically resolved inputs
                    transformer_list.append(get_transformer_object(transforms_toml, xn))
                except:
                    # for vertically resolved inputs, remove layer index
                    _splits = xn.split("_")
                    transformer_list.append(get_transformer_object(transforms_toml, "_".join(_splits[:-1])))

            return transformer_list

        self.X_transformers = transformers_as_list(transforms_toml, [var for var in self.hparams["X_NAMES"] if "alt_highres" not in var])
        self.z_transformers = transformers_as_list(transforms_toml, [var for var in self.hparams["X_NAMES"] if "alt_highres" in var])
        self.y_transformers = transformers_as_list(transforms_toml, [var for var in self.hparams["Y_NAMES"] if "NO2_conc_sim_highres" in var or "F-value" in var])

    def expand_feature_matrix(self, X, *, z_columns):
        """
        Expand a feature matrix, where more than a single altitude per instance is given
        (see expand_features_targest in codebase.datafunctions.py)
        :param X: The input matrix to transform
        :param z_columns: The number of altitudes given per instance
        """

        X_expanded = np.repeat(X[:, :-z_columns], z_columns, axis=0)
        z_expanded = X[:, -z_columns:].reshape(-1, 1)
        return np.hstack([X_expanded, z_expanded])

    def unexpand_prediction_vector(self, y, *, z_columns):
        """
        Inverts the functionality of expand_feature_matrix, but on a vector of predictions y.
        """

        return y.reshape(-1, z_columns)

    def transform_feature_matrix(self, Xz, *, transform_X, transform_z):
        """
        Transforms a given feature matrix using the stores transformers
        """

        if transform_X:
            for idx, tf in enumerate(self.X_transformers):
                if type(tf) == QuantileTransformer:
                    Xz[:, idx] = tf.transform(Xz[:, idx].reshape(-1, 1)).reshape(-1)
                else:
                    Xz[:, idx] = tf.transform(Xz[:, idx])

        if transform_z:
            Xz[:, -1] = self.z_transformers[0].transform(Xz[:, -1])

        return Xz

    def X_names(self):
        return [var for var in self.hparams["X_NAMES"] if "alt_highres_" not in var]

    def __call__(self, _X, z, batchsize, return_z=False):
        """
        A wrapper around the forward pass through the neural network at the core of this model.
        _X is the input matrix
        z is the vertical target grid
        If z is a 1-D np.ndarray, it is assumed as the vertical grid for all instances in _X.
        If z is a higher dimensional np.ndarary, it is hstacked to _X.
        if return_z, the function returns z along with the prediction results (helpful, because it returns the *broadcasted* version of z if z is 1-D)
        """

        print(f"Forward pass through '{self.name}'", end=": ")

        # ensure input to be numpy arrays
        if type(_X) is torch.Tensor:
            _X = _X.detach().numpy()
        if type(z) is torch.Tensor:
            z = z.detach().numpy()

        # ensure always the same shape
        if _X.ndim == 1:
            _X = _X.reshape(1, -1)

        def process_batch(_X, z):

            # ensure always the same shape
            if _X.ndim == 1:
                _X = _X.reshape(1, -1)

            if z.ndim == 1:
                # a single z array that must be broadcasted along _X
                z = z.reshape(1, -1)
                z = np.repeat(z, _X.shape[0], axis=0)

                # concatenate _X and z
                _X = np.hstack([_X, z])

            # expand the concatenated input matrix _X
            _X = self.expand_feature_matrix(_X, z_columns=z.shape[1])

            # apply variable transformations
            _X = self.transform_feature_matrix(_X, transform_X=True, transform_z=True)

            # convert to torch tensor
            _X = _X.astype(np.float32)

            _X = torch.tensor(_X, device=self.nn.device)

            # neural network forward pass and backwards transformation
            y = self.nn(_X)

            y = y.cpu().detach().numpy()
            y = self.unexpand_prediction_vector(y, z_columns=z.shape[1])

            return y

        y_shape = (_X.shape[0], z.shape[0] if z.ndim == 1 else z.shape[-1])
        y = np.zeros(y_shape)
        for start_idx in range(0, _X.shape[0], batchsize):
            print(f"{100 * start_idx / _X.shape[0]:.0f} % - ", end="", flush=True)
            end_idx = min(start_idx + batchsize, _X.shape[0])

            X_indexed = _X[start_idx:end_idx]

            if np.squeeze(z).ndim == 1:
                z_indexed = z
            else:
                z_indexed = z[start_idx:end_idx]

            y[start_idx:end_idx] = process_batch(X_indexed, z_indexed)
        print("100 %")

        # inverse transform of y
        y = self.y_transformers[0].inverse_transform(y)

        if y.shape[0] == 1:
            y = y.flatten()

        if return_z:
            try:
                return y, z.reshape(y.shape)
            except:
                return y, z * np.ones_like(y)

        return y

class ProfileModelGroup:
    """
    A class for a combination of ProfileModels, which make predictions on disjoint partitions of the z-axis
    A ProfileModelGroup is essentially a list of ProfileModels, each making predictions for different vertical regions.
    The ProfileModelGroup class takes care of
    - controlling its contained ProfileModels
    - applying empirical bias correction
    """

    def predict_using_submodel(self, model, _X, _z, X_names, batchsize, alt_min, alt_max):
        """
        Helper function that predicts using a single sub-model of the ProfileModelGroup
        :return:
        """

        access_indices = [X_names.index(x) for x in model.hparams["X_NAMES"] if "alt_highres" not in x]
        if _X.ndim == 1:
            _X = _X.reshape(1, -1)
        y, z = model(
            _X[:, access_indices], z=_z, batchsize=batchsize,
            return_z=True
        )
        mask = ((z >= alt_min) & (z < alt_max))
        y[~mask] = 0
        return y

    def __init__(self, submodel_dict, device="cpu"):

        self.model_dict = {}

        for e, (submodel_name, submodel_params) in enumerate(submodel_dict.items()):

            submodel = ProfileModel(submodel_params["dir"], device=device)
            self.model_dict[submodel_name] = submodel
            self.model_dict[submodel_name].usage_params["alt_min"] = submodel_params["alt_min"]
            self.model_dict[submodel_name].usage_params["alt_max"] = submodel_params["alt_max"]

            # the first model in NitroNet's TOML configuration file is automatically the 'main' model
            if e == 0:
                self.main_model = self.model_dict[submodel_name]
                print(f"Using {self.main_model.model_directory} as the 'main' model")

    def __call__(self, _X, z, X_names, batchsize, use_bias_correction=True, print_prefix=""):

        predictions_y = []

        for model_name, model_obj in self.model_dict.items():
            print(print_prefix, end="")
            y = self.predict_using_submodel(
                model_obj,
                _X, z, X_names, batchsize,
                model_obj.usage_params["alt_min"], model_obj.usage_params["alt_max"],
            )
            predictions_y.append(y)

        prediction = np.stack(predictions_y).sum(axis=0)

        if use_bias_correction:
            bias_corrector = self.bias_correction_interpolator()
            prediction = prediction * bias_corrector(z)

        return prediction

    def X_names(self):
        return self.main_model.X_names()

    def bias_correction_interpolator(self):
        return self.main_model.bias_correction_interpolator

    def nn_error_interpolator(self):
        return self.main_model.bias_correction_interpolator

    def kde_dict_path(self):
        return os.path.join(self.main_model.model_directory, "kde_dict.pkl")

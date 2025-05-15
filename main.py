# Copyright (c) 2025 Leon Kuhn
#
# This file is part of a software project licensed under a Custom Non-Commercial License.
#
# You may use, modify, and distribute this code for non-commercial purposes only.
# Use in academic or scientific research requires prior written permission.
#
# For full license details, see the LICENSE file or contact l.kuhn@mpic.de

import os
import sys
from copy import deepcopy
from datetime import datetime, timedelta
from glob import glob
from itertools import chain
import netCDF4  # do not remove; for some reason netCDF4 must be explicitly imported before numpy
import xarray as xr
import pickle
import numpy as np
import pandas as pd
import tomli
import torch
import multiprocessing as mp

from codebase.nitronet import ProfileModelGroup, ProfileModel

# parse args
conf_file = sys.argv[1]
slurm_array_task_id = int(sys.argv[2])

# load config
with open(conf_file, mode="rb") as fp:
    c = tomli.load(fp)
    conf = {**c["general"], **c["main"], **c}
    del c

# determine device
if conf["use_cuda"] and torch.cuda.is_available():  # if using CUDA
    cuda = True
    device = "cuda:0"
    print("This NitroNet run uses device 'cuda'", flush=True)

elif conf["use_cuda"] and torch.version.hip is not None:  # if using ROCM
    cuda = True
    device = "cuda:0"

else:
    cuda = False
    device = "cpu"
    if conf["use_cuda"]:
        print(f"Warning: 'use_cuda = true' in {conf_file}, but torch.cuda.is_available() == False and torch.version.hip is None.", flush=True)
    print("This NitroNet run uses device 'cpu'", flush=True)

# load model
model = ProfileModelGroup(conf["submodels"], device=device)

# if an F-model was specified, initialize it
if "dir" in conf["main"]["F-model"]:
    F_model = ProfileModel(conf["F-model"]["dir"], device=device)
else:
    F_model = None

# load the model's KDE info
with open(model.kde_dict_path(), 'rb') as f:
    kde_dict = pickle.load(f)

# set random seed
np.random.seed(conf["random_seed"])


def get_local_density_vec(var, X, hist_dict):

    X_kde = hist_dict[var]["x"]
    Y_kde = hist_dict[var]["y"]

    below_min = (X <= min(X_kde))
    above_max = (X >= max(X_kde))
    is_nan = (np.isnan(X))

    Y = np.ones_like(X) * np.nan
    Y[below_min] = 0
    Y[above_max] = 0

    def find_lower_idx(x):
        return Y_kde[np.max(np.argwhere(X_kde <= x))]

    find_lower_idx_vectorized = np.vectorize(find_lower_idx, otypes=[float])
    Y[(~is_nan) & (~below_min) & (~above_max)] = find_lower_idx_vectorized(X[(~is_nan) & (~below_min) & (~above_max)])

    return Y

def process_single_input_file(model, filename, F_model=None, output_dir=None):
    """
    Takes a NitroNet input file in netCDF format and computes the corresponding prediction
    """

    print(f"Processing {filename}", flush=True)
    ds = xr.open_dataset(filename)
    orbit = filename.split("/")[-1].split(".")[0].split("_")[-1]

    # if chosen, perform error propagation based on the uncertainty of the NO2 VCDs
    if conf["uncerts"]:
        unit_conversion_factor = 6.022141e+19 / 1E16
        VCD_err = ds["nitrogendioxide_tropospheric_column_precision"].data.flatten() * unit_conversion_factor
        uncerts = {"NO2_tropcol_sat": VCD_err}
    else:
        uncerts = None

    # convert dataset to 2D input data
    def ds_to_table(ds):

        number_instances = ds.scanline.size * ds.ground_pixel.size  # Total amount of instances in final dataset
        shape = (ds.scanline.size, ds.ground_pixel.size)

        def dataarray_to_1d(da, shape):
            """
            Transform a dataarary of a dataset to a 1d np.ndarray. If the dataarray contains vertical resolution (N levels)
            the function will return N such 1d np.ndarrays.
            """

            vertical_levels_tm5 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

            if var_name in ["number", "pressure_level"]:
                return [], []

            # variable name blacklist
            if da.name in ["corner", "polynomial_exponents",
                           "intensity_offset_polynomial_exponents", "layer", "vertices", "level", "delta_time",
                           "time_utc", "nitrogendioxide_tropospheric_columns_precision_kernel", "cloud_albedo",
                           "wind_u", "wind_v", "averaging_kernel", "date", "u", "v", "glat", "glon", "time",
                           "tm5_constant_a", "tm5_constant_b"]:
                return [], []

            if da.name == "scanline":
                da = da.expand_dims(dim={"ground_pixel": shape[1]}, axis=1)
            if da.name == "ground_pixel":
                da = da.expand_dims(dim={"scanline": shape[0]}, axis=0)

            try:
                assert (da.dims[0] == "scanline")
                assert (da.dims[1] == "ground_pixel")
                assert (da.dims[-1] in ["ground_pixel", "pressure_level", "layer"])

            except Exception:
                print(f"Skipping variable '{var_name}' due to its array dimensions", flush=True)
                print(f"Array dimensions were {da.dims} but must be exactly: ['scanline', 'ground_pixel'] or ['scanline', 'ground_pixel', 'pressure_level / layer']", flush=True)
                return [], []

            try:
                tm5_layer_idx = da.dims.index("layer")
            except ValueError:
                tm5_layer_idx = None

            try:
                ERA5_layer_idx = da.dims.index("pressure_level")
            except ValueError:
                ERA5_layer_idx = None

            result_arrays = []
            csv_headers = []

            if tm5_layer_idx:
                # if variable is resolved on the tm5 layers
                for layer_idx in vertical_levels_tm5:
                    da_at_layer = da.isel(layer=layer_idx).data.flatten()
                    result_arrays.append(da_at_layer)
                    csv_headers.append(f"{da.name}_{layer_idx}")

            elif ERA5_layer_idx:
                # if variable is resolved on the ERA5 pressure levels
                for layer_idx in range(da["pressure_level"].size):
                    da_at_layer = da.isel(pressure_level=layer_idx).data.flatten()
                    result_arrays.append(da_at_layer)
                    csv_headers.append(f"{da.name}_{layer_idx}")

            else:
                result_arrays.append(da.data.flatten())
                csv_headers.append(f"{da.name}")

            for h, arr in zip(csv_headers, result_arrays):
                assert (arr.size == number_instances)
            return csv_headers, result_arrays

        all_csv_headers = []
        all_arrays = []

        for var_name in ds.variables:
            csv_headers, result_arrays = dataarray_to_1d(ds[var_name], shape)
            all_csv_headers.append(csv_headers)
            all_arrays.append(result_arrays)

        all_csv_headers = list(chain(*all_csv_headers))
        all_arrays = list(chain(*all_arrays))

        table = pd.DataFrame(np.array(all_arrays).T, columns=all_csv_headers)

        return table

    tabular_ds = ds_to_table(ds)

    # the columns of X must be put in correct oders (same as used when training neural network)
    def table_to_model_input(model, tabular_ds):

        model_input = tabular_ds[[x for x in model.X_names() if x in tabular_ds.columns]]

        missing_variables = []
        for n in model.X_names():
            if n not in model_input.columns and "alt_highres_" not in n:
                missing_variables.append(n)

        if missing_variables != []:
            raise Exception(f"Model input variables '{missing_variables}' were not found in the tabular dataset")

        return model_input

    X = table_to_model_input(model, tabular_ds)
    input_X_names = list(X.columns)
    X = X.to_numpy()

    # if chosen, replace outliers
    def winsorize(_model, _X, kde_dict, method, threshold, print_info=False):

        assert method in ["skip", "remove", "replace", "ignore"]

        X = deepcopy(_X)

        statistics_dict = {}

        if method == "skip":
            # essentially skips the winsorization entirely
            # 'ignore' would do the same, but still compute the densities in order to obtain statistical information
            return X, np.ones_like(X) * np.nan, None

        # first, obtain the local densities of the input features in the training set
        local_feature_densities = np.zeros_like(X)
        for i, var_name in enumerate([name for name in _model.X_names() if "alt_highres" not in name]):
            local_density = get_local_density_vec(var_name, X[:, i], kde_dict)
            local_feature_densities[:, i] = local_density

        if method == "remove":
            # replace instances with low local density by np.nan
            X[local_feature_densities <= threshold] = np.nan

        elif method == "replace":
        # sample from the priors to replace X with low local density
            for e, var_name in enumerate([name for name in _model.X_names() if "alt_highres" not in name]):
                if var_name in conf["winsor_blacklist"]:
                    continue

                samples = np.random.choice(.5 * (kde_dict[var_name]["x"][1:] + kde_dict[var_name]["x"][:-1]),
                                           size=X[:, e].shape[0],
                                           p=kde_dict[var_name]["y"] / np.sum(kde_dict[var_name]["y"]))

                X[:, e] = np.where(local_feature_densities[:, e] < threshold, samples, X[:, e])

        if method == "replace" or method == "ignore":
            """
            print some information on the sample replacement process:
            
            nr_replaced_entries: number of single matrix elements of X which were replaced
            replaced_entry_ratio: ratio of single matrix elements of X which were replaced
            ^^^ this does not take np.nan entries into account, which remain untouched by the winsorization ^^^
            
            nr_replaced_rows: number of matrix rows with at least one element replaced
            replaced_row_ratio: ratio if matrix rows with at least one element replaced
            
            """

            is_valid_row = ~np.isnan(np.sum(X, axis=1))
            nr_valid_rows = np.count_nonzero(is_valid_row)

            if nr_valid_rows == 0:
                # print("Warning: Input matrix contained no valid instances", flush=True)
                return X, local_feature_densities, statistics_dict

            nr_rows_with_replaced_entries = (np.count_nonzero(local_feature_densities[is_valid_row] < threshold, axis=1) > 0).sum()

            nr_replaced_entries = (local_feature_densities[is_valid_row] < threshold).sum()
            nr_valid_entries = local_feature_densities[is_valid_row].size

            statistics_dict.update({
                "replaced_entry_ratio": nr_replaced_entries/nr_valid_entries,
                "nr_replaced_entries": nr_replaced_entries,
                "replaced_row_ratio": nr_rows_with_replaced_entries/nr_valid_rows,
                "nr_replaced_rows": nr_rows_with_replaced_entries
            })

            if print_info:
                print(f"{orbit}: Ratio of replaced entries: {100 * statistics_dict['replaced_entry_ratio']:.1f} %" if nr_valid_entries > 0 else "np.inf", flush=True)
                print(f"{orbit}: Total valid instances: {nr_valid_rows}", flush=True)
                print(f"{orbit}: Ratio of replaced instances: {100 * statistics_dict['replaced_row_ratio']:.1f} %" if nr_valid_rows > 0 else "np.inf", flush=True)

                # iterate over all feature columns
                print(f"{orbit}: Ratio of replaced entries per input category:", flush=True)
                for i in range(local_feature_densities.shape[1]):
                    local_feature_density_i = local_feature_densities[is_valid_row][:, i]
                    print(f"{orbit}:     {_model.X_names()[i]}: {100 * (local_feature_density_i < threshold).sum() / (local_feature_density_i.size):.1f} %", flush=True)

        return X, local_feature_densities, statistics_dict

    X, local_feature_densities, stat_dict = winsorize(model, X, kde_dict, method=conf["sample_method"], threshold=conf["winsor_threshold"], print_info=conf["print_winsorization_info"])

    z_space = np.array(conf["z_space"])
    batchsize = conf["batchsize"]

    y = model(X, z=z_space, X_names=input_X_names, batchsize=batchsize, use_bias_correction=conf["use_bias_correction"], print_prefix=f"{orbit}: ")

    if conf["reject_partial_nans"]:
        invalid_profiles = np.isnan(y).any(axis=1)
        y[invalid_profiles] *= np.nan

    if F_model is not None:
        # if (X-names of F_model) is subset of (X-names of model), we already know the feature densities;
        if set(F_model.X_names()).issubset(model.X_names()):
            F_indices = [model.X_names().index(name) for name in F_model.X_names() if "alt_highres" not in name]
            X_F, local_feature_densities_F = X[:, F_indices], X[:, F_indices]
        else:
            X_F = table_to_model_input(F_model, tabular_ds).to_numpy()
            X_F, local_feature_densities_F, _ = winsorize(F_model, X_F, kde_dict, method=conf["sample_method"], threshold=conf["winsor_threshold"])

        print(f"{orbit}: Computing F-values: ", end="", flush=True)
        F_values = F_model(X_F, z=z_space, batchsize=batchsize, return_z=False)
    else:
        F_values = None

    qa = tabular_ds.qa.to_numpy() * np.sign(tabular_ds.NO2_tropcol_sat.to_numpy())  # improved qa flag (with sign)
    lat = tabular_ds.latitude.to_numpy()
    lon = tabular_ds.longitude.to_numpy()

    def y_to_da(y_arr, name, units="", long_name="", comment=""):
        # Turn a numpy array into a suitable xarray Dataset for merging

        if y_arr is None:
            return xr.ones_like(ds.NO2_tropcol_sat) * np.nan

        # for arrays defined in 3 dimensions, e.g. concentration
        if y_arr.size == ds.scanline.size * ds.ground_pixel.size * z_space.size:  # e.g concentration
            y_da = xr.DataArray(y_arr.reshape((ds.scanline.size, ds.ground_pixel.size, len(z_space))),
                                coords=(
                                {"scanline": ds.scanline, "ground_pixel": ds.ground_pixel, "altitude": z_space}))

        # for arrays defined in two dimensions, e.g. qa, lat, lon
        elif y_arr.size == ds.scanline.size * ds.ground_pixel.size:  # e.g. qa, lat, lon
            y_da = xr.DataArray(y_arr.reshape((ds.scanline.size, ds.ground_pixel.size)),
                                coords=({"scanline": ds.scanline, "ground_pixel": ds.ground_pixel}))
        y_da.name = name
        y_da.attrs = {
            "units": units,
            "long_name": long_name,
            "comment": comment
        }

        return y_da.to_dataset()

    y = y_to_da(y, "no2_conc", units="molec cm-3",
                long_name="Nitrogen dioxide concentration",
                comment="Nitrogen dioxide concentration",
                )

    F = y_to_da(F_values, "F_value", units="",
                long_name="Correction factor (F-value)",
                comment="Correction factor (F-value) for molybdenum-based in-situ measurements (see Lamsal et al., 2008)",
                )

    qa = y_to_da(qa, "qa", units="", long_name="quality flag of the TROPOMI VCD", comment="Negative values mean, that the corresponding NO2 VCD was negative.")
    lat = y_to_da(lat, "lat", units="", long_name="Latitude")
    lon = y_to_da(lon, "lon", units="", long_name="Longitude")

    y = xr.merge([y, qa, lat, lon, ds.time_utc, F])

    # treat uncertainty propagation
    if uncerts is not None:
        # uncertainties are used, each entry along the depth axis (2) is a single uncertainty sample
        X_uncert_lower = deepcopy(X)
        X_uncert_upper = deepcopy(X)

        for key, arr in uncerts.items():
            assert (arr.shape[0] == X.shape[0] and arr.ndim == 1)

            # the last dimension of X (depth) holds the uncertainties of feature X_i
            feature_idx = model.X_names().index(key)
            X_uncert_lower[:, feature_idx] -= arr
            X_uncert_upper[:, feature_idx] += arr

        y_uncert_lower = model(X_uncert_lower, z=z_space, batchsize=batchsize, X_names=input_X_names, print_prefix=f"{orbit}: Uncertainty estimation (lower): ")
        y_uncert_upper = model(X_uncert_upper, z=z_space, batchsize=batchsize, X_names=input_X_names, print_prefix=f"{orbit}: Uncertainty estimation (upper): ")

        y_uncert_lower = y_to_da(y_uncert_lower, "no2_conc_err_lower", "molec cm-3",
                                 long_name="Lower error boundary of the nitrogen dioxide concentration",
                                 comment="Corresponds to the 68% confidence band")
        y_uncert_upper = y_to_da(y_uncert_upper, "no2_conc_err_upper", "molec cm-3",
                                 long_name="Upper error boundary of the nitrogen dioxide concentratio",
                                 comment="Corresponds to the 68% confidence band")

        ds_final = xr.merge([y, y_uncert_lower, y_uncert_upper])

    else:
        ds_final = y

    # transfer the NN prediction uncertainties if they exist
    if (nn_error_interpolator := model.nn_error_interpolator()) is not None:
        y_uncert_nn_rel = nn_error_interpolator(z_space)
        y_uncert_nn_rel = xr.DataArray(y_uncert_nn_rel, coords=({"altitude": z_space}))
        y_uncert_nn_rel.name = "no2_conc_NN_err"
        y_uncert_nn_rel.attrs = {
            "units": "%",
            "long_name": "Relative error of the neural network prediction",
            "comment": ""
        }

        ds_final = xr.merge([ds_final, y_uncert_nn_rel])

    else:
        print("Warning: The chosen model has no error profile. Model output will have limited uncertainty information.", flush=True)

    ds_final.attrs = {"sample_method": conf["sample_method"]}
    if stat_dict is not None:
        for key in stat_dict:
            ds_final[key] = stat_dict[key]

    # if an output_dir is specified, save the resulting dataset there
    if output_dir is not None:
        dt = datetime.date(datetime.strptime("/".join(filename.split("/")[-4:-1]), "%Y/%m/%d"))
        year, month, day = str(dt.year).zfill(2), str(dt.month).zfill(2), str(dt.day).zfill(2)
        savedir = os.path.join(output_dir, year, month, day)
        os.makedirs(savedir, exist_ok=True)
        savename = os.path.join(savedir, f"{orbit}.nc")
        print(f"{orbit}: Writing to output file: {savename}", flush=True)
        ds_final.to_netcdf(savename)

    return ds_final

if __name__ == "__main__":

    # all input files to process
    date_range = [conf["start_date"] + timedelta(days=d) for d in range((conf["end_date"] - conf["start_date"]).days + 1)]
    all_filenames = sorted([fn for fn in glob(conf["preproc"]["output_dir"] + "*/*/*/*.nc") if datetime.date(datetime.strptime("/".join(fn.split("/")[-4:-1]), "%Y/%m/%d")) in date_range])

    # distribute across nodes
    n_proc = int(conf["slurm"]["--cpus-per-task"])
    n_jobs = conf["slurm"]["--array"]

    if n_proc > 1:
        print("n_proc cannot be larger than 1 (multiprocessing not supported). NitroNet allows n_proc > 1 to allow corresponding feature specifications in slurm, but will use n_proc = 1 from hereon")
        n_proc = 1
        conf["slurm"]["--cpus-per-task"] = 1

    coretype = "GPU" if cuda else "core"

    print(f"""
    Efficiency check:
    {n_jobs} jobs
    1 {coretype} per job 
    {1 * n_jobs} total {coretype}s
    {len(all_filenames)} orbit files requested for processing
    """)

    # assign input files to cores
    files_per_core = len(all_filenames) / (n_proc * n_jobs)
    filename_to_processor_idx = {}
    for e, fn in enumerate(all_filenames):
        filename_to_processor_idx.update({fn: e % (n_proc * n_jobs)})

    this_jobs_procs = list(range((slurm_array_task_id - 1) * n_proc, (slurm_array_task_id) * n_proc))
    this_jobs_filenames = [fn for (fn, proc_id) in filename_to_processor_idx.items() if proc_id in this_jobs_procs]

    # instantiate process pool
    if n_proc == 1:
        print("Running in single-process mode")
        args = [[model, fn, F_model, conf["output_dir"]] for fn in this_jobs_filenames]
        for arg in args:
            process_single_input_file(*arg)

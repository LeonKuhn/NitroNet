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

import tomli
import numpy as np
import xarray as xr
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
from scipy.interpolate import griddata

from codebase.find_nearest_neighbours import get_influx
from codebase.plot_functions import geoplot, geomesh, add_cbar, _PROJECTION
from codebase.regions import generic_region

# parse args
conf_file = sys.argv[1]
slurm_array_task_id = int(sys.argv[2])

# load config
with open(conf_file, mode="rb") as fp:
    c = tomli.load(fp)
    conf = {**c["general"], **c["preproc"]}
    del c

# code
def get_all_orbit_filepaths(date_range):

    orbit_nrs = []
    result = []
    result_filtered = []
    for day in date_range:
        NO2_orbit_dir_day = os.path.join(conf["NO2_tropomi_dir"], str(day.year), str(day.month).zfill(2), str(day.day).zfill(2))
        NO2_orbit_filenames = [os.path.join(NO2_orbit_dir_day, x) for x in os.listdir(NO2_orbit_dir_day)]
        result += NO2_orbit_filenames

    for res in result:
        # Remove duplicate orbits...
        if res.split("/")[-1][52:57] not in orbit_nrs:
            result_filtered.append(res)
            orbit_nrs.append(res.split("/")[-1][52:57])

    return sorted(result_filtered)

def process_orbit(NO2_filepath, O3_dir, ERA5_dir, EDGAR_dir, lat, lon, output_dir, debug_plot_dir):
    """
    Take an NO2 orbit file in netCDF format and merges it with matching O3 data, ERA5 data, and EDGARv5 data
    """

    print(f"Processing {NO2_filepath}", flush=True)

    orbit = NO2_filepath.split("/")[-1][52:57]
    dt = datetime.strptime(NO2_filepath.split("/")[-1][20:28], "%Y%m%d")
    unit_conversion_factor = 6.02214e+19 / 1E16  # to convert to 1E16 molec. cm^-2

    def load_tropomi_no2_data():

        no2_orbit_variables = [
            # variable name in orbit file                        netCDF group in which the variable is stored
            ("cloud_radiance_fraction_nitrogendioxide_window",  "PRODUCT/SUPPORT_DATA/DETAILED_RESULTS"),
            ("cloud_fraction_crb_nitrogendioxide_window",       "PRODUCT/SUPPORT_DATA/DETAILED_RESULTS"),
            ("cloud_albedo_crb",                                "PRODUCT/SUPPORT_DATA/INPUT_DATA"),
            ("cloud_pressure_crb",                              "PRODUCT/SUPPORT_DATA/INPUT_DATA"),
            ("surface_albedo_nitrogendioxide_window",           "PRODUCT/SUPPORT_DATA/INPUT_DATA"),
            ("eastward_wind",                                   "PRODUCT/SUPPORT_DATA/INPUT_DATA"),
            ("northward_wind",                                  "PRODUCT/SUPPORT_DATA/INPUT_DATA"),
            ("surface_pressure",                                "PRODUCT/SUPPORT_DATA/INPUT_DATA"),
            ("aerosol_index_354_388",                           "PRODUCT/SUPPORT_DATA/INPUT_DATA"),
            ("surface_classification",                          "PRODUCT/SUPPORT_DATA/INPUT_DATA"),
            ("solar_zenith_angle",                              "PRODUCT/SUPPORT_DATA/GEOLOCATIONS"),
            ("solar_azimuth_angle",                             "PRODUCT/SUPPORT_DATA/GEOLOCATIONS"),
            ("viewing_zenith_angle",                            "PRODUCT/SUPPORT_DATA/GEOLOCATIONS"),
            ("viewing_azimuth_angle",                           "PRODUCT/SUPPORT_DATA/GEOLOCATIONS")
        ]

        # load NO2 data
        no2_ds = xr.open_dataset(NO2_filepath, group="PRODUCT")
        for (var_name, group) in no2_orbit_variables:
            no2_ds[var_name] = xr.open_dataset(NO2_filepath, group=group, cache=True)[var_name]

        # a few variables are not directly available in the dataset and must be added manually
        no2_ds["nitrogendioxide_tropospheric_column"] *= unit_conversion_factor
        no2_ds["trop_AK"] = no2_ds["averaging_kernel"] * (no2_ds["air_mass_factor_total"] / no2_ds["air_mass_factor_troposphere"])
        no2_ds["trop_AK"] = no2_ds["trop_AK"].where(no2_ds.layer < no2_ds["tm5_tropopause_layer_index"], other=0)
        no2_ds["date"] = (("scanline", "ground_pixel"), np.tile(dt, (no2_ds["scanline"].size, no2_ds["ground_pixel"].size)))
        no2_ds["day"] = (("scanline", "ground_pixel"), np.tile(0 if dt.weekday() in [0, 1, 2, 3, 4] else 1, (no2_ds["scanline"].size, no2_ds["ground_pixel"].size)))
        time_utc_squeezed = np.vectorize(lambda x: datetime.strptime(x.split(".")[0], "%Y-%m-%dT%H:%M:%S"))(no2_ds["time_utc"])
        no2_ds["time_utc"] = (("scanline", "ground_pixel"), np.tile(time_utc_squeezed, (no2_ds["ground_pixel"].size, 1)).T)

        print(f"{orbit}: Finished loading NO2 data", flush=True)
        return no2_ds

    def merge_with_tropomi_o3_data(no2_ds):

        # find matching O3 orbit
        o3_subdir = os.path.join(O3_dir, str(dt.year), str(dt.month).zfill(2), str(dt.day).zfill(2))
        o3_file = [x for x in os.listdir(o3_subdir) if x[52:57] == orbit]
        if len(o3_file) != 1:
            print(f"{orbit}: Critical warning: Found {len(o3_file)} possible O3 files, expected exactly 1. This orbit will be skipped")
            return None

        # load data
        o3_ds = xr.open_dataset(os.path.join(o3_subdir, o3_file[0]), group="PRODUCT")
        no2_ds["ozone_total_vertical_column"] = o3_ds["ozone_total_vertical_column"] * unit_conversion_factor

        print(f"{orbit}: Finished merging O3 data", flush=True)
        return no2_ds

    def merge_with_edgar_v5(no2_ds):

        """
        Define emission classes. These must be identical to those used when producing NitroNet's training data.
        The user should not change these emission classes, unless there is a good reason for it.
        """
        vertical_emission_classes = {
            "emi_surface": ["ags", "awb", "mnm", "neu", "rco", "tnr_aviation_cds", "tnr_aviation_crs",
                            "tnr_aviation_lto", "tnr_other", "tnr_ship", "tro_nores", "wwt"],
            "emi_energy_1": ["ene", "ref_trf"],
            "emi_industry_3": ["ind"],
            "emi_industry_4": ["che", "foo_pap", "iro", "nfe", "nmm", "pru_sol"],
        }

        emission_variables = {}  # dict with the new 2D variables to be extracted

        def load_emission_file(filepath):
            """
            read EDGARv5 emission files corresponding to a single sector and month
            """

            ds = xr.open_dataset(filepath, decode_times=False).isel(time=0)
            if "nv" in ds.dims:
                # For some files there exists a dimension "nv", whose meaning we do not know.
                # We check whether the emission inventory is identical across all coordinates of "nv" and then
                # select it by the first index.
                assert (ds.isel(nv=0)["nox_no2"].equals(ds.isel(nv=1)["nox_no2"]))

            # Important bugfix! EDGAR uses a silly [0, 380°] interval for the longitude...
            ds = ds.assign_coords({"lon": (ds.lon + 180) % 360 - 180})

            ds = ds.where(
                (lat[0] <= ds.lat) &
                (lat[1] >= ds.lat) &
                (lon[0] <= ds.lon) &
                (lon[1] >= ds.lon),
                drop=True
            )

            return ds.isel(nv=0)

        def interpolate_to_tropomi_grid(ds):
            edgar_lat, edgar_lon = np.meshgrid(ds.lat.data, ds.lon.data)
            edgar_lat, edgar_lon = edgar_lat.T.flatten(), edgar_lon.T.flatten()
            TROPOMI_lat, TROPOMI_lon = no2_ds.latitude, no2_ds.longitude

            return griddata((edgar_lat, edgar_lon), ds.nox_no2.data.flatten(), (TROPOMI_lat, TROPOMI_lon))

        # compute emission variable "emi_all"
        all_sec = sum(list(vertical_emission_classes.values()), [])
        all_sec = map(lambda x: os.path.join(EDGAR_dir, f"nox_no2_{x}", f"nox_no2_2015{dt.strftime('%m')}.nc"), all_sec)
        all_sec = filter(lambda x: os.path.isfile(x), all_sec)
        all_emi = sum([load_emission_file(x) for x in all_sec])
        all_emi = interpolate_to_tropomi_grid(all_emi)

        print(f"{orbit}: Finished computing emission variable 'emi_all'", flush=True)
        emission_variables["emi_all"] = all_emi

        # compute the emission variablse corresponding to the vertical emission classes defined above
        for key, sec in vertical_emission_classes.items():

            sec = map(lambda x: os.path.join(EDGAR_dir, f"nox_no2_{x}", f"nox_no2_2015{dt.strftime('%m')}.nc"), sec)
            sec = filter(lambda x: os.path.isfile(x), sec)
            emi = sum([load_emission_file(x) for x in sec])
            emi = interpolate_to_tropomi_grid(emi)

            print(f"{orbit}: Finished computing emission variable '{key}'", flush=True)
            emission_variables[key] = emi

        # place all the newly computed emission variables in the joint dataset
        for emi_variable_name, emi_variable in emission_variables.items():
            no2_ds[emi_variable_name] = (("scanline", "ground_pixel"), emi_variable)

        return no2_ds

    def merge_with_era5(no2_ds, debug_plot_dir):

        # specify which variables to load from the ERA5 data
        era5_vars = ["blh", "bld", "t2m", "w", "u", "v"]

        # load ERA5 data for the day of the orbit and the following day (for seamless temporal interpolation)
        era5_ds = xr.open_dataset(os.path.join(ERA5_dir, f"ERA5_{dt.year}-{str(dt.month).zfill(2)}-{str(dt.day).zfill(2)}.nc"))
        next_dt = dt + timedelta(days=1)
        era5_ds_nextday = xr.open_dataset(os.path.join(ERA5_dir, f"ERA5_{next_dt.year}-{str(next_dt.month).zfill(2)}-{str(next_dt.day).zfill(2)}.nc"))

        min_hour = int(np.nanmin(pd.DatetimeIndex(no2_ds["time_utc"].data.flatten()).hour))
        max_hour = int(np.nanmax(pd.DatetimeIndex(no2_ds["time_utc"].data.flatten()).hour))

        # interpolate the ERA5 data spatially to the TROPOMI grid and assign correct coordinates
        def interpolate_spatially_to_tropomi_grid(era5_ds, hour):
            """
            Interpolate ERA5 dataset spatially at a fixed hour
            """

            # lat-lon of satellite grid
            TROPOMI_lat = np.array(no2_ds["latitude"]).flatten()
            TROPOMI_lon = np.array(no2_ds["longitude"]).flatten()

            # lat-lon of ERA5 grid
            lon_mesh, lat_mesh = np.meshgrid(era5_ds.longitude, era5_ds.latitude)

            ERA5_interp_ds = xr.Dataset()

            for var in era5_vars:
                print(f"{orbit}: Interpolating ERA5 variable '{var}' at hour {hour}", flush=True)

                if hour < 24:
                    da = era5_ds.where(era5_ds.valid_time.dt.hour == hour, drop=True).isel(valid_time=0)[var]
                else:
                    da = era5_ds_nextday.where(era5_ds_nextday.valid_time.dt.hour == (hour % 24), drop=True).isel(valid_time=0)[var]

                if "pressure_level" not in da.dims:
                    var_interp = griddata(
                        (lat_mesh.flatten(), lon_mesh.flatten()),
                        da.data.flatten(),
                        (TROPOMI_lat, TROPOMI_lon),
                        method="linear"
                    ).reshape(no2_ds["latitude"].shape)

                    ERA5_interp_ds[var] = (("scanline", "ground_pixel"), var_interp)

                else:
                    var_interp = np.stack([griddata(
                        (lat_mesh.flatten(), lon_mesh.flatten()),
                        da.sel(pressure_level=pressure_level).data.flatten(),
                        (TROPOMI_lat, TROPOMI_lon),
                        method="linear"
                    ).reshape(no2_ds["latitude"].shape) for pressure_level in da.pressure_level])

                    ERA5_interp_ds[var] = (("pressure_level", "scanline", "ground_pixel"), var_interp)

            return ERA5_interp_ds

        hours = range(min_hour, max_hour + 2)
        single_hour_datasets = [interpolate_spatially_to_tropomi_grid(era5_ds, h) for h in hours]
        era5_ds_interp_spatially = xr.concat(single_hour_datasets, dim="valid_time")
        era5_ds_interp_spatially = era5_ds_interp_spatially.assign_coords({"valid_time": [datetime(dt.year, dt.month, dt.day, h%24) for h in hours]})
        era5_ds_interp_spatially = era5_ds_interp_spatially.assign_coords({"pressure_level": era5_ds.pressure_level})

        # interpolate the ERA5 data temporally to the overpass time
        def interpolate_temporally(era5_interp_spatially):
            """
            Interpolate ERA5 dataset temporally (to be used *after* spatial interpolation)
            """

            # start with a None object
            ERA5_interp_temporally = None

            for h in hours:
                # determine, which lat-lon of the satellite dataset belong to this hour (rounding down)
                T_mask = (no2_ds["time_utc"].dt.hour == h)

                # Compute interpolation weight between hour h and h+1
                weights = 1 - no2_ds["time_utc"].dt.minute / 60

                # Add the entries of the temporally interpolated ds at hour h, replaxing NaN with 0's
                ds_at_t = era5_interp_spatially.where(era5_interp_spatially.valid_time.dt.hour == h, drop=True).mean("valid_time")
                ds_at_t_next = era5_interp_spatially.where(era5_interp_spatially.valid_time.dt.hour == h + 1, drop=True).mean("valid_time")
                weighted_arr = T_mask * (weights * ds_at_t + (1 - weights) * ds_at_t_next).fillna(0)
                if ERA5_interp_temporally is None:
                    ERA5_interp_temporally = weighted_arr
                else:
                    ERA5_interp_temporally += weighted_arr

            # Set all 0's back to NaNs
            ERA5_interp_temporally = ERA5_interp_temporally.where(ERA5_interp_temporally.blh > 0)

            return ERA5_interp_temporally

        era5_ds_interp_final = interpolate_temporally(era5_ds_interp_spatially)

        for variable_name in era5_vars:
            no2_ds[variable_name] = era5_ds_interp_final[variable_name]
        no2_ds["ERA5_total_wind"] = np.sqrt(no2_ds["u"] ** 2 + no2_ds["v"] ** 2)
        no2_ds = no2_ds.assign_coords({"pressure_level": era5_ds.pressure_level})

        # optionally create debug plots
        def create_era5_diagnostic_plots(era5_ds, era5_ds_interp_final, plot_hours, debug_plot_dir):

            os.makedirs(debug_plot_dir, exist_ok=True)

            def plot_single_era5_variable(variable_name, plot_hours, level=None):

                print(f"{orbit}: Plotting '{variable_name}'", end="", flush=True)
                if level is not None:
                    print(f" at pressure level {level.item()} hPa", flush=True)
                else:
                    print("", flush=True)  # should be kept for correct formatting

                # figure setup
                fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, subplot_kw={'projection': _PROJECTION}, figsize=(8.0, 3.0))

                title_string = f"{variable_name}"
                if level is not None:
                    title_string += f" at pressure level {level.item()} hPa"
                plt.suptitle(title_string)

                region = generic_region(lat=era5_ds.latitude, lon=era5_ds.longitude)
                ax1 = geoplot(ax1, region, draw_ticks="tl", aspect="auto"); ax1.set_title("original")
                ax2 = geoplot(ax2, region, draw_ticks="tl", aspect="auto"); ax2.set_title("spatially interp.")
                ax3 = geoplot(ax3, region, draw_ticks="tl", aspect="auto"); ax3.set_title("spatial + temporal interp.")

                if level is None:
                    era5_orig_arr = era5_ds.where(era5_ds.valid_time.dt.hour.isin(plot_hours))[variable_name].mean("valid_time").data
                    era5_spatial_arr = era5_ds_interp_spatially.where(era5_ds_interp_spatially.valid_time.dt.hour.isin(plot_hours))[variable_name].mean("valid_time").data
                    era5_final_arr = era5_ds_interp_final[variable_name].data

                else:
                    era5_orig_arr = era5_ds.where(era5_ds.valid_time.dt.hour.isin(plot_hours))[variable_name].mean("valid_time").sel(pressure_level=level).data
                    era5_spatial_arr = era5_ds_interp_spatially.where(era5_ds_interp_spatially.valid_time.dt.hour.isin(plot_hours))[variable_name].mean("valid_time").sel(pressure_level=level).data
                    era5_final_arr = era5_ds_interp_final[variable_name].sel(pressure_level=level).data

                p1 = geomesh(ax1, *np.meshgrid(era5_ds.longitude, era5_ds.latitude), era5_orig_arr)
                p2 = geomesh(ax2, np.array(no2_ds["longitude"]), np.array(no2_ds["latitude"]), era5_spatial_arr)
                p3 = geomesh(ax3, era5_ds_interp_final.longitude, era5_ds_interp_final.latitude, era5_final_arr)

                # add color bars
                add_cbar(ax1, p1, "bottom")
                add_cbar(ax2, p2, "bottom")
                add_cbar(ax3, p3, "bottom")

                # save figure
                plt.tight_layout()
                savename = f"{orbit}_ERA5_{variable_name}"
                if level is not None:
                    savename += f"_{int(level.item())}-hPa"
                savename += ".png"
                plt.savefig(os.path.join(debug_plot_dir, savename))
                #plt.show()
                plt.close()

            for variable_name in era5_vars:
                if "pressure_level" not in no2_ds[variable_name].dims:
                    plot_single_era5_variable(variable_name, plot_hours, level=None)
                else:
                    for level in no2_ds.pressure_level:
                        plot_single_era5_variable(variable_name, plot_hours, level=level)

        if type(debug_plot_dir) is str:
            create_era5_diagnostic_plots(era5_ds, era5_ds_interp_final, hours, debug_plot_dir)

        print(f"{orbit}: Finished loading ERA5 data", flush=True)
        return no2_ds

    def add_influx(NO2_ds):

        influx_array = get_influx(
            NO2_ds["nitrogendioxide_tropospheric_column"].data,
            NO2_ds["eastward_wind"].data,
            NO2_ds["northward_wind"].data
        )

        influx = xr.DataArray(
            data=influx_array,
            dims=["scanline", "ground_pixel"],
            coords=dict(
                scanline=NO2_ds.scanline,
                ground_pixel=NO2_ds.ground_pixel,
            ),
            attrs=dict(
                units="10^16 molecules cm-2 m s-1",
                full_name="Influx of NO2 from neighbouring cells",
            ),
        )

        NO2_ds["influx"] = influx

        print(f"{orbit}: Finished computing NO2 influx", flush=True)
        return NO2_ds

    def add_surface_classes(NO2_ds):

        surface_classes = [
            {"name": "urban", "bits": [8], "hex_mask": 0xF9},
            {"name": "cropland", "bits": [16, 24, 32, 40, 48], "hex_mask": 0xF9},
            {"name": "forest", "bits": [88, 96, 104, 112, 120, 128, 136], "hex_mask": 0xF9},
        ]

        def get_surface_class(name):
            """
            Take a surface class name from the surface_classes dict and return a one-hot encoded DataArray
            """

            s_class = [s_class for s_class in surface_classes if s_class["name"] == name][0]

            hex_mask = s_class["hex_mask"]
            bits = s_class["bits"]

            def is_nth_bit_set(x, hex_mask, bit):
                """
                Check whether the n-th bit is set in x after applying a hex-mask
                """
                hex_masked_array = x & hex_mask
                return hex_masked_array == bit

            partial_masks = []
            surf_class_as_int = NO2_ds["surface_classification"].fillna(-1).astype(int)
            for bit in bits:
                mask = xr.apply_ufunc(is_nth_bit_set, surf_class_as_int, hex_mask, bit, vectorize=True)
                partial_masks.append(mask)

            for m in partial_masks:
                mask = mask | m

            mask_np = np.array(mask)

            mask_da = xr.DataArray(
                data=mask_np,
                dims=["scanline", "ground_pixel"],
                coords=dict(
                    glat=NO2_ds.scanline,
                    glon=NO2_ds.ground_pixel,
                ),
            )

            return mask_da

        NO2_ds["SC_" + "urban"] = get_surface_class("urban")
        NO2_ds["SC_" + "urban"].attrs = {"units": "", "full_name": f"Surface class 'urban'"}
        NO2_ds["SC_" + "cropland"] = get_surface_class("cropland")
        NO2_ds["SC_" + "cropland"].attrs = {"units": "", "full_name": f"Surface class 'cropland'"}
        NO2_ds["SC_" + "forest"] = get_surface_class("forest")
        NO2_ds["SC_" + "forest"].attrs = {"units": "", "full_name": f"Surface class 'forest'"}

        print(f"{orbit}: Finished computing surface classes", flush=True)

        return NO2_ds

    no2_ds = load_tropomi_no2_data()  # load TROPOMI NO2 data
    no2_ds = merge_with_tropomi_o3_data(no2_ds)  # load TROPOMI O3 data

    # check if orbit data is within the specified lat-lon boundaries
    if no2_ds is None:
        # e.g. if no O3 data was found
        return 0

    no2_ds = no2_ds.isel(time=0)
    lat_lon_mask = (
        (no2_ds.latitude > lat[0]) & (no2_ds.latitude < lat[1]) &
        (no2_ds.longitude > lon[0]) & (no2_ds.longitude < lon[1])
    )
    if not lat_lon_mask.any():
        print(f"{orbit}: skipped because orbit did not intersect with prediction region", flush=True)
        return 0

    no2_ds = no2_ds.where(lat_lon_mask, drop=True)

    no2_ds = merge_with_edgar_v5(no2_ds)  # merge with EDGARv5 data
    no2_ds = merge_with_era5(no2_ds, debug_plot_dir)  # merge with ERA5 data
    no2_ds = add_influx(no2_ds)
    no2_ds = add_surface_classes(no2_ds)

    # rename arrays in the dataset so it matches the naming conventions used when creating training data
    no2_ds = no2_ds.rename({
        "cloud_radiance_fraction_nitrogendioxide_window": "crf",
        "cloud_fraction_crb_nitrogendioxide_window": "cf",
        "cloud_albedo_crb": "cloud_albedo",
        "cloud_pressure_crb": "cloud_pressure",
        "surface_albedo_nitrogendioxide_window": "surface_albedo",
        "eastward_wind": "wind_u",
        "northward_wind": "wind_v",
        "aerosol_index_354_388": "aerosol_index",
        "solar_zenith_angle": "sza",
        "solar_azimuth_angle": "saa",
        "viewing_zenith_angle": "za_sat",
        "viewing_azimuth_angle": "aa_sat",
        "bld": "bld_ERA5",
        "blh": "blh_ERA5",
        "t2m": "T2m_ERA5",
        "qa_value": "qa",
        "nitrogendioxide_tropospheric_column": "NO2_tropcol_sat",
        "ozone_total_vertical_column": "O3_totalcol_sat",
        "w": "vertical_speed_ERA5",
        "air_mass_factor_troposphere": "trop_AMF"
    })

    # final sanity check for all processed variables
    def plot_variable(var_name):

        print(f"{orbit}: Plotting '{var_name}'", end=" - ", flush=True)
        var = no2_ds[var_name]

        region = generic_region(lat=no2_ds.latitude, lon=no2_ds.longitude)

        # not all variables are suitable for plotting - in this case, skip them
        if ("scanline" not in var.dims) or ("ground_pixel" not in var.dims):
            not_plotted_reason = "does not have 'scanline' and 'ground_pixel' as dimensions"
        elif not np.issubdtype(var.dtype, np.number) and var.dtype != bool:
            not_plotted_reason = "does not have 'scanline' and 'ground_pixel' as dimensions"
        elif "vertices" in var.dims:
            not_plotted_reason = "contains 'vertices' as a coordinate"
        else:
            not_plotted_reason = None

        if not_plotted_reason is not None:
            print("not suitable for plotting, because:", not_plotted_reason, flush=True)
            return

        if "pressure_level" in var.dims:
            # variables defined on the ERA5 pressure grid are plotted on a facet grid
            fig, axarr = plt.subplots(
                int(no2_ds.pressure_level.size), 1,
                subplot_kw={'projection': _PROJECTION},
                figsize=(8, 6 * int(no2_ds.pressure_level.size))
            )

            var = no2_ds[var_name]

            for e in range(no2_ds.pressure_level.size):
                ax = axarr[e]
                ax.set_title(f"{var_name} at pressure level {no2_ds.pressure_level[e].item()} hPa")
                ax = geoplot(ax, region, draw_ticks="TL", aspect="auto")
                p = geomesh(ax, no2_ds.longitude.data, no2_ds.latitude.data, var.isel(pressure_level=e).data)

            savename_prefix = "final"

        else:
            var = no2_ds[var_name]
            if "layer" in var.dims:
                # e.g. for the averaging kernels; here just the lowest layer is plotted
                var = var.isel(layer=0)

            fig, ax1 = plt.subplots(1, 1, subplot_kw={'projection': _PROJECTION})
            plt.title(f"{var_name}")

            ax1 = geoplot(ax1, region, draw_ticks="TL", aspect="auto")
            p = geomesh(ax1, no2_ds.longitude.data, no2_ds.latitude.data, var.data)

            savename_prefix = "final"


        add_cbar(plt.gca(), p, "bottom")
        print("ok", flush=True)  # keep for good logging

        # save figure
        plt.tight_layout()
        plt.savefig(os.path.join(debug_plot_dir, f"{orbit}_{savename_prefix}_{var_name}.png"))
        # plt.show()
        plt.close()

    if type(debug_plot_dir) is str:
        for var in no2_ds.variables:
            plot_variable(var)

    # if outout_dir is given, save the resulting dataset there
    if output_dir is not None:
        output_dir = os.path.join(output_dir, f"{dt.strftime('%Y')}", f"{dt.strftime('%m')}", f"{dt.strftime('%d')}")
        os.makedirs(output_dir, exist_ok=True)
        full_savename = os.path.join(output_dir, f"nn_input_{orbit}.nc")
        print(f"Saving to file: {full_savename}")
        no2_ds.to_netcdf(full_savename)

    return 0

if __name__ == "__main__":

    # obtain all TROPOMI orbit files to process
    date_range = [conf["start_date"] + timedelta(days=d) for d in range((conf["end_date"] - conf["start_date"]).days + 1)]
    all_orbit_filepaths = get_all_orbit_filepaths(date_range)

    n_proc = int(conf["slurm"]["--cpus-per-task"])
    n_jobs = conf["slurm"]["--array"]

    print(f"""
    Efficiency check:
    {n_jobs} jobs
    {n_proc} cores per job 
    {n_proc * n_jobs} total cores
    {len(all_orbit_filepaths)} orbit files requested for processing
    """)

    # assign TROPOMI orbit files to cores
    files_per_core = len(all_orbit_filepaths) / (n_proc * n_jobs)
    filename_to_processor_idx = {}
    for e, fn in enumerate(all_orbit_filepaths):
        filename_to_processor_idx.update({fn: e % (n_proc * n_jobs)})

    this_jobs_procs = list(range((slurm_array_task_id - 1) * n_proc, (slurm_array_task_id) * n_proc))
    this_jobs_filenames = [fn for (fn, proc_id) in filename_to_processor_idx.items() if proc_id in this_jobs_procs]

    # instantiate process pool
    if n_proc == 1:
        print("Running in single-process mode")
        args = [[fp, conf["O3_tropomi_dir"], conf["ERA5_dir"], conf["EDGARv5_dir"], conf["lat_boundaries"], conf["lon_boundaries"], conf["output_dir"], conf["debug_plot_dir"]]
                for fp in this_jobs_filenames]
        for arg in args:
            process_orbit(*arg)
    else:
        print("Running in multi-process mode")
        p = mp.Pool(processes=n_proc)
        args = [[fp, conf["O3_tropomi_dir"], conf["ERA5_dir"], conf["EDGARv5_dir"], conf["lat_boundaries"], conf["lon_boundaries"], conf["output_dir"], conf["debug_plot_dir"]]
                for fp in this_jobs_filenames]

        results = [p.apply_async(process_orbit, args=a) for a in args]
        p.close()
        p.join()

        # error tracing
        for arg, r in zip(args, results):
            try:
                r.get()
            except Exception as e:
                print("Error has occured with filename:", arg[0])
                print(e)

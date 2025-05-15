# Copyright (c) 2025 Leon Kuhn
#
# This file is part of a software project licensed under a Custom Non-Commercial License.
#
# You may use, modify, and distribute this code for non-commercial purposes only.
# Use in academic or scientific research requires prior written permission.
#
# For full license details, see the LICENSE file or contact l.kuhn@mpic.de


import os
import xarray as xr
import subprocess
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import re

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="e.g. ./data/ERA5/test")
    args = parser.parse_args()

    test_plot = False

    for fn in os.listdir(args.input_dir):
        path = Path(args.input_dir) / Path(fn)
        if path.suffix == ".zip":  # because .zip and .nc filenames are redundant
            continue

        stem = path.stem
        date_id = re.findall("\d{4}-\d{2}-\d{2}_to_\d{4}-\d{2}-\d{2}", stem)
        if date_id == []:
            print(f"Skipping file {path}")
            continue
        else:
            date_id = date_id[0]

        split_files(args.input_dir, date_id, test_plot=test_plot)

def split_files(input_dir, filename_id, test_plot=False):

    single_level_filename = f"ERA5_single-levels_{filename_id}.zip"
    pressure_level_filename = f"ERA5_pressure-levels_{filename_id}.nc"

    try:
        os.chdir(input_dir)  # required to unzip in-place, but throws error after first time being called
    except:
        pass

    cmd = ["unzip", "-o", single_level_filename]
    subprocess.check_call(cmd)

    with xr.open_mfdataset([
            "data_stream-oper_stepType-accum.nc",
            "data_stream-oper_stepType-instant.nc",
             pressure_level_filename
        ]) as ds:

        for dt in sorted(set(ds.indexes["valid_time"].date)):
            print(f"Processing {dt.strftime('%Y-%m-%d')}")
            # iterate over the unique days in the combined dataset
            daily_ds = ds.sel(valid_time=xr.date_range(dt.strftime('%Y-%m-%d'), freq="h", periods=24))
            daily_ds.to_netcdf(dt.strftime('ERA5_%Y-%m-%d.nc'))

            if test_plot:
                # debugging routine to ensure correct function
                plt.plot(daily_ds.mean(["latitude", "longitude"]).blh)
                plt.title(dt.strftime('%Y-%m-%d')); plt.xlabel("BLH"); plt.ylabel("hour of the day")
                plt.show()

if __name__ == "__main__":
    main()

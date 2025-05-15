# Copyright (c) 2025 Leon Kuhn
#
# This file is part of a software project licensed under a Custom Non-Commercial License.
#
# You may use, modify, and distribute this code for non-commercial purposes only.
# Use in academic or scientific research requires prior written permission.
#
# For full license details, see the LICENSE file or contact l.kuhn@mpic.de

import os
import cdsapi
from datetime import datetime, timedelta
import argparse

def download_data(all_days, bbox, download_destination):

    # setup download and remote directories
    os.makedirs(download_destination, exist_ok=True)

    # merge pressure and single levels for each day
    single_level_filename = f"ERA5_single-levels_{all_days[0].strftime('%Y-%m-%d')}_to_{all_days[-1].strftime('%Y-%m-%d')}.zip"
    pressure_level_filename = f"ERA5_pressure-levels_{all_days[0].strftime('%Y-%m-%d')}_to_{all_days[-1].strftime('%Y-%m-%d')}.nc"

    c = cdsapi.Client()
    time_array = [f'{str(n).zfill(2)}:00' for n in range(24)]  # ['00:00', '01:00', ... '23:00']

    """
    Note: Fetching data from ERA5 single levels currently returns a .zip file which contains two netcdf4 files that
    must be merged together (see below).
    For information on the resolution ('grid'), see 
    https://confluence.ecmwf.int/display/CKB/How+to+download+ERA5#HowtodownloadERA5-StepB:DownloadERA5datalistedinCDSthroughCDSAPI
    """

    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'data_format': 'netcdf',
            'variable': ['boundary_layer_height', "2m_temperature", "boundary_layer_dissipation"],
            'date': [d.strftime("%Y-%m-%d") for d in all_days],
            'time': time_array,
            'area': bbox,
            'grid': [0.25, 0.25]
        },
        os.path.join(download_destination, single_level_filename))

    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'pressure_level': ['1000', '950', '900', '850', '800', '750', '700'],
            'data_format': 'netcdf',
            'variable': ['vertical_velocity', 'u_component_of_wind', 'v_component_of_wind'],
            'date': [d.strftime("%Y-%m-%d") for d in all_days],
            'time': time_array,
            'area': bbox,
            'grid': [0.25, 0.25]
        },
        os.path.join(download_destination, pressure_level_filename))

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("start_to_end", type=str, help="e.g. 2022-05-21/2022-06-21")
    parser.add_argument("bbox", type=str, help="south-west-north-east, e.g. 45-5-55-15")
    parser.add_argument("savedir", type=str, help="subfolder name to save this download under, e.g. ./data/ERA5/test")
    args = parser.parse_args()

    start, end = args.start_to_end.split("/")
    start = datetime.strptime(start, "%Y-%m-%d")
    end = datetime.strptime(end, "%Y-%m-%d")

    bbox = args.bbox.split("-")
    bbox = [int(b) for b in bbox]

    # here: split all downloads into single days, to allow for large domains
    for timestep in range((end - start).days + 2):
        download_data([start + timedelta(days=timestep)], bbox, args.savedir)

    # otherwise:
    #end = end + timedelta(days=1)  # one extra day required for full model input
    #all_days = [start + timedelta(days=x) for x in range((end - start).days + 1)]
    #download_data(all_days, bbox, args.savedir)

if __name__ == "__main__":
    main()

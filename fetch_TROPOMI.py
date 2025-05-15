# Copyright (c) 2025 Leon Kuhn
#
# This file is part of a software project licensed under a Custom Non-Commercial License.
#
# You may use, modify, and distribute this code for non-commercial purposes only.
# Use in academic or scientific research requires prior written permission.
#
# For full license details, see the LICENSE file or contact l.kuhn@mpic.de

import os
import subprocess
from datetime import datetime
from dateutil import rrule
import argparse
from getpass import getpass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("start_to_end", type=str, help="e.g. 2022-05-21/2022-06-21")
    parser.add_argument("tracegasses", type=str, help="O3_TOT or NO2")
    parser.add_argument("timeliness", type=str, help="RPRO or OFFL")
    parser.add_argument("savedir", type=str, help="subfolder name to save this download under, e.g. ./data/TROPOMI/")

    args = parser.parse_args()

    start, end = args.start_to_end.split("/")
    start = datetime.strptime(start, "%Y-%m-%d")
    end = datetime.strptime(end, "%Y-%m-%d")

    timeliness = args.timeliness.split("/")
    tracegasses = args.tracegasses.split("/")

    user = input("gesdic Username: ")     # username in the gesdisc portal
    pw = getpass("gesdisc Password: ")    # password in the gesdisc portal

    for tl, tg in zip(timeliness, tracegasses):
        download_file(start, end, tl, tg, user, pw, args.savedir)

def download_file(start, end, timeliness, tracegas, user, pw, savedir):

    # setup download and remote directories
    wgetpath = '/usr/bin/wget'
    ruler = rrule.rrule(rrule.DAILY, dtstart=start, until=end)
    remote_base_dir = "https://tropomi.gesdisc.eosdis.nasa.gov/data/S5P_TROPOMI_Level2"
    downloaddir = os.path.join(savedir, "L2")
    os.makedirs(downloaddir, exist_ok=True)

    # start the download
    for dt in ruler:
        downdir = os.path.join(downloaddir, tracegas, timeliness, f'{dt:%Y}', f'{dt:%m}', f'{dt:%d}')

        varstring = '{}'.format(tracegas).ljust(7, '_')
        tracegasfold = f'S5P_L2__{varstring}HiR.2'

        l2_remotedir = os.path.join(remote_base_dir, tracegasfold)
        l2_datedir = os.path.join(l2_remotedir, f'{dt:%Y}', f'{dt:%j}')

        cmd = [wgetpath, '-e robots=off', f'{l2_datedir}', '--no-proxy',
               '-c', '-r', '-l', '1', '-nH', '-nd',
               '-np', '-A', f"*{timeliness}*.nc", '-P', str(downdir),
               '--user', user, '--password', pw]

        subprocess.check_call(cmd)

if __name__ == "__main__":
    main()

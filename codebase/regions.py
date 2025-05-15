# Copyright (c) 2025 Leon Kuhn
#
# This file is part of a software project licensed under a Custom Non-Commercial License.
#
# You may use, modify, and distribute this code for non-commercial purposes only.
# Use in academic or scientific research requires prior written permission.
#
# For full license details, see the LICENSE file or contact l.kuhn@mpic.de

from dataclasses import dataclass
import numpy as np

@dataclass
class Region:
    latS: float
    latN: float
    lonW: float
    lonE: float

    lon_ticks: list
    lat_ticks: list

    lon_minor_ticks: list = None
    lat_minor_ticks: list = None

    def central_longitude(self):
        return 0.5 * (self.lonW + self.lonE)

    def central_latitude(self):
        return 0.5 * (self.latS + self.latN)

test_region = Region(
    latS=45, latN=55, lonW=5, lonE=15,
    lon_ticks=[5, 10, 15], lat_ticks=[45, 50, 55]
)

def generic_region(*, lat, lon, lat_ticks=None, lon_ticks=None, lat_minor_ticks=None, lon_minor_ticks=None):
    """
    creates a generic_region based on lat and lon boundaries
    :param lat: tuple of lower and upper lat boundaries
    :param lon: tuple of lower and upper lon boundaries
    """

    if lat_ticks is None:
        lat_ticks = [np.min(lat), np.max(lat)]
    if lon_ticks is None:
        lon_ticks = [np.min(lon), np.max(lon)]

    return Region(latS=np.min(lat), latN=np.max(lat), lonW=np.min(lon), lonE=np.max(lon),
            lon_ticks=lon_ticks,
            lat_ticks=lat_ticks,
            lon_minor_ticks=lon_minor_ticks,
            lat_minor_ticks=lat_minor_ticks,
            )
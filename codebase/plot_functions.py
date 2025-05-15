# Copyright (c) 2025 Leon Kuhn
#
# This file is part of a software project licensed under a Custom Non-Commercial License.
#
# You may use, modify, and distribute this code for non-commercial purposes only.
# Use in academic or scientific research requires prior written permission.
#
# For full license details, see the LICENSE file or contact l.kuhn@mpic.de

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

_PROJECTION = ccrs.PlateCarree()  # PlateCarree is the standard projection used in this code

def add_cbar(ax, mappable, pos, **kwargs):
    # add a cbar to a geoplot

    if pos.lower() == "right":
        cax = ax.inset_axes((1.03, 0, 0.02, 1))
        cbar = plt.colorbar(mappable=mappable, cax=cax, orientation="vertical", **kwargs)

    elif pos.lower() == "bottom":
        cax = ax.inset_axes((0, -0.08, 1, 0.04))
        cbar = plt.colorbar(mappable=mappable, cax=cax, orientation="horizontal", **kwargs)

    elif pos.lower() == "left":
        cax = ax.inset_axes((-.05, 0, 0.02, 1))
        cbar = plt.colorbar(mappable=mappable, cax=cax, orientation="vertical", **kwargs)
        cbar.ax.yaxis.set_ticks_position("left")
        cbar.ax.yaxis.set_label_position("left")

    elif pos.lower() == "top":
        cax = ax.inset_axes((0, 1.04, 1, 0.04))
        cbar = plt.colorbar(mappable=mappable, cax=cax, orientation="horizontal", **kwargs)
        cbar.ax.xaxis.set_ticks_position("top")
        cbar.ax.xaxis.set_label_position("top")

    else:
        raise Exception("pos argument must be either 'right', 'bottom', 'left', or 'top'")

    return cbar


def geoscatter(ax, *args, **kwargs):
    # wrapper for scatter using some standard settings (for use with geoplot)
    return ax.scatter(*args, transform=ax.projection, zorder=3, **kwargs)


def geomesh(ax, *args, **kwargs):
    # wrapper for pcpolormesh using some standard settings (for use with geoplot)

    return ax.pcolormesh(*args, transform=ax.projection, rasterized=True, **kwargs)


def geoplot(ax, region, draw_ticks=None, aspect=None):
    """
    Creates a geographic plot
    :param ax: axis to draw the plot on
    :param region: generic_region to cover in this plot (see generic_region.py)
    :param draw_ticks: tick position identifier as string. E.g. "Tl" draws ticks top and left with tick labels on top.
    :param aspect: None or 'auto'; 'auto' stretches the subplots to fill the entire figure space
    """

    projection = ax.projection

    # set axis extent (order is: [left, right, top, bottom]):
    ax.set_extent([region.lonW, region.lonE, region.latN, region.latS], crs=projection)

    # plot country lines etc.
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)

    # for rasterization
    ax.set_rasterization_zorder(0)

    # show ticks
    if draw_ticks:
        if "l" in draw_ticks.lower():
            ax.yaxis.tick_left()
            ax.set_yticks(region.lat_ticks, crs=_PROJECTION)
            lat_formatter = LatitudeFormatter(number_format='.0f', degree_symbol='°')
            ax.yaxis.set_major_formatter(lat_formatter)

            # if tick string is not capital, only write the ticks but no tick labels
            if "L" not in draw_ticks:
                ax.set_yticklabels([])

        elif "r" in draw_ticks.lower():
            ax.yaxis.tick_right()
            ax.set_yticks(region.lat_ticks, crs=_PROJECTION)
            lat_formatter = LatitudeFormatter(number_format='.0f', degree_symbol='°')
            ax.yaxis.set_major_formatter(lat_formatter)

            if "R" not in draw_ticks:
                ax.set_yticklabels([])

        if "t" in draw_ticks.lower():
            ax.xaxis.tick_top()
            ax.set_xticks(region.lon_ticks, crs=_PROJECTION)
            lon_formatter = LongitudeFormatter(number_format='.0f', degree_symbol='°', dateline_direction_label=True,
                                               zero_direction_label=True)
            ax.xaxis.set_major_formatter(lon_formatter)

            if "T" not in draw_ticks:
                ax.set_xticklabels([])

        elif "b" in draw_ticks.lower():
            ax.xaxis.tick_bottom()
            ax.set_xticks(region.lon_ticks, crs=_PROJECTION)
            lon_formatter = LongitudeFormatter(
                number_format='.0f', degree_symbol='°',
                dateline_direction_label=True, zero_direction_label=True)
            ax.xaxis.set_major_formatter(lon_formatter)

            if "B" not in draw_ticks:
                ax.set_yticklabels([])

    if aspect is not None:
        ax.set_aspect(aspect)

    # minor ticks are sometimes poorly chosen by cartopy; we can set them manually
    if region.lat_minor_ticks is not None:
        ax.set_yticks(region.lat_minor_ticks, minor=True)
    if region.lon_minor_ticks is not None:
        ax.set_xticks(region.lon_minor_ticks, minor=True)

    return ax

# Copyright (c) 2025 Leon Kuhn
#
# This file is part of a software project licensed under a Custom Non-Commercial License.
#
# You may use, modify, and distribute this code for non-commercial purposes only.
# Use in academic or scientific research requires prior written permission.
#
# For full license details, see the LICENSE file or contact l.kuhn@mpic.de

import numpy as np

def get_moore_neighbourhood(arr):
    """
    Given a 2d numpy array arr, this function returns the values of each pixels moore neighbourhood.
    """

    row_indices = np.arange(arr.shape[0])
    col_indices = np.arange(arr.shape[1])

    j, i = np.meshgrid(col_indices, row_indices)

    i_offsets = np.array([-1, -1, 0, +1, +1, +1, 0, -1])
    j_offsets = np.array([0, +1, +1, +1, 0, -1, -1, -1])

    i_offsets = np.expand_dims(i_offsets, axis=(0, 1))
    j_offsets = np.expand_dims(j_offsets, axis=(0, 1))

    i_nn = np.expand_dims(i, axis=2) + i_offsets
    j_nn = np.expand_dims(j, axis=2) + j_offsets

    # neighbourhood values are "mirrored" at the edges of the array
    i_nn = np.clip(i_nn, 0, arr.shape[0] - 1)
    j_nn = np.clip(j_nn, 0, arr.shape[1] - 1)

    result = arr[i_nn, j_nn]

    return result

def get_influx(VCD_arr, u_arr, v_arr):
    """
    Given a 2d numpy array of VCDs (VCD_arr) and corresponding wind arrays (u_arr, v_arr), this function returns
    the influx of VCDs into each pixel of VCD_arr.
    """

    u = np.array((0, 1))  # Unit vector pointing west
    v = np.array((1, 0))  # Unit vector pointing north

    # Normalized radial vector that points from a cell to its neighbours:
    r = np.array([-v, (-v + u), +u, (+v+u), +v, (+v - u), -u, (-v-u)]).T
    normalization = np.array([1, 1/np.sqrt(2), 1, 1/np.sqrt(2), 1, 1/np.sqrt(2), 1, 1/np.sqrt(2)]).T

    u_nn = get_moore_neighbourhood(u_arr)
    v_nn = get_moore_neighbourhood(v_arr)

    wind = np.stack([v_nn, u_nn], axis=-1)

    """
    In order compute the influx, we need the dot product of the neighbouring wind vectors with the radial
    vectors pointing to the neighbouring cells. Whether a neighbouring cell's flux contributes to a center cell's
    influx depends on the angle phi between wind vector and radial vector.
    For N, W, S, E: -pi/2 <= phi <= pi/2
    For NW, SW, SE, NE: -pi/4 <= phi <= pi/4
    """

    inwards_wind = np.einsum('ijkl,kl->ijk', wind, r.T * np.expand_dims(normalization, 1)) # dot product of wind vector and radial vector
    lengths_A = np.sqrt(wind[(..., 0)]**2 + wind[(..., 1)]**2)
    angles = np.arccos(inwards_wind / lengths_A)

    allowed_angles = [np.pi/2, np.pi/4, np.pi/2, np.pi/4, np.pi/2, np.pi/4, np.pi/2, np.pi/4]
    inwards_wind[np.abs(angles) > allowed_angles] = 0  # Discard all fluxes that don't point towards center cell
    inwards_wind = inwards_wind / np.expand_dims(normalization, (0, 1))

    nn_vcd = get_moore_neighbourhood(VCD_arr)
    influx = np.einsum('ijk,ijk->ij', inwards_wind, nn_vcd)

    return influx

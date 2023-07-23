import os
from glob import glob

import random
import warnings

import xarray as xr
import pandas as pd
import numpy as np
from sstats import signals as sg
from sstats import sigp as sigp
from sstats import tseries as ts
from sstats import get_cmap_colors

import pynsitu as pyn


"""
PATH
---------------------------------------------
"""
# images_dir = '/Users/mdemol/code/PhD/filtering/images'

# data_dir='/home/datawork-lops-osi/aponte/cswot/drifters' # datarmor
data_dir = "/Users/mdemol/DATA_DRIFTERS/drifters"  # local
# root_dir = '/home1/datahome/mdemol/PhD/insitu_drifters_trajectories' #datarmor
root_dir = "/Users/mdemol/code/PhD/insitu_drifters_trajectories"
# images_dir = '/home1/datahome/mdemol/PhD/insitu_drifters_trajectories/images'#datarmor
images_dir = "/Users/mdemol/ownCloud/PhD/images"  # local

download_dir = data_dir + "/downloads"
raw_dir = data_dir + "/raw"


"""
DATATYPE
---------------------------------------------
"""
KEYS = [
    "carthe_cnr",
    "carthe_lops",
    "code_ogs",
    "svp_ogs",
    "svp_scripps",
    "svp_shom",
    "svp_bcg",
    "spotter_lops",
    "carthe_uwa",
]
color = {
    "carthe_cnr": "darkorange",
    "carthe_lops": "orange",
    "code_ogs": "green",
    "svp_ogs": "darkblue",
    "svp_scripps": "teal",
    "svp_shom": "lightblue",
    "svp_bcg": "blue",
    "spotter_lops": "yellow",
    "carthe_uwa": "coral",
}
columns = [
    "time",
    "lat",
    "lon",
    "x",
    "y",
    "velocity_east",
    "velocity_north",
    "velocity",
    "acceleration_east",
    "acceleration_north",
    "acceleration",
]
"""
SYNTHETIC TRAJ GENERATION
----------------------------------------------
"""
"""
Parameters
"""
# timeline: 100 days with 10 min sampling
dt = 1 / 24
t = (100, dt)
# number of random draws
N = 10

# use a common decorrelation timescale, no rationale
# T = [5,10,20,40]
T = 10

# velocity amplitudes
U_low = 0.3
U_ni = 0.2
U_2 = 0.05
U_1 = 0.02
tau_eta = 0.1  # short timescale
n_layers = 5  # number of layers

"""
ERRORS
----------------------------------------------
"""


def displacement_error(
    ds_true,
    ds_comp,
    xcomp="x",
    ycomp="y",
    xycomp="xy",
    xtrue="x",
    ytrue="y",
    xytrue="xy",
    time="time",
    inplace=True,
):
    if not inplace:
        ds_comp = ds_comp.copy()
    ds_comp["x_error"] = np.sqrt(((ds_comp[xcomp] - ds_true[xtrue]) ** 2).mean(time))
    ds_comp["y_error"] = np.sqrt(((ds_comp[ycomp] - ds_true[ytrue]) ** 2).mean(time))
    ds_comp["xy_error"] = np.sqrt(((ds_comp["xy"] - ds_true["xy"]) ** 2).mean(time))
    if not inplace:
        return ds_comp


def velocity_error(
    ds_true,
    ds_comp,
    ucomp="u",
    vcomp="v",
    uvcomp="uv",
    utrue="u",
    vtrue="v",
    uvtrue="uv",
    time="time",
    inplace=True,
):
    if not inplace:
        ds_comp = ds_comp.copy()
    ds_comp["u_error"] = np.sqrt(((ds_comp[ucomp] - ds_true[utrue]) ** 2).mean(time))
    ds_comp["v_error"] = np.sqrt(((ds_comp[vcomp] - ds_true[vtrue]) ** 2).mean(time))
    ds_comp["uv_error"] = np.sqrt(((ds_comp[uvcomp] - ds_true[uvtrue]) ** 2).mean(time))
    if not inplace:
        return ds_comp


def acceleration_error(
    ds_true,
    ds_comp,
    axcomp="ax",
    aycomp="ay",
    axycomp="axy",
    axtrue="ax",
    aytrue="ay",
    axytrue="axy",
    time="time",
    inplace=True,
):
    if not inplace:
        ds_comp = ds_comp.copy()
    ds_comp["ax_error"] = np.sqrt(((ds_comp[axcomp] - ds_true[axtrue]) ** 2).mean(time))
    ds_comp["ay_error"] = np.sqrt(((ds_comp[aycomp] - ds_true[aytrue]) ** 2).mean(time))
    ds_comp["axy_error"] = np.sqrt(
        ((ds_comp[axycomp] - ds_true[axytrue]) ** 2).mean(time)
    )
    if not inplace:
        return ds_comp


def add_errors(
    ds_true,
    ds_comp,
    xcomp="x",
    ycomp="y",
    xycomp="xy",
    xtrue="x",
    ytrue="y",
    xytrue="xy",
    ucomp="u",
    vcomp="v",
    uvcomp="uv",
    utrue="u",
    vtrue="v",
    uvtrue="uv",
    axcomp="ax",
    aycomp="ay",
    axycomp="axy",
    axtrue="ax",
    aytrue="ay",
    axytrue="axy",
    time="time",
    inplace=True,
):
    if not inplace:
        ds_comp = ds_comp.copy()
    displacement_error(
        ds_true, ds_comp, xcomp, ycomp, xycomp, xtrue, ytrue, xytrue, time
    )
    velocity_error(ds_true, ds_comp, ucomp, vcomp, uvcomp, utrue, vtrue, uvtrue, time)
    acceleration_error(
        ds_true, ds_comp, axcomp, aycomp, axycomp, axtrue, aytrue, axytrue, time
    )
    if not inplace:
        return ds_comp


"""
MEAN SQUARE DATAFRAME
----------------------------------------------
"""


def ms_dataframe(D, list_var=["x", "y", "u", "v", "ax", "ay"]):
    df = pd.DataFrame()
    df["Trajectory"] = D.keys()
    for var in list_vars:
        df[var] = [float((D[l][var] ** 2).mean().values) for l in D]
    df = df.set_index("Trajectory")
    df

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

images_dir = "/Users/mdemol/code/PhD/filtering/images"

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
Functions
"""


def synthetic_traj(t, N, T, tau_eta, n_layers, U_low, U_ni, U_2, U_1):
    list_uv = []
    list_u = []
    list_v = []
    if U_low != None:
        ## low frequency signal: a la Viggiano
        u_low = (
            ts.spectral_viggiano(t, T, tau_eta, n_layers, draws=N, seed=0)
            .compute()
            .rename("u_low")
            * U_low
        )
        v_low = (
            ts.spectral_viggiano(t, T, tau_eta, n_layers, draws=N, seed=1)
            .compute()
            .rename("v_low")
            * U_low
        )
        list_uv += [u_low, v_low]
        list_u.append("u_low")
        list_v.append("v_low")

    if U_ni != None:
        ## near-inertial signal: Sykulski et al. 2016

        f = pyn.geo.coriolis(45) * 86400
        E_ni = lambda omega: 1 / ((omega + f) ** 2 + T**-2)

        uv_ni = ts.spectral(t, spectrum=E_ni, draws=N).compute() * U_ni
        u_ni = np.real(uv_ni).rename("u_ni")
        v_ni = np.imag(uv_ni).rename("v_ni")
        list_uv += [u_ni, v_ni]
        list_u.append("u_ni")
        list_v.append("v_ni")

    ## tidal signals

    # see high frequency spectrum
    # u_high = sg.high_frequency_signal()
    # u_high.analytical_spectrum

    # semi-diurnal
    if U_2 != None:
        omega0 = 2 * np.pi * 2  # semi-diurnal
        E_2 = lambda omega: (omega**2 + omega0**2 + T**-2) / (
            (omega**2 - omega0**2) ** 2
            + T**-2 * (omega**2 + omega0**2)
            + T**-4
        )

        uv_2 = ts.spectral(t, spectrum=E_2, draws=N).compute() * U_2
        u_2 = np.real(uv_2).rename("u_2")
        v_2 = np.imag(uv_2).rename("v_2")
        list_uv += [u_2, v_2]
        list_u.append("u_2")
        list_v.append("v_2")

    # diurnal
    if U_2 != None:
        omega0 = 2 * np.pi  # semi-diurnal
        E_1 = lambda omega: (omega**2 + omega0**2 + T**-2) / (
            (omega**2 - omega0**2) ** 2
            + T**-2 * (omega**2 + omega0**2)
            + T**-4
        )

        uv_1 = ts.spectral(t, spectrum=E_1, draws=N).compute() * U_1
        u_1 = np.real(uv_1).rename("u_1")
        v_1 = np.imag(uv_1).rename("v_1")
        list_uv += [u_1, v_1]
        list_u.append("u_1")
        list_v.append("v_1")

    # combine all time series

    ds = xr.merge(list_uv)
    ds["u"] = sum([ds[u] for u in list_u]).assign_attrs(units="m/s")
    ds["v"] = sum([ds[v] for v in list_v]).assign_attrs(units="m/s")

    ds["time"] = ds["time"].assign_attrs(units="days")
    ds["x"] = ds["u"].cumulative_integrate("time").assign_attrs(units="m") * 86400
    ds["y"] = ds["v"].cumulative_integrate("time").assign_attrs(units="m") * 86400
    ds["ax"] = (
        ds["u"].differentiate("time").assign_attrs(units="m/s^2") / 86400
    )  # rescale in m/s2
    ds["ay"] = (
        ds["v"].differentiate("time").assign_attrs(units="m/s^2") / 86400
    )  # rescale in m/s2

    # transform time in actual dates
    t0 = pd.Timestamp("2000/01/01")
    ds["time_days"] = ds["time"]
    ds["time"] = t0 + ds["time"] * pd.Timedelta("1D")
    return ds


def add_position_noise(ds, t, position_noise, inplace=False):
    # second method: independent noise realizations
    # scale represents the noise
    N = ds.dims["draw"]
    time_dims = ds.dims["time"]
    if not inplace:
        ds = ds.copy()
    ds["x_noise"] = (ds.x.dims, ts.normal(t, draws=N, seed=0).data * position_noise)
    ds["y_noise"] = (ds.x.dims, ts.normal(t, draws=N, seed=1).data * position_noise)
    ds["x"] = ds["x"] + ds["x_noise"]
    ds["y"] = ds["y"] + ds["y_noise"]
    if not inplace:
        return ds


def add_velocity_accelerations(ds, x, y, suffix=""):
    # note: DataArray.differentiate: Differentiate the array with the second order accurate central differences.
    ds["u" + suffix] = x.differentiate("time", datetime_unit="s")
    ds["v" + suffix] = y.differentiate("time", datetime_unit="s")
    ds["ax" + suffix] = ds["u" + suffix].differentiate("time", datetime_unit="s")
    ds["ay" + suffix] = ds["v" + suffix].differentiate("time", datetime_unit="s")


def add_gap(ds, t, T=1, rms=1, threshold=2.0, inplace=False):
    N = ds.dims["draw"]
    if not inplace:
        ds = ds.copy()
    noise = ts.exp_autocorr(t, T, rms, draws=N, seed=20)
    noise1 = noise * 0 + 1
    n = noise1.where(noise < threshold, other=0)
    ds["gaps"] = (n.dims, n.data)
    # apply masking
    for v in ["x", "y"]:
        ds[v] = ds[v].where(ds.gaps == 1)
    if not inplace:
        return ds


"""
def random_time_sampling(ds,t,dt, inplace=False):
    if not inplace : 
        ds=ds.copy()
    offset = ts.uniform(t, low=-dt/2, high=dt/2)*pd.Timedelta("1D")
    ds["time_off"] = (ds.time.dims, ds.time.data + offset.data)
    ds["time_off"] = ds["time_off"].where(ds.time_off>ds.time[0], other=ds.time[0])
    ds["time_off"] = ds["time_off"].where(ds.time_off<ds.time[-1], other=ds.time[-1])
    time_off = ds["time_off"].values
    ds_off = ds.interp(time=time_off)[['x', 'y', 'time_days']]
    ds_off["time_uniform"] = (ds.time.dims, ds.time.data)
    if not inplace : 
        return ds_off
"""


def random_time_sampling(
    ds, t, dt=1 / 24, offset_type="uniform", file=None, inplace=False
):
    if not inplace:
        ds = ds.copy()

    if offset_type == "uniform":
        offset = (ts.uniform(t, low=-dt / 2, high=dt / 2) * pd.Timedelta("1D")).data

    if offset_type == "carthe":
        path_dt = "/Users/mdemol/code/PhD/filtering/example_dt_list/"
        file = path_dt + "/carthe_dt.csv"
        dt = (pd.read_csv(path_dt + "carthe_dt.csv")["dt"] * pd.Timedelta("1s")).values
        try:
            offset = dt[0 : int(np.ceil(t[0] / t[1]))]
        except:
            assert False, "Need more dt in csv files"

    if offset_type == "file":
        try:
            dt = (pd.read_csv(file)["dt"] * pd.Timedelta("1s")).values
        except:
            assert False, "Please give file argument"
        try:
            offset = dt[0 : int(np.ceil(t[0] / t[1]))]
        except:
            assert False, "Need more dt in csv files"

    ds["time_off"] = (ds.time.dims, ds.time.data + offset)
    ds["time_off"] = ds["time_off"].where(ds.time_off > ds.time[0], other=ds.time[0])
    ds["time_off"] = ds["time_off"].where(ds.time_off < ds.time[-1], other=ds.time[-1])
    time_off = ds["time_off"].values
    ds_off = ds.interp(time=time_off)[["x", "y", "time_days"]]
    ds_off["time_uniform"] = (ds.time.dims, ds.time.data)
    if not inplace:
        return ds_off


def cyclic_selection(array, istart, replicate=1):
    cyclic_array = np.concatenate((array[istart:], array[0:istart]))
    if replicate == 1:
        return cyclic_array
    else:
        return np.concatenate([cyclic_array] * replicate)


def cyclic_selection_len(array, istart, n):
    narray = len(array)
    if n == None:
        n = narray
    if n > narray:
        warnings.warn("dt dasaset to small, will contain duplicated values")
    if istart + n < narray:
        return array[istart : istart + n]
    else:
        return np.concatenate((array[istart:], array[0 : n - narray + istart]))


def time_from_dt_array(tstart, tend, dt):
    time_length = np.sum(dt)
    replicate = (tend - tstart) // time_length + 1
    if replicate > 1:
        warnings.warn("dt dasaset to small, will contain duplicated values of dt")
    istart = random.randrange(len(dt))
    dt_ = cyclic_selection(dt, istart, replicate)

    time = xr.DataArray(tstart.values + np.cumsum(dt))
    time = time.where(time < tend, drop=True)
    return time


def irregular_time_sampling(
    ds, t, dt=1 / 24, offset_type="random_uniform", file=None, inplace=False
):
    if not inplace:
        ds = ds.copy()

    if offset_type == "random_uniform":
        offset = (ts.uniform(t, low=-dt / 2, high=dt / 2) * pd.Timedelta("1D")).data
        ds["time_off"] = (ds.time.dims, ds.time.data + offset)

    elif offset_type == "carthe":
        path_dt = "/Users/mdemol/code/PhD/filtering/example_dt_list/"
        file = path_dt + "/carthe_dt.csv"
        typical_dt = pd.Timedelta("300s")
        DT = (pd.read_csv(path_dt + "carthe_dt.csv")["dt"] * pd.Timedelta("1s")).values
        ds["time_off"] = time_from_dt_array(ds.time.min(), ds.time.max(), DT)
        # time = ds.time.data
        # for frequency computation
        # time[1] = time[0]+typical_dt
        # ds['time'] = time
    elif offset_type == "file":
        try:
            dt = (pd.read_csv(file)["dt"] * pd.Timedelta("1s")).values
        except:
            assert False, "Please give file argument"
        try:
            offset = dt[0 : int(np.ceil(t[0] / t[1]))]
        except:
            assert False, "Need more dt in csv files"
    else:
        assert False, "Provide a valid offset_type ( 'random_uniform','carthe', 'file')"
    ds["time_off"] = ds["time_off"].where(ds.time_off > ds.time[0], other=ds.time[0])
    ds["time_off"] = ds["time_off"].where(ds.time_off < ds.time[-1], other=ds.time[-1])
    time_off = ds["time_off"].values
    ds_off = ds.interp(time=time_off)[["x", "y", "time_days"]]
    ds_off["time_uniform"] = xr.DataArray(data=ds.time.data, dims=["time_uniform"])
    if not inplace:
        return ds_off


def add_norm(ds, x="x", y="y", u="u", v="v", ax="ax", ay="ay"):
    ds["xy"] = np.sqrt(ds.x**2 + ds.y**2)
    ds["uv"] = np.sqrt(ds.u**2 + ds.v**2)
    ds["axy"] = np.sqrt(ds.ax**2 + ds.ay**2)


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

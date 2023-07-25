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

from lib import raw_dir, root_dir, images_dir, KEYS


"""
SYNTHETIC TRAJ GENERATION
----------------------------------------------
"""
"""
Default parameters
"""
# timeline: 100 days with 10 min sampling
dt = 1 / 24
t = (100, dt)

ref_case = dict(T = 5, U_low =0.3, U_ni = 0.2, U_2 = 0, U_1 = 0, tau_eta=0.1, n_layers = 5)
typical_case = dict(T = 5, U_low =0.3, U_ni = 0.2, U_2 = 0.05, U_1 = 0.02, tau_eta=2, n_layers = 5)
# number of random draws
N = 50

"""
Functions
"""


def synthetic_traj(
    t, N, T, tau_eta, n_layers, U_low, U_ni, U_2, U_1, all_component_pos_acc=False
):
    """
    Generate a synthetic trajectory
    Parameters:
    -----------
            t : tuple, ndarray or int
                Time series
            N : int,
                nb of draws
            tau_eta : float,
                short correlation scale in days
            n_layers : int
                number of layers for OU process
            U_low : float, 
                background std m/s
            U_ni : float
                amplitude of the inertial term m/s
            U_2 : float, 
                amplitude of semi-diurnal tide component m/s
            U_1 : float, 
                amplitude of diurnal tide component m/s 
            all_componenet_pos_acc : boolean
                compute/include all positions and accelerations for all components
    """
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
        E_ni = lambda omega: 1 / ((omega + f) ** 2 + T**-2)  ##CHANGE + to -

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
    if U_1 != None:
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

    # transform time in actual dates
    t0 = pd.Timestamp("2000/01/01")
    ds["time_days"] = ds["time"].assign_attrs(units="days")
    ds["time"] = t0 + ds["time"] * pd.Timedelta("1D")
    
    def pos_vel_acc(suf='') :
        #compute position
        ds["x"+suf] = ds["u"+suf].cumulative_integrate("time", datetime_unit = 's').assign_attrs(units="m")
        ds["y"+suf] = ds["v"+suf].cumulative_integrate("time", datetime_unit = 's').assign_attrs(units="m")
        #update velocities (diff cumulate_integrate vs differentiate do not give same result, numerical errors)
        ds["u"+suf] = ds["x"+suf].differentiate("time", datetime_unit = 's').assign_attrs(units="m/s")
        ds["v"+suf] = ds["y"+suf].differentiate("time", datetime_unit = 's').assign_attrs(units="m/s")
        # compute acceleration
        ds["au"+suf] = ds["u"+suf].differentiate("time", datetime_unit = 's').assign_attrs(units=r"$m/s^2$")
        ds["av"+suf] = ds["v"+suf].differentiate("time", datetime_unit = 's').assign_attrs(units=r"$m/s^2$")
        
    if all_component_pos_acc:
        pos_vel_acc(suf = '_low')
        pos_vel_acc(suf = '_ni')
        pos_vel_acc(suf = '_2')
        pos_vel_acc(suf = '_1')
    pos_vel_acc()
    
    return ds


def add_position_noise(ds, t, position_noise, ntype='white_noise', update_vel_acc = True, inplace=False):
    """
    Return  noised time dataset
    Parameters:
    -----------
            ds : xarray dataset
                offset_type : 'random_uniform', in lib.KEYS (for dt from insitu dt), 'gdp_raw', 'file' to use the file given by the file argument
            t : tuple, ndarray or int
            position_noise : float,
                position noise std
            ntype : str,
                'white_noise' or 'red_noise' implemented
            update_vel_acc : boolean,
                compute velocities and accelerations from noised positions
            inplace : boolean
    """

    N = ds.dims["draw"]
    
    if not inplace:
        ds = ds.copy()
    
    if ntype == 'white_noise' : 
        xn = ts.normal(t, draws=N, seed=0).data * position_noise
        yn = ts.normal(t, draws=N, seed=1).data * position_noise
        ds["x_noise"] = (ds.x.dims, xn )
        ds["y_noise"] = (ds.x.dims, yn )
    elif ntype == 'red_noise' :
        def E_red(omega):
            E = omega**(-2)
            E= np.where(np.isinf(E),0, E)
            return E
        #E_red = lambda omega: 1 / (omega**2 + f ** 2)
        xy_2 = ts.spectral(t, spectrum=E_red, draws=N).compute()
        #normalize
        #xm = xy_2.mean("time")
        #std = np.sqrt(( np.real(xy_2-xm)**2 + np.imag(xy_2-xm)**2 ).mean("time"))
        #xy_2 = xy_2/std*position_noise
        xn = np.real(xy_2).data
        yn = np.imag(xy_2).data
        
        ds['x_noise'] = (ds.x.dims, xn )
        ds['y_noise'] = (ds.x.dims, yn )
        rmsx = np.sqrt((ds.x_noise**2).mean('time'))
        rmsy = np.sqrt((ds.y_noise**2).mean('time'))
        ds['x_noise'] = ds.x_noise/rmsx*position_noise
        ds['y_noise'] = ds.y_noise/rmsy*position_noise
        
    else :
        assert False

    if update_vel_acc :
        add_velocity_acceleration(ds, ds.x, ds.y, '')
    if not inplace:
        return ds

def add_velocity_acceleration(ds, x, y, suffix="", inplace=True):
    # note: DataArray.differentiate: Differentiate the array with the second order accurate central differences.
    if not inplace:
        ds = ds.copy()
    ds["u" + suffix] = x.differentiate("time", datetime_unit="s")
    ds["v" + suffix] = y.differentiate("time", datetime_unit="s")
    ds["au" + suffix] = ds["u" + suffix].differentiate("time", datetime_unit="s")
    ds["av" + suffix] = ds["v" + suffix].differentiate("time", datetime_unit="s")
    if not inplace:
        return ds

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
IRREGULAR TIME SAMPLING
----------------------------------------------
"""


def cyclic_selection(array, istart, replicate=1):
    """
    Return an array from an other array cycling and replicating
    Parameters:
    -----------
            array : array to cycle or replicate
            istart : int, indice of the start in the given array
            replicate : int,  number of replicate
    """
    cyclic_array = np.roll(array, istart)
    if replicate == 1:
        return cyclic_array
    else:
        return np.concatenate([cyclic_array] * replicate)


def time_from_dt_array(tstart, tend, dt, istart=None):
    """
    Return irregular sampled time list from the dt list. The starting dt in the dt list is randomly chosen and the list is replicated if needed
    Parameters:
    -----------
            tstart : np.datetime, starting time
            tend : np.datetime, ending time
            dt : list of dt
    """
    time_length = np.sum(dt)  # total time length of the dt list
    replicate = (tend - tstart) // time_length + 1
    if replicate > 1:
        warnings.warn("dt dasaset to small, will contain duplicated values of dt")
    if not istart:
        istart = random.randrange(len(dt))
    print(istart)
    dt_ = cyclic_selection(dt, istart, replicate)

    time = xr.DataArray(tstart.values + np.cumsum(dt_))
    time = time.where(time < tend, drop=True)
    return time


def irregular_time_sampling(
    ds,
    offset_type="random_uniform",
    t=None,
    dt=1 / 24,
    file=None,
    istart=None,
    time_uniform=False,
):
    """
    Return irregular sampled time dataset from a dt list or randomly.
    The starting dt in the dt list is randomly chosen and the list is replicated if needed.
    Positions are then interpolated on this new irregularly sampled time.
    Parameters:
    -----------
            ds : xarray dataset
            offset_type : 'random_uniform',
                in lib.KEYS (for dt from insitu dt), 'gdp_raw', 'file' to use the file given by the file argument
            t : tuple, ndarray or int,
                if 'random_uniform'
            dt : float,
                amplitude of the random gap if 'random_uniform'
            file : path to file,
                if offset_type='file'
            istart : int, 
                indice of first dt in the dt file
            time_uniform : boolean,
                keep time_uniform or not
            
    """

    # Random sampling option
    if offset_type == "random_uniform":
        if not t:
            assert False, "provide t"
        offset = (ts.uniform(t, low=-dt / 2, high=dt / 2) * pd.Timedelta("1D")).data
        ds["time_off"] = (ds.time.dims, ds.time.data + offset)

    # Irregular dt from in situ trajectories
    elif '_'.join(offset_type.split('_')[:2]) in KEYS :
        path_dt = os.path.join(
            root_dir, "example_dt_list", "dt_" + offset_type + ".csv"
        )
        typical_dt = pd.Timedelta("300s")
        DT = (pd.read_csv(path_dt)["dt"] * pd.Timedelta("1s")).values
        ds["time_off"] = time_from_dt_array(ds.time.min(), ds.time.max(), DT, istart)
        # time = ds.time.data
        # for frequency computation
        # time[1] = time[0]+typical_dt
        # ds['time'] = time

    # Irregular dt from GDP raw trajectories
    elif offset_type == "gdp_raw":
        path_dt = "/Users/mdemol/code/PhD/filtering/example_dt_list/"
        file = path_dt + "/gdpraw_dt.csv"
        typical_dt = pd.Timedelta("60min")
        DT = (
            pd.read_csv(path_dt + "gdpraw_dt.csv")["dt"] * pd.Timedelta("1min")
        ).values
        ds["time_off"] = time_from_dt_array(ds.time.min(), ds.time.max(), DT)

    # Others
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
        assert False, "Provide a valid offset_type ( 'random_uniform', KEYS, 'file')"

    ds["time_off"] = ds["time_off"].where(
        ds.time_off > ds.time[0], other=ds.time[0]
    )  # end and start
    ds["time_off"] = ds["time_off"].where(ds.time_off < ds.time[-1], other=ds.time[-1])

    time_off = ds["time_off"].values
    # interpolate data of the new irregular sampling
    ds_off = ds.interp(time=time_off)[["x", "y", "time_days"]]  
    # interpolate noise of the new irregular sampling
    if 'x_noise' in ds:
        ds_off_noise = ds.interp(time=time_off, method = 'nearest')[["x_noise", "y_noise"]] 
    
    ds_off = xr.merge([ds_off, ds_off_noise])
    
    if time_uniform:
        ds_off["time_uniform"] = xr.DataArray(
            data=ds.time.data, dims=["time_uniform"]
        )  # keep regular dt
    return ds_off

"""
APPLY BOTH NOISE AND IRREGULAR SAMPLING
----------------------------------------------
"""
def noise_irregular_sampling(ds,
                             t,
                             position_noise,
                             ntype='white_noise',
                             update_vel_acc = True,
                             offset_type="random_uniform",
                             dt=1 / 24,
                             file=None,
                             istart=None, 
                             time_uniform=False,
                            ):
    """
    Return irregular sampled and noised time dataset
    Parameters:
    -----------
            ds : xarray dataset
                offset_type : 'random_uniform', in lib.KEYS (for dt from insitu dt), 'gdp_raw', 'file' to use the file given by the file argument
            t : tuple, ndarray or int
            position_noise : float,
                position noise std
            ntype : str,
                'white_noise' or 'red_noise' implemented
            update_vel_acc : boolean,
                compute velocities and accelerations from noised positions
            offset_type : 'random_uniform',
                in lib.KEYS (for dt from insitu dt), 'gdp_raw', 'file' to use the file given by the file argument
            dt : float,
                amplitude of the random gap if 'random_uniform'
            file : path to file,
                if offset_type='file'
            istart : int, 
                indice of first dt in the dt file
            time_uniform : boolean,
                keep time_uniform or not
    """
    ds = ds.copy()
    add_position_noise(ds, t, position_noise, ntype, update_vel_acc, inplace=True)
    ds = irregular_time_sampling(ds, offset_type, t, dt, file, istart, time_uniform=time_uniform)
    
    ds['x'] = ds['x']+ds['x_noise']
    ds['y'] = ds['y']+ds['y_noise']
    
    add_velocity_acceleration(ds, ds.x, ds.y)
    return ds

"""
OTHER
----------------------------------------------
"""

def add_norm(ds, x="x", y="y", u="u", v="v", ax="ax", ay="ay", prefix=""):
    ds["xy" + prefix] = np.sqrt(ds["x" + prefix] ** 2 + ds["y" + prefix] ** 2)
    ds["uv" + prefix] = np.sqrt(ds["u" + prefix] ** 2 + ds["v" + prefix] ** 2)
    ds["axy" + prefix] = np.sqrt(ds["ax" + prefix] ** 2 + ds["ay" + prefix] ** 2)


def negpos_spectra(ds, freqkey="frequency"):
    """Return two datasets with cyclonic/anticyclonic spectra"""
    ds_inv = ds.sortby(freqkey, ascending=False)
    dsneg = ds_inv.where(ds_inv[freqkey] <= 0, drop=True)
    dsneg[freqkey] = -dsneg[freqkey]
    dspos = ds.where(ds[freqkey] >= 0, drop=True)
    return dsneg, dspos



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
DATASET TO DATAFRAME (MIMICS INSITU DATA TO APPLY SMOOTHING METHOD)
----------------------------------------------
"""


def dataset2dataframe(ds, velocity=True, names=("u", "v", "U")):
    DF = []
    for d in ds.draw:
        df = ds.sel(draw=d).to_dataframe()
        if velocity:
            pyn.drifters.compute_velocities(
                df,
                "x",
                "y",
                "index",
                names=names,
                distance=None,
                inplace=True,
                centered=True,
                fill_startend=False,
            )
        DF.append(df)
    return pd.concat(DF).rename(columns={"draw": "id"})


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

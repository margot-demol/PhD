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

import matplotlib.pyplot as plt

import hvplot.xarray
import hvplot.pandas
import holoviews as hv

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

ref_case = dict(T = 5, U_low =0.3, U_ni = 0.2, U_2 = 0, U_1 = 0, tau_eta=0.1, n_layers = 5, spectral_diff = False)
typical_case = dict(T = 5, U_low =0.3, U_ni = 0.2, U_2 = 0.05, U_1 = 0.02, tau_eta=0.1, n_layers = 5, spectral_diff = False)
# number of random draws
N = 50
T =5

"""
Functions
"""
def dataset2dataframe(ds):
    DF = []
    for d in ds.draw:
        df = ds.sel(draw=d).to_dataframe()
        DF.append(df)
    return pd.concat(DF)

def pos_vel_acc_spectral(df, dt, suf='') :
    #compute position
    df["x"+suf] = df.groupby('draw')['u'+suf].apply(pyn.geo.spectral_diff, dt, order = -1)
    df["y"+suf] = df.groupby('draw')['v'+suf].apply(pyn.geo.spectral_diff, dt, order = -1)
    #update velocities (diff cumulate_integrate vs differentiate do not give same result, numerical errors)
    df["u"+suf] = df.groupby('draw')['x'+suf].apply(pyn.geo.spectral_diff, dt, order = 1)
    df["v"+suf] = df.groupby('draw')['y'+suf].apply(pyn.geo.spectral_diff, dt, order = 1)
    df["U"+suf] = np.sqrt(df["u"+suf]**2+df["v"+suf]**2)
    # compute acceleration
    df["ax"+suf] = df.groupby('draw')['x'+suf].apply(pyn.geo.spectral_diff, dt, order =2)
    df["ay"+suf] = df.groupby('draw')['y'+suf].apply(pyn.geo.spectral_diff, dt, order =2)
    df["Axy"+suf] = np.sqrt(df["ax"+suf]**2+df["ay"+suf]**2)
        
def synthetic_traj(
    t, N, T, tau_eta, n_layers, U_low, U_ni, U_2, U_1, out_put='dataset', all_comp_pos_acc =False, spectral_diff=False
):
    """
    Generate a synthetic trajectory
    Parameters:
    -----------
            t : tuple, ndarray or int or str
                Time series
            N : int,
                nb of draws
            T : float 
                long correleration time scale in days
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
            out_put : str,
            all_comp_pos_acc : boolean
    """
    list_uv = []
    list_u = []
    list_v = []
    
    t0 = pd.Timestamp("2000/01/01")
    type_t = isinstance(t[1], str)
    if type_t:
        t_date = pd.date_range(t0, t0 + t[0]* pd.Timedelta("1D"), freq=t[1])
        t = (t_date -t0)/pd.Timedelta("1D")
    
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
    ds["uo"] = sum([ds[u] for u in list_u]).assign_attrs(units="m/s")
    ds["vo"] = sum([ds[v] for v in list_v]).assign_attrs(units="m/s")
    
    # transform time in actual dates
    if not type_t :
        t_date = t0 + ds['time'] * pd.Timedelta("1D")
    ds["time_days"] = ds["time"].assign_attrs(units="days")
    ds["time"] = t_date
    
    if not spectral_diff :
        ds["x"] = ds["uo"].cumulative_integrate("time", datetime_unit='s').assign_attrs(units="m")
        ds["y"] = ds["vo"].cumulative_integrate("time", datetime_unit='s').assign_attrs(units="m")
        ds['X'] = np.sqrt(ds['x']**2 + ds['y']**2)
        if all_comp_pos_acc :
            def pos(ds, suf=''):
                ds["x"+suf] = ds["u"+suf].cumulative_integrate("time", datetime_unit = 's').assign_attrs(units="m")
                ds["y"+suf] = ds["v"+suf].cumulative_integrate("time", datetime_unit = 's').assign_attrs(units="m")
                ds['X'+suf] = np.sqrt(ds['x'+suf]**2 + ds['y'+suf]**2)

            pos(ds, suf ='_low')
            pos(ds, suf ='_ni')
            pos(ds, suf ='_1')
            pos(ds, suf='_2')
        print('centred diff')

    # compute xy, uv and axy in fourier space
    # dataframe
    df = dataset2dataframe(ds)
    dt = (df.index[1] - df.index[0])/pd.Timedelta('1s')
    
    if spectral_diff:
        #xy
        df['x'] = df.groupby('draw')['uo'].apply(pyn.geo.spectral_diff , dt, order = -1)
        df['y'] = df.groupby('draw')['vo'].apply(pyn.geo.spectral_diff, dt, order = -1)
        df['X'] = np.sqrt(df['x']**2+df['y']**2)
        #uv
        df['u'] = df.groupby('draw')['x'].apply(pyn.geo.spectral_diff, dt, order = 1)
        df['v'] = df.groupby('draw')['y'].apply(pyn.geo.spectral_diff, dt, order = 1)
        df['U'] = np.sqrt(df['u']**2+df['v']**2)
        #axay
        df['ax'] = df.groupby('draw')['x'].apply(pyn.geo.spectral_diff, dt, order = 2)
        df['ay'] = df.groupby('draw')['y'].apply(pyn.geo.spectral_diff, dt, order = 2)
        df['Axy'] = np.sqrt(df['ax']**2+df['ay']**2)
        #axay
        df['au'] = df.groupby('draw')['u'].apply(pyn.geo.spectral_diff, dt, order = 1)
        df['av'] = df.groupby('draw')['v'].apply(pyn.geo.spectral_diff, dt, order = 1)
        df['Auv'] = np.sqrt(df['au']**2+df['av']**2)
        
        if all_comp_pos_acc:
            pos_vel_acc_spectral(df, dt, suf = '_low')
            pos_vel_acc_spectral(df, dt, suf = '_ni')
            pos_vel_acc_spectral(df, dt, suf = '_2')
            pos_vel_acc_spectral(df, dt, suf = '_1')
        print('spectral_diff')
    else : 
        #uv
        df = df.groupby('draw').apply(pyn.geo.compute_velocities,time='index', distance='xy', names=('u', 'v', 'U'), fill_startend=True, centered=True)
        #axay
        df = df.groupby('draw').apply(pyn.geo.compute_accelerations,from_ =('xy', 'x', 'y'), names=('ax', 'ay', 'Axy'))
        #auav
        df = df.groupby('draw').apply(pyn.geo.compute_accelerations,from_ =('velocities', 'u', 'v'), names=('au', 'av', 'Auv'))
        print('centred diff')
        if all_comp_pos_acc : 
            for suf in ['_low', '_ni', '_2', '_1']:
                df = df.groupby('draw').apply(pyn.geo.compute_accelerations,from_ =('xy', 'x'+suf, 'y'+suf), names=('ax'+suf, 'ay'+suf, 'Axy'+suf))
            
    df = df.groupby('draw').apply(pyn.geo.compute_dt, time='index')
    print(df.x.mean())
    if out_put == 'dataset' : 
        df = df.reset_index().set_index(['time', 'draw'])
        ds = df.to_xarray()
        return ds
    else : 
        return df
    return ds

def add_position_noise(ds, t, position_noise, ntype='white_noise', inplace=False):

    N = ds.dims["draw"]
    
    type_t = isinstance(t[1], str)
    if type_t:
        t0 = pd.Timestamp("2000/01/01")
        t_date = pd.date_range(t0, t0 + t[0]* pd.Timedelta("1D"), freq=t[1])
        t = (t_date -t0)/pd.Timedelta("1D")
        
        
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
            offset_type : 'random_uniform', in lib.KEYS (for dt from insitu dt), 'gdp_raw', 'file' to use the file given by the file argument
            t : tuple, ndarray or int, if 'random_uniform'
            dt : float, amplitude of the random gap if 'random_uniform'
            file : path to file,  if offset_type='file'
            inplace : boolean
    """
    ds = ds.copy()
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
        #typical_dt = pd.Timedelta("300s")
        DT = (pd.read_csv(path_dt)["dt"] * pd.Timedelta("1s")).values
        ds["time_off"] = time_from_dt_array(ds.time.min(), ds.time.max(), DT, istart)

    # Irregular dt from GDP raw trajectories
    elif offset_type == "gdp_raw":
        path_dt = "/Users/mdemol/code/PhD/filtering/example_dt_list/"
        file = path_dt + "/gdpraw_dt.csv"
        #typical_dt = pd.Timedelta("60min")
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
                             offset_type="random_uniform",
                             dt=1 / 24,
                             file=None,
                             istart=None, 
                             time_uniform=False,
                             method_diff = 'pyn',
                             noise_true = False,
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
            offset_type : 'random_uniform',
                in lib.KEYS (for dt from insitu dt), 'gdp_raw', 'file' to use the file given by the file argument
            dt : float,
                amplitude of the random gap if 'random_uniform'
            file : path to file,
                if offset_type='file'
            istart : int, 
                indice of first dt in the dt file
            time_uniform : boolean,
    """    
    #NOISE
    ds = add_position_noise(ds, t, position_noise, ntype)
    noise_t = ds[['x_noise','y_noise']]
    #IRREGULAR TIME SAMPLING
    ds = irregular_time_sampling(ds, offset_type, t, dt, file, istart, time_uniform=time_uniform)
    
    #MERGE
    ds['x'] = ds['x']+ds['x_noise']
    ds['y'] = ds['y']+ds['y_noise']
    ds['X'] = np.sqrt(ds['x']**2 + ds['y']**2)

    # COMPUTE VELOCITIES/ACCELERATIONS (default is by spectral methods)
    ds = add_velocity_acceleration(ds, method = method_diff)
    if noise_true : 
        return ds, noise_t
    return ds

def add_velocity_acceleration(ds, suffix="", method ='pyn', groupby = 'draw'):
    ds = ds.copy()
    # note: DataArray.differentiate: Differentiate the array with the second order accurate central differences.
    if method == 'spectral' :
        dt = ds.time.diff('time')/pd.Timedelta('1s')
        assert (dt[1:] == dt[1]).all(), 'time must be regularly sampled to apply spectral method'
        df = dataset2dataframe(ds)
        df["u" + suffix] = pyn.geo.spectral_diff(df['x'], dt, order = 1)
        df["v" + suffix] = pyn.geo.spectral_diff(df['y'], dt, order = 1)
        df['U'+ suffix] = np.sqrt(df['u'+ suffix]**2+df['v'+ suffix]**2)
        df["ax" + suffix] = pyn.geo.spectral_diff(df['x'], dt, order = 2)
        df["ay" + suffix] = pyn.geo.spectral_diff(df['y'], dt, order = 2)
        df['Axy'+ suffix] = np.sqrt(df['ax'+ suffix]**2+df['ay'+ suffix]**2)
        df = df.reset_index().set_index(['time', 'draw'])
        ds = df.to_xarray()
        return ds
    elif method == 'xr':
        ds["u" + suffix] = ds['x'].differentiate("time", datetime_unit="s")
        ds["v" + suffix] = ds['y'].differentiate("time", datetime_unit="s")
        ds['U'+ suffix] = np.sqrt(ds['u'+ suffix]**2+ds['v'+ suffix]**2)
        ds["au" + suffix] = ds["u" + suffix].differentiate("time", datetime_unit="s")
        ds["av" + suffix] = ds["v" + suffix].differentiate("time", datetime_unit="s")
        ds['Auv'+ suffix] = np.sqrt(ds['au'+ suffix]**2+ds['av'+ suffix]**2)
        return ds
    elif method =='pyn':
        df = dataset2dataframe(ds)
        
        df = df.groupby(groupby).apply(pyn.geo.compute_velocities, 'index', ("u" + suffix, "v" + suffix, "U" + suffix), centered = True,fill_startend = True,  distance ='xy')
        
        df = df.groupby(groupby).apply(pyn.geo.compute_accelerations, from_ = ('xy', 'x'+suffix, 'y'+suffix), names = ('ax' + suffix, 'ay' + suffix, 'Axy' + suffix))
        
        df = df.groupby(groupby).apply(pyn.geo.compute_accelerations, from_ = ('velocities', 'u'+suffix, 'v'+suffix), names = ('au' + suffix, 'av' + suffix, 'Auv' + suffix))
        
        df = df.reset_index().set_index(['time', 'draw'])
        ds = df.to_xarray()
        return ds
    else : 
        assert False, "method smust be 'xr' or 'pyn' or 'spectral'"
"""
OTHER
----------------------------------------------
"""
def negpos_spectra(ds, freqkey="frequency"):
    """Return two datasets with cyclonic/anticyclonic spectra"""
    ds_inv = ds.sortby(freqkey, ascending=False)
    dsneg = ds_inv.where(ds_inv[freqkey] <= 0, drop=True)
    dsneg[freqkey] = -dsneg[freqkey]
    dspos = ds.where(ds[freqkey] >= 0, drop=True)
    return dsneg, dspos

"""
DIAGNOSTIC
----------------------------------------------
"""
def hvplot_DF(DF, d, var=None):
    if not var :
        var = ['x', 'y', 'u','v','ax', 'ay']
    Hv = []
    for v in var :
        init = 0
        for l in DF :
            df = DF[l]
            if init == 0:
                hvplot = df[df.id == d][v].hvplot(label = l )
                init = 1
            hvplot *= df[df.id == d][v].hvplot(label = l )
        Hv.append(hvplot)
    print(len(Hv))
    layout = hv.Layout(Hv[0] + Hv[1] + Hv[2] + Hv[3] + Hv [4]+ Hv [5]).cols(2)
    return layout

# MS VALUES ##############################################################
def ms_diff(DF, true_key, var = ['x', 'y', 'u','v','ax', 'ay', 'X', 'U', 'Axy']):
    DF = DF.copy()
    dft = DF[true_key]
    dft_ = (dft.set_index('id')[var]**2).groupby('id').mean()
    dfms  = pd.DataFrame(index = DF.keys(), columns = var)
    dfmsr = pd.DataFrame(index = DF.keys(), columns = var)
    for l in DF :
        df = DF[l]
        if np.all(df.index.values == dft.index.values):
            df_ = df.set_index('id')[var]-dft.set_index('id')[var]
            #dfr_ = (df.set_index('id')[var]-dft.set_index('id')[var])/dft.set_index('id')[var]
            dfms.loc[l] = (df_**2).groupby('id').mean().mean()
            dfmsr.loc[l] = ((df_**2).groupby('id').mean()/dft_).mean()
        else : 
            print(l + ' has not the same time index')
            continue
    dfms = pd.concat([dfms, dfmsr.rename(columns = {v : 'ratio_'+v for v in var})], axis=1).dropna()
    return dfms

def diff_ms(DF, true_key, var = ['x', 'y', 'u','v','ax', 'ay', 'X', 'U', 'Axy']):
    DF = DF.copy()
    dft = DF[true_key]
    dfms  = pd.DataFrame(index = DF.keys(), columns = var)
    dfmsr = pd.DataFrame(index = DF.keys(), columns = var)
    dft_ = (dft.set_index('id')[var]**2).groupby('id').mean()#mean sur une traj
    for l in DF :
        df = DF[l]
        dfms.loc[l] = abs((df.set_index('id')[var]**2).groupby('id').mean()-dft_).mean()#mean sur une traj puis sur ttes
        dfmsr.loc[l] =abs(((dft.set_index('id')[var]**2).groupby('id').mean()-dft_)/dft_).mean()
    dfms = pd.concat([dfms, dfmsr.rename(columns = {v : 'ratio_'+v for v in var})], axis=1).dropna()    
    return dfms

def ratio_ms(DF, true_key = 'True_2h', var = ['x', 'y', 'u','v','ax', 'ay', 'X', 'U', 'Axy']):
    DF = DF.copy()
    dft = DF[true_key]
    dfms  = pd.DataFrame(index = DF.keys(), columns = var)
    dft_ = (dft.set_index('id')[var]**2).groupby('id').mean()#mean sur une traj
    for l in DF :
        df = DF[l]
        dfms.loc[l] = ((df.set_index('id')[var]**2).groupby('id').mean()/dft_).mean()#mean sur une traj puis sur ttes
    return dfms.dropna()

# SPECTRA ##############################################################
def spectrum_df(df, nperseg='20D', detrend=False):
    if 'gap_mask' in df and len(df[df.gap_mask ==1] )!= 0 :
        print('WARNING : INTERPOLATED HOLES')
    ds = df.reset_index().set_index(['time', 'id']).to_xarray()
    vc = [("x", "y"), ("u", "v"), ("ax","ay")]
    if 'ax_noise' in ds :
        vc+=[("x_noise", "y_noise"), ("u_noise", "v_noise"), ("ax_noise","ay_noise")]
    E = []
    for tup in vc :
        ds_ = ds.ts.spectrum(unit = '1D',nperseg=nperseg,detrend=False, complex=tup)
        E.append(ds_)
    return xr.merge(E)

def test_regular_dt(df):
    if df.index.name == 'time' : 
        return np.all(np.round(df.dt.dropna().values) == np.round((df.index[3]-df.index[2])/pd.Timedelta('1s')))
    else : 
        return np.all(np.round(df.dt.dropna().values) == np.round((df.time[3]-df.time[2])/pd.Timedelta('1s')))

def spectrum_DF(DF, nperseg='20D', detrend = False) :
    DSE = dict()
    for l in DF :
        df = DF[l]
        test = test_regular_dt(df) 
        if test_regular_dt(df) :
            DSE[l] = spectrum_df(df, nperseg)    
            print(l)
        else :
            continue
    return DSE

def DSE2dse(DSE):
    ds = xr.concat(DSE.values()[['x_y', 'u_v', 'ax_ay']], dim =list(DSE.keys()))
    return ds.rename({'concat_dim':'Trajectory'})

# SPECTRA OF THE DIFF ##############################################################
def build_DF_diff(DF, true_key, var = ['x', 'y', 'u','v','ax', 'ay', 'X', 'U', 'Axy']):
    dft = DF[true_key]
    DF_diff = dict()
    for l in DF :
        if l ==true_key : continue
        df = DF[l]
        if np.all(df.index.values == dft.index.values):
            df_ = df[var]-dft[var]
            df_['dt'] = df['dt']
            df_['id'] = df['id']
            DF_diff[l] = df_
        else : 
            print(l + ' has not the same time index')
            continue
    return DF_diff

# INTEGRATION PER BAND ##############################################################
def var_per_band(ds, fmin, fmax):
    return ds.where((ds.frequency >= fmin) & (ds.frequency < fmax), drop=True).integrate("frequency")

def ds_int_band(ds, LF = (0,0.5), NI = (0.5,2.5), HF = (2.5, 6)):
    if np.any(ds.frequency<0):
        ds = sum(negpos_spectra(ds))
    BF = var_per_band(ds, LF[0], LF[1]).rename({v:'bf_'+v for v in [v for v in ds.keys() if v not in ds.coords]})
    NI = var_per_band(ds, NI[0], NI[1]).rename({v:'ni_'+v for v in [v for v in ds.keys() if v not in ds.coords]})
    HF = var_per_band(ds, HF[0], HF[1]).rename({v:'hf_'+v for v in [v for v in ds.keys() if v not in ds.coords]})
    return xr.merge([BF, NI, HF])


def diagnostics(DF,param_name, num, true_key = 'True_2h', d=0) : 
    
    #ms of the diff
    dfms = ms_diff(DF, true_key)
    dfms = dfms[dfms.index != true_key]
    n =len(dfms)
    if num :
        dfms[param_name]=[float(v.split('=')[-1]) for v in dfms.index]
        dfms = dfms.reset_index().set_index(param_name)
        fig, axs = plt.subplots(3,3,sharex=True, figsize = (8,5))
        i=0
        axs = axs.flatten()
        for var in ['x', 'y','X', 'u','v','U', 'ax','ay','Axy']  : 
            ax=axs[i]
            dfms[var].plot(ax=ax, color ='teal', marker ='+')
            ax.grid()
            i+=1
            ax.set_ylabel(fr'$\langle ({var}-{var}_t)^2 \rangle $')
        fig.suptitle(fr'$\langle (\alpha-\alpha_t)^2 \rangle $ depending on '+ param_name)
        fig.tight_layout()
        
        fig1, axs = plt.subplots(3,3,sharex=True, figsize = (8,5))
        i=0
        axs = axs.flatten()
        for var in ['x', 'y','X', 'u','v','U', 'ax','ay','Axy']: 
            ax=axs[i]
            dfms['ratio_'+var].plot(ax=ax, color ='teal', marker ='+')
            ax.grid()
            i+=1
            ax.set_ylabel(fr'$\langle ({var}-{var}_t)^2 \rangle / \langle {var}_t^2 \rangle $')
        fig1.suptitle(fr'$\langle (\alpha-\alpha_t)^2 \rangle/ \langle \alpha_t^2 \rangle $ depending on '+ param_name)
        fig1.tight_layout()
        
    else : 
        fig, axs = plt.subplots(3,2, sharey=True, figsize = (8,n))
        i=0
        axs = axs.flatten()
        for var in ['x', 'y', 'u','v','ax','ay' 'X', 'U', 'Axy'] : 
            ax=axs[i]
            dfms[var].plot.barh(ax=ax, color ='teal', width =0.8)
            ax.grid()
            ax.set_xlim(-0.1*dfms[var].max(), dfms[var].max()*1.5)
            i+=1
            ax.set_xlabel(fr'$\langle ({var}-{var}_t)^2 \rangle$')
            ax.bar_label(ax.containers[0], labels =[ np.format_float_scientific(l,precision = 4, exp_digits=2) for l in dfms[var].values])
        
        fig.suptitle(fr'$\langle (\alpha-\alpha_t)^2 \rangle $')
        fig.tight_layout()
        
        fig1, axs = plt.subplots(3,2, sharey=True, figsize = (8,n))
        i=0
        axs = axs.flatten()
        for var in ['x', 'y', 'u','v','ax','ay' 'X', 'U', 'Axy']  : 
            ax=axs[i]
            dfms['ratio_' + var].plot.barh(ax=ax, color ='teal', width =0.8)
            ax.grid()
            ax.set_xlim(-0.1*dfms['ratio_' + var].max(), dfms['ratio_' + var].max()*1.5)
            i+=1
            ax.set_xlabel(fr'$\langle ({var}-{var}_t)^2 \rangle / \langle {var}_t^2 \rangle $')
            ax.bar_label(ax.containers[0], labels =[ np.format_float_scientific(l,precision = 4, exp_digits=2) for l in dfms['ratio_' +var].values])
        fig1.suptitle(fr'$\langle (\alpha-\alpha_t)^2 \rangle/ \langle \alpha_t^2 \rangle $')
        fig1.tight_layout()
    """
    #diff of the ms
    dfmsd = diff_ms(DF, true_key)
    dfmsd = dfmsd[dfmsd.index != true_key]
    nd= len(dfmsd)
    
    fig2, axs = plt.subplots(3,2, sharey=True, figsize = (8,nd))
    i=0
    axs = axs.flatten()
    for var in ['x', 'y', 'u','v','ax','ay'] : 
        ax=axs[i]
        dfmsd[var].plot.barh(ax=ax, color ='teal', width =0.8)
        ax.grid()
        ax.set_xlim(-0.1*dfmsd[var].max(), dfmsd[var].max()*1.5)
        i+=1
        ax.set_xlabel(fr'$abs(\langle {var}^2 \rangle - \langle {var}_t^2 \rangle )$')
        ax.bar_label(ax.containers[0], labels =[ np.format_float_scientific(l,precision = 4, exp_digits=2) for l in dfmsd[var].values])
    fig2.suptitle(fr'$abs(\langle \alpha^2 \rangle - \langle \alpha_t^2 \rangle )$')
    fig2.tight_layout()

    fig3, axs = plt.subplots(3,2, sharey=True, figsize = (8,nd))
    i=0
    axs = axs.flatten()
    for var in ['x', 'y', 'u','v','ax','ay'] : 
        ax=axs[i]
        dfmsd['ratio_' + var].plot.barh(ax=ax, color ='teal', width =0.8)
        ax.grid()
        ax.set_xlim(-0.1*dfmsd['ratio_' + var].max(), dfmsd['ratio_' + var].max()*1.5)
        i+=1
        ax.set_xlabel(fr'$ abs(\langle {var}^2 \rangle - \langle {var}_t^2 \rangle))/\langle {var}_t^2 \rangle$')
        ax.bar_label(ax.containers[0], labels =[ np.format_float_scientific(l,precision = 4, exp_digits=2) for l in dfmsd['ratio_' +var].values])
    fig3.suptitle(fr'$ abs(\langle \alpha^2 \rangle - \langle \alpha_t^2 \rangle))/\langle \alpha_t^2 \rangle$')
    fig3.tight_layout()

    #RATIO OF MS
    dfra = ratio_ms(DF, true_key)
    fig9, axs = plt.subplots(3,2, sharey=True, figsize = (8,nd))
    i=0
    axs = axs.flatten()
    for var in ['x', 'y', 'u','v','ax','ay'] : 
        ax=axs[i]
        dfra[var].plot.barh(ax=ax, color ='teal', width =0.8)
        ax.axvline(1, ls='--', color='r')
        ax.grid()
        ax.set_xlim(-0.1*dfra[var].max(), dfra[var].max()*1.5)
        i+=1
        ax.set_xlabel(fr'$\langle {var}^2 \rangle /\langle {var}_t^2 \rangle$')
        ax.bar_label(ax.containers[0], labels =[ np.format_float_scientific(l,precision = 4, exp_digits=2) for l in dfra[var].values])
    fig9.suptitle(fr'$\langle \alpha^2 \rangle /\langle \alpha_t^2 \rangle$')
    fig9.tight_layout()
    """
    """
    #SPECTRA
    DSE = spectrum_DF(DF)
    dse = DSE2dse(DSE)
    dsen = sum(negpos_spectra(dse))
    fig4, axs = plt.subplots(1, 3, figsize=(10,5))

    for l in dsen.Trajectory :
        dsen.x_y.mean('id').plot(hue='Trajectory', ax=axs[0], add_legend=True)
        dsen.u_v.mean('id').plot(hue='Trajectory', ax=axs[1], add_legend=False)
        dsen.ax_ay.mean('id').plot(hue='Trajectory', ax=axs[2], add_legend=False)
    for ax in axs :
        ax.grid()
        ax.set_xscale('log')
        ax.set_yscale('log')
    fig4.suptitle("PSD")
    fig4.tight_layout()
    """
    """
    # SPECTRA OF THE DIFF
    DF_diff = build_DF_diff(DF, true_key)
    DSE_d = spectrum_DF(DF_diff)
    dse_d = DSE2dse(DSE_d)
    dsen_d = sum(negpos_spectra(dse_d))
    
    fig5, axs = plt.subplots(1, 3, figsize=(10,5))

    for l in dsen.Trajectory :
        dsen_d.x_y.mean('id').plot(hue='Trajectory', ax=axs[0], add_legend=True)
        dsen_d.u_v.mean('id').plot(hue='Trajectory', ax=axs[1], add_legend=False)
        dsen_d.ax_ay.mean('id').plot(hue='Trajectory', ax=axs[2], add_legend=False)
    for ax in axs :
        ax.grid()
        ax.set_xscale('log')
        ax.set_yscale('log')
    fig5.suptitle("PSD of the difference")
    fig5.tight_layout()
    
    #INT PER BAND
    dsib = ds_int_band(dsen).mean('id')
    dfib = dsib.to_dataframe().dropna()
    
    fig6, axs = plt.subplots(3,1, sharey=True, figsize = (8,n))
    i=0
    axs = axs.flatten()
    for var in ['bf_x_y', 'ni_x_y', 'hf_x_y'] : 
        ax =axs[i]
        dfib[var].plot.barh(ax=ax, color ='teal', width =0.8)
        ax.bar_label(ax.containers[0], labels =[ np.format_float_scientific(l,precision = 4, exp_digits=2) for l in dfib[var].values])
        ax.grid()
        ax.set_xlim(-0.1*dfib[var].max(), dfib[var].max()*1.5)
        ax.set_xlabel(var)
        i+=1
    axs[0].set_xlabel('BF')
    axs[1].set_xlabel('NI')
    axs[2].set_xlabel('HF')
    fig6.suptitle('Integration per band on position (x,y) PSD')   
    fig6.tight_layout()
    
    fig7, axs = plt.subplots(3,1, sharey=True, figsize = (8,n))
    i=0
    axs = axs.flatten()
    for var in ['bf_u_v', 'ni_u_v', 'hf_u_v'] : 
        ax =axs[i]
        dfib[var].plot.barh(ax=ax, color ='teal', width =0.8)
        ax.bar_label(ax.containers[0], labels =[ np.format_float_scientific(l,precision = 4, exp_digits=2) for l in dfib[var].values])
        ax.grid()
        ax.set_xlim(-0.1*dfib[var].max(), dfib[var].max()*1.5)
        ax.set_xlabel(var)
        i+=1
    axs[0].set_xlabel('BF')
    axs[1].set_xlabel('NI')
    axs[2].set_xlabel('HF')
    fig7.suptitle('Integration per band on velocities (u,v) PSD')  
    fig7.tight_layout()
    
    fig8, axs = plt.subplots(3,1, sharey=True, figsize = (8,n))
    i=0
    axs = axs.flatten()
    for var in ['bf_ax_ay', 'ni_ax_ay', 'hf_ax_ay'] : 
        ax =axs[i]
        dfib[var].plot.barh(ax=ax, color ='teal', width =0.8)
        ax.bar_label(ax.containers[0], labels =[ np.format_float_scientific(l,precision = 4, exp_digits=2) for l in dfib[var].values])
        ax.grid()
        ax.set_xlim(-0.1*dfib[var].max(), dfib[var].max()*1.5)
        ax.set_xlabel(var)
        i+=1
    axs[0].set_xlabel('BF')
    axs[1].set_xlabel('NI')
    axs[2].set_xlabel('HF')
    fig8.suptitle('Integration per band on acceleration (ax,ay) PSD')  
    fig8.tight_layout()
    """
    #return dfms, dfmsd,dfra, dse, dse_d, dfib, fig, fig1, fig2, fig3,fig9, fig4, fig5, fig6, fig7, fig8
    return dfms

def ms_dataframe(D, list_var=["x", "y", "u", "v", "ax", "ay"]):
    df = pd.DataFrame()
    df["Trajectory"] = D.keys()
    for var in list_vars:
        df[var] = [float((D[l][var] ** 2).mean().values) for l in D]
    df = df.set_index("Trajectory")
    df

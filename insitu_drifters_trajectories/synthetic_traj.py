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

ref_case = dict(T = 5, U_low =0.3, U_ni = 0.2, U_2 = 0, U_1 = 0, tau_eta=0.1, n_layers = 5)
typical_case = dict(T = 5, U_low =0.3, U_ni = 0.2, U_2 = 0.05, U_1 = 0.02, tau_eta=2, n_layers = 5)
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

def pos_vel_acc(df, dt, suf='') :
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
    t, N, T, tau_eta, n_layers, U_low, U_ni, U_2, U_1, out_put='dataset', all_comp_pos_acc =False
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
            out_put : str,
            all_comp_pos_acc : boolean
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
    ds["uo"] = sum([ds[u] for u in list_u]).assign_attrs(units="m/s")
    ds["vo"] = sum([ds[v] for v in list_v]).assign_attrs(units="m/s")

    # transform time in actual dates
    t0 = pd.Timestamp("2000/01/01")
    ds["time_days"] = ds["time"].assign_attrs(units="days")
    ds["time"] = t0 + ds["time"] * pd.Timedelta("1D")

    # compute xy, uv and axy in fourier space
    # dataframe
    df = dataset2dataframe(ds)
    dt = (df.index[1] - df.index[0])/pd.Timedelta('1s')
    
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
    #auav COMPARISON WITH OTHER COMPUTATION WAYS SHOWS THAT WE SHOULD NOT COMPUTE ACCELERATION FROM UV
    #df['au'] = df.groupby('draw')['u'].apply(pyn.geo.spectral_diff, dt, order = 1)
    #df['av'] = df.groupby('draw')['v'].apply(pyn.geo.spectral_diff, dt, order = 1)
    #df['Auv'] = np.sqrt(df['ax']**2+df['ay']**2)

    df = df.groupby('draw').apply(pyn.geo.compute_dt, time='index')
    
    if all_comp_pos_acc:
        pos_vel_acc(df, dt, suf = '_low')
        pos_vel_acc(df, dt, suf = '_ni')
        pos_vel_acc(df, dt, suf = '_2')
        pos_vel_acc(df, dt, suf = '_1')
    
    if out_put == 'dataset' : 
        df = df.reset_index().set_index(['time', 'draw'])
        ds = df.to_xarray()
        return ds
    else : 
        return df
    return ds

def add_position_noise(ds, t, position_noise, ntype='white_noise', update_vel_acc = False, inplace=False):

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
        typical_dt = pd.Timedelta("300s")
        DT = (pd.read_csv(path_dt)["dt"] * pd.Timedelta("1s")).values
        ds["time_off"] = time_from_dt_array(ds.time.min(), ds.time.max(), DT, istart)

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
                             method_diff = 'pyn',
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
    """    
    #NOISE
    ds = add_position_noise(ds, t, position_noise, ntype, update_vel_acc)
    #IRREGULAR TIME SAMPLING
    ds = irregular_time_sampling(ds, offset_type, t, dt, file, istart, time_uniform=time_uniform)
    
    #MERGE
    ds['x'] = ds['x']+ds['x_noise']
    ds['y'] = ds['y']+ds['y_noise']
    
    # COMPUTE VELOCITIES/ACCELERATIONS (default is by spectral methods)
    ds = add_velocity_acceleration(ds, method = method_diff)
    return ds

def add_velocity_acceleration(ds, suffix="", method ='pyn', groupby = 'draw'):
    ds = ds.copy()
    # note: DataArray.differentiate: Differentiate the array with the second order accurate central differences.
    if method == 'spectral' :
        dt = ds.time.diff('time')/pd.Timedelta('1s')
        assert (dt[1:] == dt[1]).all(), 'time must be regularly sampled to apply spectral method'
        df = dataset2dataframe(ds)
        df['X'+ suffix] = np.sqrt(df['x'+ suffix]**2+df['y'+ suffix]**2)
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
        ds['X'+ suffix] = np.sqrt(ds['x'+ suffix]**2+ds['y'+ suffix]**2)
        ds["u" + suffix] = ds['x'].differentiate("time", datetime_unit="s")
        ds["v" + suffix] = ds['y'].differentiate("time", datetime_unit="s")
        ds['U'+ suffix] = np.sqrt(ds['u'+ suffix]**2+ds['v'+ suffix]**2)
        ds["au" + suffix] = ds["u" + suffix].differentiate("time", datetime_unit="s")
        ds["av" + suffix] = ds["v" + suffix].differentiate("time", datetime_unit="s")
        ds['Auv'+ suffix] = np.sqrt(ds['au'+ suffix]**2+ds['av'+ suffix]**2)
        return ds
    elif method =='pyn':
        df = dataset2dataframe(ds)
        df['X'+ suffix] = np.sqrt(df['x'+ suffix]**2+df['y'+ suffix]**2)
        
        df = df.groupby(groupby).apply(pyn.geo.compute_velocities, 'index', ("u" + suffix, "v" + suffix, "U" + suffix), centered = True,fill_startend = True,  distance ='xy')
        
        df = df.groupby(groupby).apply(pyn.geo.compute_accelerations, from_ = ('xy', 'x'+suffix, 'y'+suffix), names = ('ax' + suffix, 'ay' + suffix, 'Axy' + suffix))
        
        df = df.groupby(groupby).apply(pyn.geo.compute_accelerations, from_ = ('velocities', 'u'+suffix, 'v'+suffix), names = ('au' + suffix, 'av' + suffix, 'Auv' + suffix))
        
        df = df.reset_index().set_index(['time', 'draw'])
        ds = df.to_xarray()
        return ds
    else : 
        assert False, "method smust be 'xr' or 'spectral'"
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


# SPECTRA ##############################################################
def spectrum_df(df, nperseg=24*5):
    if 'gap_mask' in df and len(df[df.gap_mask ==1] )!= 0 :
        print('WARNING : INTERPOLATED HOLES')
    ds = df.reset_index().set_index(['time', 'id']).to_xarray()
    vc = [("x", "y"), ("u", "v"), ("ax","ay")]
    E = []
    for tup in vc :
        ds_ = ds.ts.spectrum(unit = '1D',nperseg=nperseg, complex=tup)
        E.append(ds_)
    return xr.merge(E)

def test_regular_dt(df):
    if df.index.name == 'time' : 
        return np.all(np.round(df.dt.dropna().values) == np.round((df.index[3]-df.index[2])/pd.Timedelta('1s')))
    else : 
        return np.all(np.round(df.dt.dropna().values) == np.round((df.time[3]-df.time[2])/pd.Timedelta('1s')))

def spectrum_DF(DF, nperseg=24*10) :
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
    ds = xr.concat(DSE.values(), dim =list(DSE.keys()))
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

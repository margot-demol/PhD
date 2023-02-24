import numpy as np
import pandas as pd
import xarray as xr
import dask.dataframe as dd

import drifters.utils as ut
import pynsitu as pin
from sstats import signals as sg
from sstats import sigp as sigp
from sstats import tseries as ts

"""
DIRECTORIES AND FILES
---------------------------------------------------------------------------------------------
"""
root_dir = "/home1/datawork/mdemol/GDP"

"""
VELOCITIES+ACCELERATIONS PARQUET FILES
--------
parquet_velocity_acceleration.ipynb

generates files containing relevants data + velocities and accelerations : 
columns =['time', 'id', 'lon', 'lat', 'vex', 'vny', 'vxy', 've', 'vn', 'ae', 'an',
       'aen', 'vex_diff', 'vny_diff', 'vxy_diff', 'aex', 'any', 'axy', 'x',
       'y', 'typebuoy', 'gap', 'deploy_date', 'deploy_lat', 'deploy_lon',
       'end_date', 'end_lat', 'end_lon', 'drogue_lost_date', 'typedeath',
       'lon360', 'err_lat', 'err_lon', 'err_ve', 'err_vn']
"""
gps_av = "/home1/datawork/mdemol/GDP/gps_av_time.parquet"
argos_av = "/home1/datawork/mdemol/GDP/argos_av_time.parquet"


"""

--------
containing relevants data + velocities and accelerations : 
columns =['time', 'id', 'lon', 'lat', 'vex', 'vny', 'vxy', 've', 'vn', 'ae', 'an',
       'aen', 'vex_diff', 'vny_diff', 'vxy_diff', 'aex', 'any', 'axy', 'x',
       'y', 'typebuoy', 'gap', 'deploy_date', 'deploy_lat', 'deploy_lon',
       'end_date', 'end_lat', 'end_lon', 'drogue_lost_date', 'typedeath',
       'lon360', 'err_lat', 'err_lon', 'err_ve', 'err_vn']
"""

"""
GENERATE NOISED TRAJECTORIES
"""

"""
def white_noise_time_series(t, noise_std, lon_ref=0.0, lat_ref= 45.0, add_to = 'lonlat'):
    draw = 2  # x, y
    da = ts.normal(time=t, draws=draw) * noise_std
    distance = 'geoid'
    
    if add_to == 'lonlat' :
        nlon = da.isel(draw=0).rename("nlon").drop("draw")
        nlat = da.isel(draw=1).rename("nlat").drop("draw")
        nlon[0] = 0  # centering
        nlat[0] = 0
        lon = (nlon / np.cos(np.pi/180*lat_ref) + lon_ref).rename("lon")
        lat = (nlat + lat_ref).rename("lat")
        ds = xr.merge([nlon, nlat, lon, lat])
    
    if add_to != 'lonlat' : 
        lon = xr.ones_like(da.isel(draw=0).drop("draw")).rename("lon")*lon_ref
        lat = xr.ones_like(da.isel(draw=0).drop("draw")).rename("lat")*lat_ref
        ds = xr.merge([lon, lat])
    
    ds["noise_std"] = noise_std
    ds["id"] = 0
    #ds.attrs = {"description": f"white noise with std={noise_std} on {add_to}"}
    df = ds.to_dataframe()

    # add x, y , velocity and acceleration noise
    # INDEX TIME ?
    if not df.index.name == "time":
        warnings.warn("Are you sure time is the index ? ", UserWarning)
    # SORTED TIME ?
    if not df.index.is_monotonic_increasing:
        warnings.warn("time sorting dataframe", UserWarning)
        df.sort_index()

    _geo = pin.geo.GeoAccessor(df)
    
    if add_to == 'xy' :
        nx =  da.isel(draw=0).drop("draw")
        ny = da.isel(draw=1).drop("draw")
        nx[0] = 0  # centering
        ny[0] = 0
        _geo._obj['x'] = nx
        _geo._obj['y'] = ny
        distance = ''
    
    if add_to == 'v' : 
        
        vx = da.isel(draw=0).rename("vx")
        vy = da.isel(draw=1).rename("vy")
        _geo._obj['vx'] = vx
        _geo._obj['vy'] = vy
        _geo._obj['x'] = vx*0
        _geo._obj['y'] = vy*0
        
    if add_to != 'v':
        _geo.compute_velocities(
            centered=True,
            names=(
                "vx",
                "vy",
                "vxy",
            ),
            distance = distance,
            inplace=True,
        )
  
    _geo.compute_accelerations(
        names=("ax", "ay", "axy"),
        from_=("velocities", "vx", "vy"),
        centered_velocity=True,
        inplace=True,
    )

    _geo.compute_accelerations(
        names=("ax", "ay", "axy"),
        from_=("xy", "x", "y"),
        centered_velocity=True,
        inplace=True,
    )
    _geo._obj.attrs = {"description": f"white noise with std={noise_std} on {add_to}"}
    return _geo._obj
"""


def white_noise_time_series(t, noise_std, lon_ref=0.0, lat_ref=45.0, add_to="lonlat"):
    """
    Genererate dataframe containing the time series of a 'static' trajectories, where variations are purely due to white noise on positions or velocities
    Parameters:
    -----------
            t : pd.date_range
                time serie
            noise_std : float
                        std of the noise 
            lon_ref : float
                    longitude of the 'static' trajectorie (does not have much importance)
            lat_ref : float
                    latitude of the 'static' trajectorie (does not have much importance)
            add_to : str
                    "lonlat", or "xy", or "v", allow to choose on which variable the white noise should be added.
                    The other time series are computed by integrations or differentiations.
            
    """
    draw = 2  # x, y
    da = ts.normal(time=t, draws=draw) * noise_std
    distance = "geoid"

    if add_to == "lonlat":
        nlon = da.isel(draw=0).rename("nlon").drop("draw")
        nlat = da.isel(draw=1).rename("nlat").drop("draw")
        nlon[0] = 0  # centering
        nlat[0] = 0
        lon = (nlon / np.cos(np.pi / 180 * lat_ref) + lon_ref).rename("lon")
        lat = (nlat + lat_ref).rename("lat")
        ds = xr.merge([nlon, nlat, lon, lat])

    if add_to != "lonlat":
        lon = xr.ones_like(da.isel(draw=0).drop("draw")).rename("lon") * lon_ref
        lat = xr.ones_like(da.isel(draw=0).drop("draw")).rename("lat") * lat_ref
        ds = xr.merge([lon, lat])

    ds["noise_std"] = noise_std
    ds["id"] = 0
    # ds.attrs = {"description": f"white noise with std={noise_std} on {add_to}"}
    df = ds.to_dataframe()

    # add x, y , velocity and acceleration noise
    # INDEX TIME ?
    if not df.index.name == "time":
        warnings.warn("Are you sure time is the index ? ", UserWarning)
    # SORTED TIME ?
    if not df.index.is_monotonic_increasing:
        warnings.warn("time sorting dataframe", UserWarning)
        df.sort_index()

    _geo = pin.geo.GeoAccessor(df)

    if add_to == "xy":
        nx = da.isel(draw=0).drop("draw")
        ny = da.isel(draw=1).drop("draw")
        nx[0] = 0  # centering
        ny[0] = 0
        _geo._obj["x"] = nx
        _geo._obj["y"] = ny
        distance = ""

    if add_to == "v":
        vx = da.isel(draw=0).rename("vx")
        vy = da.isel(draw=1).rename("vy")
        _geo._obj["vx"] = vx
        _geo._obj["vy"] = vy
        _geo._obj["x"] = vx * 0
        _geo._obj["y"] = vy * 0

    if add_to != "v":
        _geo.compute_velocities(
            centered=True,
            names=(
                "vx",
                "vy",
                "vxy",
            ),
            distance=distance,
            inplace=True,
        )
        _geo.compute_velocities(
            centered=False,
            names=(
                "vx_unc",
                "vy_unc",
                "vxy_unc",
            ),
            distance=distance,
            inplace=True,
        )
    if add_to == "lonlat":
        _geo.compute_accelerations(
            names=("ax", "ay", "axy"),
            from_=("lon", "lon", "lat"),
            inplace=True,
        )
    if add_to == "xy":
        _geo.compute_accelerations(
            names=("ax", "ay", "axy"),
            from_=("xy", "x", "y"),
            inplace=True,
        )
    if add_to == "v":
        _geo.compute_accelerations(
            names=("ax", "ay", "axy"),
            from_=("velocities", "vx", "vy"),
            centered_velocity=True,
            inplace=True,
        )
    _geo._obj.attrs = {"description": f"white noise with std={noise_std} on {add_to}"}
    return _geo._obj


def process_uv(lon, lat, u, v, N, dt, **kwargs):
    """Wraps spectral calculation: add complex velocity
    Assumes the time series is regularly sampled

    Parameters:
    -----------
        u, v: pd.Series
            zonal, meridional index by time (in days)
        N: int,
            length of the spectrum
        dt: float
            Time sampling in days
        **kwargs:
            passed to mit_equinox.drifters.get_spectrum
    """
    if lon is None:
        uv = None
    else:
        uv = u + 1j * v
    return pin.tseries.get_spectrum(uv, N, dt=dt, **kwargs)


def noise_traj(
    noise_std=5e-4,
    T="60D",
    dt="1H",
    t_ref=pd.Timestamp(2000, 1, 1),
    t_size=1e6,
    lon_ref=0.0,
    lat_ref=45.0,
    add_to="lonlat",
):
    """
    Generate times series and spectra for a virtually still drifter with only noise on position
        noise_std : float
                    std of the noise in °
        T : str or Timedelta
                    lenght of the time window for spectra
        dt : str
                    time series time delta
        t_ref :     Time delta
                    time reference
        t_size :    int
                    lenght of time index
        lon_ref:    float
                    longitude of the virtual drifter
        lat_ref:    float
                    latitude of the virtual drifter
    """
    if type(T) == str:
        T = pd.Timedelta(T)
    time_unit = pd.Timedelta(dt)
    t = pd.date_range(t_ref, periods=t_size, freq=time_unit)

    df = white_noise_time_series(t, noise_std, lon_ref, lat_ref, add_to)
    attrs = df.attrs["description"]
    df = df.reset_index().rename(columns={"time": "date"})
    # add time in hours
    df["time"] = (df["date"] - t_ref) / time_unit
    df = dd.from_pandas(df, npartitions=2)

    N = int(T / time_unit)  # output size
    T = T / time_unit  # must be in the same units than time

    columns = [
        "lon",
        "lat",
    ]
    Columns = {
        "n": columns + ["x", "y"],
        "v_unc": columns + ["vx_unc", "vy_unc"],
        "v_n": columns + ["vx", "vy"],
        "a_n": columns + ["ax", "ay"],
    }

    # pin.drifters.
    group = tuple(df["id"].loc[0].values.compute())[0]
    dfg = df.groupby("id").get_group(group).compute()
    Df_chunked = {}
    for l in Columns:
        df_chunked = pin.drifters.time_window_processing(
            dfg,
            process_uv,
            Columns[l],
            T,
            N,
            id_label="id",
            dt=dt,
            geo=True,
        )
        # rename x/y
        df_chunked = df_chunked.rename(columns=dict(x="lon", y="lat"))

        Df_chunked[l] = df_chunked.drop(columns=["id", "lon", "lat"])

    D = []
    for l in Df_chunked:
        d = Df_chunked[l].mean(axis=0)
        d = (
            d.reindex(d.index.astype("float"))
            .to_xarray()
            .rename({"index": "frequency"})
            .rename(l)
            .sortby("frequency")
        )
        D.append(d)
    ds = xr.merge(D)
    ds.frequency.attrs = {"long_name": "frequency", "units": "cpd"}
    ds.attrs = {
        "lon": lon_ref,
        "lat": lat_ref,
        "noise_std": noise_std,
        "description": attrs,
    }
    return df, ds


"""
GENERATE NOISED TRAJECTORIES
"""


def psd_white_noise(freq, stdx, D=1):
    """Return the analytical psd of a white noise depending on std
    freq : array
        frequency on wich we want the PSD
    stdx : float, array
        std of the white noise
    D : number of dimension (x->1, x+iy->2)

    """
    return (xr.ones_like(freq) * stdx**2 / (freq.max() - freq.min())) * D


def psd_centered_der(psd, freq="frequency"):
    """Return the PSD of the centered derivative giving the PSD of a variable (take PSD of x return PSD of v for a centered derivation
    psd : dataarray
        contains psd
    freq: str or array
        the key of frequencies in dataarray or the frequency array
    """
    if isinstance(freq, str):
        dt = 1 / 24
        freq = psd[freq]
    else:
        dt = 1 / 24

    return np.sin(2 * np.pi * freq * dt) ** 2 / (dt * 86400) ** 2 * psd


def psd_uncentered_der(psd, freq="frequency"):
    """Return the PSD of the uncentered derivative giving the PSD of a variable (take PSD of x return PSD of v for a centered derivation
    psd : dataarray
        contains psd
    freq: str or array
        the key of frequencies in dataarray or the frequency array
    """
    if isinstance(freq, str):
        dt = 1 / 24
        freq = psd[freq]
    else:
        dt = 1 / 24

    return 2 * (1 - np.cos(2 * np.pi * freq * dt)) / (dt * 86400) ** 2 * psd


def var_centered_der(stdx, corx=None, dt=1 / 24, D=1, lagskey="lags"):
    """ 
    Return variance of alpha derivative computed by central differentiation giving the std of alpha
    
    Parameters:
    -----------
            stdx : float
                    std of alpha
            corx : darray
                    correlation function
            dt : differentiation step
            D : int
                degree: 1 for the derivative of alpha
                        2 for the derivative of alpha1 + i alpha2
    """
    if lagskey != "lags":
        corx = corx.rename({lagskey: "lags"})
    if corx is None:
        return stdx**2 / 2 / (dt * 86400) ** 2 * D
    else:
        assert False, "not implemented yet"
        return (stdx**2 - corx.sel(lags=2 * dt)) / (2 * (dt * 86400) ** 2) * D


def var_2uncentered_der(stdx, corx=None, dt=1 / 24, D=1, lagskey="lags"):
    """ 
    Return variance of alpha derivative computed by 2 uncentered differentiation giving the std of alpha
    
    Parameters:
    -----------
            stdx : float
                    std of alpha
            corx : darray
                    correlation function
            dt : differentiation step
            D : int
                degree: 1 for the derivative of alpha
                        2 for the derivative of alpha1 + i alpha2
    """
    if lagskey != "lags":
        corx = corx.rename({lagskey: "lags"})
    if corx is None:
        return 6 * stdx**2 / (dt * 86400) ** 4 * D
    else:
        assert False, "not implemented yet"
        return (
            (6 * stdx**2 - 8 * corx.sel(lags=dt) + 2 * corx.sel(lags=2 * dt))
            / (dt * 86400) ** 4
            * D
        )


"""
SELECT CYCLONIC/ANTICYCLONIC SPECTRA
"""


def negpos_spectra(ds, freqkey="frequency"):
    """Return two datasets with cyclonic/anticyclonic spectra"""
    ds_inv = ds.sortby(freqkey, ascending=False)
    dsneg = ds_inv.where(ds_inv[freqkey] <= 0, drop=True)
    dsneg[freqkey] = -dsneg[freqkey]
    dspos = ds.where(ds[freqkey] >= 0, drop=True)
    return dsneg, dspos

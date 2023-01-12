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


def white_noise_time_series(t, noise_std, lon_ref, lat_ref):
    draw = 2  # x, y
    da = ts.normal(time=t, draws=draw) * noise_std
    nlon = da.isel(draw=0).rename("nlon").drop("draw")
    nlat = da.isel(draw=1).rename("nlat").drop("draw")
    nlon[0] = 0  # centering
    nlat[0] = 0
    lon = (nlon / np.cos(lat_ref) + lon_ref).rename("lon")
    lat = (nlat + lat_ref).rename("lat")
    ds = xr.merge([nlon, nlat, lon, lat])
    ds["noise_std"] = noise_std
    ds["id"] = 0
    ds.attrs = {"description": f"white noise with std={noise_std}"}
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
    _geo.compute_velocities(
        centered=True,
        names=(
            "vx",
            "vy",
            "vxy",
        ),
        inplace=True,
    )
    _geo.compute_accelerations(
        names=("ax", "ay", "axy"),
        from_=("velocities", "vx", "vy"),
        centered_velocity=True,
        inplace=True,
    )

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

    df = white_noise_time_series(t, noise_std, lon_ref, lat_ref)
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
        "v_n": columns + ["vx", "vy"],
        "a_n": columns + ["ax", "ay"],
    }

    # pin.drifters.
    group = tuple(df.get_partition(0)["id"].loc[0].values.compute())[0]
    dfg = df.groupby("id").get_group(group).compute()
    meta = pin.drifters.time_window_processing(
        dfg, process_uv, columns + ["x", "y"], T, N, id_label="id", dt=dt, geo=True
    )

    Df_chunked = {}
    for l in Columns:
        df_chunked = (
            df.groupby("id")
            .apply(
                pin.drifters.time_window_processing,
                process_uv,
                Columns[l],
                T,
                N,
                id_label="id",
                dt=dt,
                geo=True,
                meta=meta,
            )
            .persist()
        )
        # rename x/y
        df_chunked = df_chunked.rename(columns=dict(x="lon", y="lat"))

        Df_chunked[l] = df_chunked.drop(columns=["id", "lon", "lat"])

    D = []
    for l in Df_chunked:
        d = Df_chunked[l].mean(axis=0).compute()
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
    ds.attrs = {"lon": lon_ref, "lat": lat_ref, "noise_std": noise_std}
    return df, ds

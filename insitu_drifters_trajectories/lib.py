import os
from glob import glob

import geopandas as gpd
from shapely.geometry import Polygon

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
import pynsitu as pyn

plt.rcParams['axes.linewidth'] = 0.2 #set the value globally

"""
PATH
---------------------------------------------
"""
platform ='local'

# images_dir = '/Users/mdemol/code/PhD/filtering/images'
if platform == 'datarmor' :
    data_dir='/home/datawork-lops-osi/aponte/cswot/drifters' # datarmor
    root_dir = '/home1/datahome/mdemol/PhD/insitu_drifters_trajectories' #datarmor
    images_dir = '/home1/datahome/mdemol/PhD/insitu_drifters_trajectories/images'#datarmor
if platform == 'local' :
    images_dir = "/Users/mdemol/ownCloud/PhD/images"  # local
    root_dir = "/Users/mdemol/code/PhD/insitu_drifters_trajectories"
    data_dir = "/Users/mdemol/DATA_DRIFTERS/drifters"  # local

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
    "melodi_eodyn",
]

color = {
    "carthe_cnr": "firebrick",
    "carthe_lops": "orange",
    "code_ogs": "green",
    "svp_ogs": "darkblue",
    "svp_scripps": "teal",
    "svp_shom": "lightblue",
    "svp_bcg": "blue",
    "spotter_lops": "yellow",
    "carthe_uwa": "coral",
    "melodi_eodyn":"hotpink",
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

cut_date = {'svp_scripps' : pd.to_datetime('2023-05-30 18:00:00'), 
            'svp_ogs' : pd.to_datetime('2023-05-25 15:00:00'),}
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
SWOT plots
"""
def load_swot_tracks(phase="calval", resolution=None, bbox=None, **kwargs):
    """Load SWOT tracks


    Parameters
    ----------
    phase: str, optional
        "calval" or "science"
    resolution: str, optional
        Specify resolution, for example "10s", default is "30s"
    """


    if platform == "datarmor":
        tracks_dir = "/home/datawork-lops-osi/equinox/misc/swot"
    else:
        tracks_dir = "/Users/mdemol/code/swot_tracks"
    #
    files = glob(os.path.join(tracks_dir, "*.shp"))
    files = [f for f in files if phase in f]
    if resolution is not None:
        files = [f for f in files if resolution in f]
    dfiles = {f.split("_")[-1].split(".")[0]: f for f in files}
    out = {key: gpd.read_file(f, **kwargs) for key, f in dfiles.items()}


    if bbox is None:
        return out


    central_lon = (bbox[0] + bbox[1]) * 0.5
    central_lat = (bbox[2] + bbox[3]) * 0.5


    polygon = Polygon(
        [
            (bbox[0], bbox[2]),
            (bbox[1], bbox[2]),
            (bbox[1], bbox[3]),
            (bbox[0], bbox[3]),
            (bbox[0], bbox[2]),
        ]
    )
    out = {key: gpd.clip(gdf, polygon) for key, gdf in out.items()}


    return out

def plot_swot_tracks(ax, bbox):
    tracks = load_swot_tracks(bbox=bbox)["swath"]
    swot_kwargs = dict(
        facecolor="0.7",
        edgecolor="white",
        alpha=0.2,
        zorder=-1,
    )
    #if isinstance(swot_tracks, dict):
    #    swot_kwargs.update(swot_tracks)
    proj = ax.projection
    crs_proj4 = proj.proj4_init
    ax.add_geometries(
        tracks.to_crs(crs_proj4)["geometry"],
        crs=proj,
        **swot_kwargs,
    )

"""
---------------------------------------------------------------------------------
MINOR TICKS SYMPLOG SCALE
https://stackoverflow.com/questions/20470892/how-to-place-minor-ticks-on-symlog-scale
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import Locator


class MinorSymLogLocator(Locator):
    """
    Dynamically find minor tick positions based on the positions of
    major ticks for a symlog scaling.
    """
    def __init__(self, linthresh):
        """
        Ticks will be placed between the major ticks.
        The placement is linear for x between -linthresh and linthresh,
        otherwise its logarithmically
        """
        self.linthresh = linthresh

    def __call__(self):
        'Return the locations of the ticks'
        majorlocs = self.axis.get_majorticklocs()

        # iterate through minor locs
        minorlocs = []

        # handle the lowest part
        for i in range(1, len(majorlocs)):
            majorstep = majorlocs[i] - majorlocs[i-1]
            if abs(majorlocs[i-1] + majorstep/2) < self.linthresh:
                ndivs = 10
            else:
                ndivs = 9
            minorstep = majorstep / ndivs
            locs = np.arange(majorlocs[i-1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)

        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError('Cannot get tick locations for a '
                                  '%s type.' % type(self))

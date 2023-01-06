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

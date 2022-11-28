# GDP analysis

## Generate files  

### Compute velocities and accelerations, store all select informations
parquet_velocity_acceleration.ipynb generates dataframe files containing :
‘‘‘ columns =['time', 'id', 'lon', 'lat', 'vex', 'vny', 'vxy', 've', 'vn', 'ae', 'an', 'aen', 'vex_diff', 'vny_diff', 'vxy_diff', 'aex', 'any', 'axy', 'x','y', 'typebuoy', 'gap', 'deploy_date', 'deploy_lat', 'deploy_lon','end_date', 'end_lat', 'end_lon', 'drogue_lost_date', 'typedeath','lon360', 'err_lat', 'err_lon', 'err_ve', 'err_vn'] 
‘‘‘
and store in parquet files :
gps_av_time.parquet and argos_av_time.parquet

- ve , vn velocities computed by Elipot et al. LOWESS method
- vex, ven velocities computed with geoid 
- vex_diff, vny_diff velocities computed by finite differenciation 
- vxy = $\sqrt(vex^2 + vny^2)

### Compute spectra
zarr_spectral_analysis.ipynb generates :
- spectra in ‘‘‘loctype_variable_spectra.parquet‘‘‘ with loctype is 'gps' or 'argos
- spectra binned spatially and store it in ‘‘‘loctype_geospectra_{2}.zarr‘‘‘ with dl the geographical bin in degree.
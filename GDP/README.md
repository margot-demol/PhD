# GDP Analysis

## Objectives

## Files Descriptions

### parquet_velocity_acceleration.ipynb
Computate velocity and acceleration by differentiation : 
- load Elipot GDP data
- replace -1.000000e34 by nan values
- compute velocity and acceleration with pynsitu.geo.compute_velocities and pynsitu.geo.compute_acceleration
- store all in parquet

### parquet_velocity_acceleration_xycorrected.ipynb
Generate parquet after removing aberant/problematic values in vxy and axy \\
cf diag_comparaison_va_hole.ipynb to see the problem
- replace vxy >...% velocity percentile by Nan
- replace axy >...% accelaration percentile by Nan
- expand holes with nan values
- replace vxy^2>...ven^2 and near holes values with nan values
- store it in parquet

### parquet_time_window_spectra.ipynb
Compute pectra for all trajectories taking time windows of 60 days, with no detrending and hann window by default. 
- For each time windows spectra, the mean positions is also computed. 
- store it in parquets
Spectra are the power density of $\alpha_{zonal}+i \alpha_{meridional}$ computed with pynsitu.drifters.time_window_processing and pynsitu.tseries.get_spectrum for $\alpha$ among the different positions, velocities or accelerations.


### parquet_gdp_raw.ipynb
Load raw GDP data, store as parquet and inspect it
- remove raw where positions are nan values
- put longitude in the -180-180 range
- correct id removing 3002340 preffixe

### zarr_geohist.ipynb
Compute and store in zarr geographical boxes histograms (2° bins by default)

### zarr_spectra_geobins.ipynb
Compute and store geographically bins spectra  
 - average spectra computed in parquet_time_window_spectra.ipynb on 2°bins using this mean positions
 - store it in one zarr

### zarr_spectra_nrjbins.ipynb
Compute spectra binned by energy (0.025 by default)
- Compute cinetic nrj on 60 days time window using pynsitu.drifters.time_window_processing
- average spectra computed in parquet_time_window_spectra.ipynb binning on cinetic nrj
- store it in one zarr

### zarr_spectral_noise.ipynb

### test_static_noised_traj.ipynb
Test functions (GDP_lib.noise_traj and GDP_lib.white_noise_time_series) that generate 'static' tarjectories (where variations are purely due to white noise on positions or velocities) and its spectra

### test_get_spectrum.ipynb
Test pynsitu.tseries.get_spectrum 
- test get_spectrum reaction to holes in time series, and the way it can interpolate it

### test_window_integration_variance.ipynb
Test the part of variance lost while making spectra with time window processing
- computing psd with fft.fft and an hann window -> variance = 3/8 true variance
- computind psd with scipy.signal.periodogram with hann window (used in get_spectrum) -> already corrected var = true var
- time windowing does not seem to loose variance for long time window compared to caracteritic period

### diag_globalhist.ipynb
Global hsitograms for positions, velocities and accelerations and gap + global mean, std, variance etc

### diag_error_analysis.ipynb
Elipot errors analysis
- histograms
- year dependence
- buoy type dependence
- buoy type dependence on time
- drogued or undrogued dependence
- gap dependence
- geographical repartition
- err_lon weird values

### diag_nrj_map.ipynb
Energy maps
- Computed via spectral integration
- Computed via mean squares
- Comparason vxy, ven

### diag_acceleration_map.ipynb
Variance of acceleration maps
- Computed via spectral integration
- Computed via mean squares

### diag_geohist.ipynb
Geographically bins histograms

### diag_before_after_correction.ipynb
Comparison of velocities and acceleration before and after correction (99.99% percentile for velocities, 90 for acceleration)
- Trajectories
- std/var
- Nrj map
- Spectra
- map of nan values

### diag_trajectories.ipynb
Inspect trajectories and compare Elipot and GDP raw

### diag_comparison_va.ipynb
Comparison velocities and acceleration form Elipot and differentiation =  en vs xy
- Trajectories
- std/var
- Dependance on xy
- Dependence on the gap

### diag_spectral_analysis.ipynb
### diag_der_func_spectra.ipynb

### diag_noise_fit.ipynb
Static noised trajectories theoritical fit \\
Verify the fit between theoritical psd (computed via GDP_lib functions psd_white_noise, psd_centered_der and  psd_uncentered_der) and the ones obtained simulating a 'static noised trajectories' via GDP_lib.noise_traj.

### diag_band_integration.ipynb
Spectral integration per bands 
- 0-0.5, 0.5,2.5, 2.5-
- for velocity and acceleration
- allow to give an estimation of noise

### diag_geo_band_integration.ipynb
Spectral integration per bands in geo 2°bins (0-0.5, 0.5,2.5, 2.5 cpd)
- Number of spectra per 2°bins
- spectra in one bin
- map of nrj per band
- map of pourcentage of total nrj per band
- histograms of nrj per band
- histograms of pourcentage of total nrj per band


### diag_find_pb_a.ipynb
### diag_find_pb_v.ipynb

### diag_whitenoise_integration.ipynb
Estimate the pourcentage of noise  by fitting a white noise spectra (on position)

### diag_geospectra.ipynb
Geographically binned spectra
- plot spectra for chosen geographical boxes

### diag_nrj_spectra.ipynb
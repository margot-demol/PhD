import xarray as xr
import numpy as np
from scipy.special import erfc, erf

"""
1-LAYER
_______
"""

### Theoritical PSD or correlation for exponentially correlated process
def PSDu_exp(omega, sigma, T):
    return 2*T*sigma**2/(1+T**2*omega**2)
def PSDu_exp_ds(ds):
    return PSDu_exp(ds.freq_time*2*np.pi,
            ds.attrs["sigma_u"],
            ds.T)

def PSDa_exp(omega, sigma, T):
    return 2*T*sigma**2/(1+T**2*omega**2)*(omega/86400)**2
def PSDa_exp_ds(ds):
    return PSDa_exp(ds.freq_time*2*np.pi,
            ds.attrs["sigma_u"],
            ds.T)
def Coru_exp(tau, T, sigma):
    return sigma**2*np.exp(-tau/T)

"""
n-LAYERS
_______
Theoritical PSD or correlation for n-layers (2.20) and (A6) Viggiano
CAUTION : omega=2pif
"""
#Velocity
def PSDu_ou(omega, sigma, T, tau_eta, n):
    #Formule 2.21 Viggiano
    ratio_T=T**2/(1+T**2*omega**2)
    ratio_eta = tau_eta**2/(1+tau_eta**2*omega**2)
    psd_unnormalized = ratio_T*ratio_eta**(n-1)
    qn = sigma**2/psd_unnormalized.integrate('freq_time')
    return qn*psd_unnormalized

def PSDu_ou_ds(ds) :
    if 'tau_eta' in list(ds.keys()):
        return PSDu_ou(ds.freq_time*2*np.pi,
                ds.attrs["sigma_u"],
                ds.T, 
                ds.tau_eta,
                ds.attrs["n_layers"])
    else :
        return PSDu_ou(ds.freq_time*2*np.pi,
                ds.attrs["sigma_u"],
                ds.T, 
                ds.attrs["tau_eta_days"],
                ds.attrs["n_layers"])



#ACCELERATION
def PSDa_ou(omega, sigma, T, tau_eta, n):
    return PSDu_ou(omega, sigma, T, tau_eta, n)*(omega/86400)**2
def PSDa_ou_ds(ds):
    return PSDu_ou_ds(ds)*(ds.freq_time*2*np.pi/86400)**2

#
def PSDu_ou_228(omega, sigma, T, tau_eta, n):
    #Formule 2.28 Viggiano
    qn=2*sigma**2*np.exp(-tau_eta**2/T**2)/(T*erfc(tau_eta/T))
    ratio_T=T**2/(1+T**2*omega**2)
    ratio = tau_eta**2*omega**2/(n-1)
    ratio_eta =(1/(1+ratio))**(n-1)
    return qn*ratio_T*ratio_eta

def PSDu_ou_228_ds(ds) :
    if 'tau_eta' in list(ds.keys()):
        return PSDu_ou_228(ds.freq_time*2*np.pi,
                ds.attrs["sigma_u"],
                ds.T, ds.tau_eta,
                ds.attrs["n_layers"])
    else : 
        return PSDu_ou_228(ds.freq_time*2*np.pi,
                ds.attrs["sigma_u"],
                ds.T, ds.attrs["tau_eta_days"],
                ds.attrs["n_layers"])

"""
INFINITY OF LAYERS
_______
Theoritical PSD or correlation for n->infinity (2.29) and (2.30) Viggiano
"""
def Coru_ou_inf(tau, sigma, T, tau_eta):
    #Formule 2.29 Viggiano
    ratio = sigma**2*np.exp(-abs(tau)/T)/(2*erfc(tau_eta/T))
    erf_minus = erf(abs(tau)/(2*tau_eta)-tau_eta/T)
    erfc_plus = erfc(abs(tau)/(2*tau_eta)+tau_eta/T)
    return ratio*(1+erf_minus+np.exp(2*abs(tau)/T)*erfc_plus)

def Coru_ou_inf_ds(ds) :
    return Coru_ou_inf(ds.lags,
            ds.attrs["sigma_u"],
            ds.T, ds.attrs["tau_eta_days"])
    

def Cora_ou_inf(tau, sigma, T, tau_eta):
    exp_minus = np.exp(-abs(tau)/T)
    exp_plus = np.exp(abs(tau)/T)
    exp2 = np.exp(-(tau**2/(4*tau_eta**2) + tau_eta**2/T**2))
    
    erf_minus = erf(abs(tau)/(2*tau_eta)-tau_eta/T)
    erfc_plus = erfc(abs(tau)/(2*tau_eta)+tau_eta/T)
    erfc1 = erfc(tau_eta/T)
    
    ratio =2*T/(tau_eta*np.sqrt(np.pi))

    return (sigma**2/(2*T**2*erfc1)
            *(ratio*exp2 - exp_minus*(1+erfc_plus)- exp_plus*erf_minus)
           )
def Cora_ou_inf_ds(ds):
    return Cora_ou_inf(ds.lags,
            ds.attrs["sigma_u"],
            ds.T, ds.attrs["tau_eta_days"])
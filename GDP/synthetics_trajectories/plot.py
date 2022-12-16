import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from sstats import get_cmap_colors
from cycler import cycler

def plot_timeseries(ds, tmax=100):
    colors = get_cmap_colors(ds.T.size, cmap="plasma")
    plt.rc('axes', prop_cycle=cycler(color=colors))

    fig, axes = plt.subplots(3,1, figsize=(5,10), sharex=True)

    ax = axes[0]
    ds["x"].where(ds.time < tmax).plot.line(x="time", ax=ax);
    ax.grid()
    ax.set_title("")
    ax.set_xlabel("")


    ax = axes[1]
    ds["u"].where(ds.time < tmax).plot.line(x="time", ax=ax,add_legend=False);
    ax.grid()
    ax.set_title("")
    ax.set_xlabel("")

    ax = axes[2]
    ds["a"].where(ds.time < tmax).plot.line(x="time", ax=ax,add_legend=False);
    ax.grid()
    ax.set_title("")
    ax.set_xlabel("")
    
    fig.tight_layout()
    return fig, axes

def plot_autocorrelations(ds, tau_max=100):
    colors = get_cmap_colors(ds.T.size, cmap="plasma")
    plt.rc('axes', prop_cycle=cycler(color=colors))

    fig, axes = plt.subplots(1,3, figsize=(10,4), sharex=True)

    ax = axes[0]
    ds["xx"].where(ds.lags < tau_max).plot.line(x="lags", ax=ax);
    ax.grid()
    ax.set_title("")
    ax.set_xlabel("")


    ax = axes[1]
    if 'coru' in list(ds.keys()):
        ds.coru.where(ds.lags < tau_max).plot.line(x="lags",  color='k',ls=':',ax=ax, add_legend=False);
    ds["uu"].where(ds.lags < tau_max).plot.line(x="lags", ax=ax,add_legend=False);   
    ax.grid()
    ax.set_title("")
    ax.set_xlabel("")

    ax = axes[2]
    if 'cora' in list(ds.keys()):
        ds.cora.where(ds.lags < tau_max).plot.line(x="lags",  color='k',ls=':',ax=ax, add_legend=False);
    ds["aa"].where(ds.lags < tau_max).plot.line(x="lags", ax=ax,add_legend=False);
        
    ax.grid()
    ax.set_title("")
    ax.set_xlabel("")
    
    fig.tight_layout()

    return fig, axes

def plot_psd(ds, suffixes_ls ={'':'solid'}, suffixes_leg={'':''}, title='') : 
    colors = get_cmap_colors(ds.T.size, cmap="plasma")
    plt.rc('axes', prop_cycle=cycler(color=colors))
    fig, axs = plt.subplots(1,2, figsize=(9,5), sharex=True)#, sharey=True)
    def plot(ds,c):
        ax=axs[0]
        if 'PSDu' in list(ds.keys()):
            ds.PSDu.plot(color='k', ls=':', ax=ax)
        
        i=True
        for suf in suffixes_ls :  
            _E = ds['Eu'+suf].mean("draw")
            _Enorm = ds['Eu'].mean("draw").sel(freq_time=0)
            if i :
                _E.plot(x="freq_time", ax=ax, color=c, ls =suffixes_ls[suf], label=f"T={float(_T):.0f}d")
                i=False
            else : 
                _E.plot(x="freq_time", ax=ax, color=c, ls = suffixes_ls[suf])

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("f[cpd]")
        #ax.set_ylabel("Velocity PSD Eu")
        ax.grid()
        ax.set_title('(a)')
                
    
        ax=axs[1]
        if 'PSDu' in list(ds.keys()):
            ds.PSDa.plot(color='k',ls=':',ax=ax)
        for suf in suffixes_ls :  
            _E = ds['Ea'+suf].mean("draw")
            _Enorm = ds['Ea'].mean("draw").sel(freq_time=0)
            _E.plot(x="freq_time", ax=ax, color=c, ls = suffixes_ls[suf])
            
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("f[cpd]")
        #ax.set_ylabel("Acceleration PSD Ea")
        ax.set_title('(b)')

    if ds.T.size==1 : 
        _T=0
        plot(ds,colors[0],)
    else :
        for _T, c in zip(ds.T, colors):
            dsT = ds.sel(T=_T)
            plot(dsT,c)

    # Vertical limitation
    for ax in axs :
        if ds.T.size== 1: 
            Tl=np.array(ds.T)
        else:Tl=ds.T
        for i in range(Tl.size) : 
            if i==0 : ax.axvline(x = 1/Tl[i], color=colors[i], ls='--', label='1/T')
            else : ax.axvline(x = 1/ds.T[i], color=colors[i], ls='--')
        ax.axvline(1/ds.attrs['tau_eta_days'], color = 'k',ls ='--', label=r'$1/\tau_{\eta}$')
   
    axs[0].legend()
    axs[0].grid()
    axs[1].grid()
    fig.tight_layout(rect=[0,0,1,0.95])#left, bottom, right, top (default is 0,0,1,1)
    fig.suptitle(title)
    
    return fig, axs
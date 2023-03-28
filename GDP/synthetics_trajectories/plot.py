import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from sstats import get_cmap_colors
from cycler import cycler

# Loading the lowercase alphabet to a list
import string

letter = ["(" + i + ")" for i in list(string.ascii_lowercase)]

"""
def plot_timeseries(ds, tmax=100):
    colors = get_cmap_colors(ds.T.size, cmap="plasma")
    plt.rc('axes', prop_cycle=cycler(color=colors))

    fig, axes = plt.subplots(1,3, figsize=(12,4), sharex=True)

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
"""


def plot_timeseries(
    ds, tmax=100, suffixes_ls={"": "solid"}, suffixes_leg={"": ""}, title=""
):

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True)

    ax = axes[0]
    for suf in suffixes_ls:
        ds["x" + suf].where(ds.time < tmax).plot.line(
            x="time", ax=ax, ls=suffixes_ls[suf]
        )
    ax.grid()
    ax.set_title("")
    ax.set_ylabel("Position x [m]")
    ax.set_xlabel("Time [days]")

    ax = axes[1]
    for suf in suffixes_ls:
        ds["u" + suf].where(ds.time < tmax).plot.line(
            x="time", ax=ax, ls=suffixes_ls[suf], add_legend=False
        )
    ax.grid()
    ax.set_title("")
    ax.set_ylabel("Velocity u [m/s]")
    ax.set_xlabel("Time [days]")

    ax = axes[2]
    for suf in suffixes_ls:
        ds["a" + suf].where(ds.time < tmax).plot.line(
            x="time", ax=ax, ls=suffixes_ls[suf], add_legend=False
        )
    ax.grid()
    ax.set_title("")
    ax.set_ylabel(r"Acceleration a [$m^2$/s]")
    ax.set_xlabel("Time [days]")

    # if len(suffixes_ls)>1 :
    # leg = ''
    # for suf in suffixes_leg:
    # leg += suffixes_ls[suf] +' : ' + suffixes_leg[suf]+'\n'
    # fig.text(0.5, -0.01, leg, ha="center")#, fontsize=18)#, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

    # letters
    for i in range(3):
        axes[i].set_title(letter[i])

    fig.tight_layout(
        rect=[0, 0.05, 1, 0.95]
    )  # left, bottom, right, top (default is 0,0,1,1)
    fig.suptitle(title)
    return fig, axes


"""
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
    if 'coru_inf' in list(ds.keys()):
        ds.coru_inf.where(ds.lags < tau_max).plot.line(x="lags",  color='k',ls=':',ax=ax, add_legend=False);
    ds["uu"].where(ds.lags < tau_max).plot.line(x="lags", ax=ax,add_legend=False);   
    ax.grid()
    ax.set_title("")
    ax.set_xlabel("")

    ax = axes[2]
    if 'cora_inf' in list(ds.keys()):
        ds.cora_inf.where(ds.lags < tau_max).plot.line(x="lags",  color='k',ls=':',ax=ax, add_legend=False);
    ds["aa"].where(ds.lags < tau_max).plot.line(x="lags", ax=ax,add_legend=False);
        
    ax.grid()
    ax.set_title("")
    ax.set_xlabel("")
    
    fig.tight_layout()

    return fig, axes
"""


def plot_autocorrelations(
    ds, tau_max=100, suffixes_ls={"": "solid"}, suffixes_leg={"": ""}, title=""
):
    # Correlation are normalised by the variance
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True)

    ax = axes[0]

    for suf in suffixes_ls:
        ds["xx" + suf].where(ds.lags < tau_max).plot.line(
            x="lags", ax=ax, ls=suffixes_ls[suf]
        )
    ax.grid()
    ax.set_title("")
    ax.set_ylabel(r"$C_x$")
    ax.set_xlabel(r"$\tau$ [days]")

    ax = axes[1]
    if "coru" in list(ds.keys()):
        ds.coru.where(ds.lags < tau_max).plot.line(
            x="lags", color="k", ls=":", ax=ax, add_legend=False
        )
    if "coru_inf" in list(ds.keys()):
        ds.coru_inf.where(ds.lags < tau_max).plot.line(
            x="lags", color="k", ls=":", ax=ax, add_legend=False
        )
    for suf in suffixes_ls:
        ds["uu" + suf].where(ds.lags < tau_max).plot.line(
            x="lags", ax=ax, ls=suffixes_ls[suf], add_legend=False
        )
    ax.grid()
    ax.set_title("")
    ax.set_ylabel(r"$C_u$")
    ax.set_xlabel(r"$\tau$ [days]")

    ax = axes[2]
    if "cora_inf" in list(ds.keys()):
        ds.cora_inf.where(ds.lags < tau_max).plot.line(
            x="lags", color="k", ls=":", ax=ax, add_legend=False
        )
    for suf in suffixes_ls:
        ds["aa" + suf].where(ds.lags < tau_max).plot.line(
            x="lags", ax=ax, ls=suffixes_ls[suf], add_legend=False
        )
    ax.grid()
    ax.set_title("")
    ax.set_ylabel(r"$C_a$")
    ax.set_xlabel(r"$\tau$ [days]")

    # if len(suffixes_ls)>1 :
    # leg = ''
    # for suf in suffixes_leg:
    # leg += suffixes_ls [suf] +' : ' + suffixes_leg[suf]+'\n'
    # fig.text(0.5, -0.01, leg, ha="center")#, fontsize=18)#, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

    # letters
    for i in range(3):
        axes[i].set_title(letter[i])

    fig.tight_layout(
        rect=[0, 0.05, 1, 0.95]
    )  # left, bottom, right, top (default is 0,0,1,1)
    fig.suptitle(title)
    return fig, axes


def plot_psd(ds, suffixes_ls={"": "solid"}, suffixes_leg={"": ""}, title=""):
    colors = get_cmap_colors(ds.T.size, cmap="plasma")
    plt.rc("axes", prop_cycle=cycler(color=colors))
    fig, axs = plt.subplots(1, 2, figsize=(9, 5), sharex=True)  # , sharey=True)

    def plot(ds, c):
        ax = axs[0]
        if "PSDu" in list(ds.keys()):
            ds.PSDu.plot(color="k", ls=":", ax=ax)

        i = True
        for suf in suffixes_ls:
            _E = ds["Eu" + suf].mean("draw")
            _Enorm = ds["Eu"].mean("draw").sel(freq_time=0)
            if i:
                _E.plot(
                    x="freq_time",
                    ax=ax,
                    color=c,
                    ls=suffixes_ls[suf],
                    label=f"T={float(_T):.0f}d",
                )
                i = False
            else:
                _E.plot(x="freq_time", ax=ax, color=c, ls=suffixes_ls[suf])

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("f[cpd]")
        # ax.set_ylabel("Velocity PSD Eu")
        ax.set_ylabel(r"$PSD_u$ $[m^2/s^2/cpd]$")
        ax.grid()
        ax.set_title("(a)")

        ax = axs[1]
        if "PSDa" in list(ds.keys()):
            ds.PSDa.plot(color="k", ls=":", ax=ax)
        for suf in suffixes_ls:
            _E = ds["Ea" + suf].mean("draw")
            _Enorm = ds["Ea"].mean("draw").sel(freq_time=0)
            _E.plot(x="freq_time", ax=ax, color=c, ls=suffixes_ls[suf])

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("f[cpd]")
        # ax.set_ylabel("Acceleration PSD Ea")
        ax.set_title("(b)")
        ax.set_ylabel(r"$PSD_a$ $[m^2/s^4/cpd]$")

    if ds.T.size == 1:
        _T = 0
        plot(
            ds,
            colors[0],
        )
    else:
        for _T, c in zip(ds.T, colors):
            dsT = ds.sel(T=_T)
            plot(dsT, c)

    # Vertical limitation
    for ax in axs:
        if ds.T.size == 1:
            ax.axvline(x=1 / ds.T, color=colors[0], ls="--", label="1/T")
        else:
            for i in range(ds.T.size):
                if i == 0:
                    ax.axvline(x=1 / ds.T[i], color=colors[i], ls="--", label="1/T")
                else:
                    ax.axvline(x=1 / ds.T[i], color=colors[i], ls="--")
        if "tau_eta" in list(ds.coords):
            ax.axvline(1 / ds.tau_eta, color="k", ls="--", label=r"$1/\tau_{\eta}$")
        elif "tau_eta_days" in list(ds.attrs):
            ax.axvline(
                1 / ds.attrs["tau_eta_days"],
                color="k",
                ls="--",
                label=r"$1/\tau_{\eta}$",
            )

    axs[0].legend()
    axs[0].grid()
    axs[1].grid()
    fig.tight_layout(
        rect=[0, 0, 1, 0.95]
    )  # left, bottom, right, top (default is 0,0,1,1)
    fig.suptitle(title)

    return fig, axs

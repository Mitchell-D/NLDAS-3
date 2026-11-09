"""
"""
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
from pprint import pprint

from plotting import plot_scatter,get_listed_cmap,plot_nested_bars

if __name__=="__main__":
    res_json_path = Path(
        "/discover/nobackup/mtdodson/nldas3/" + \
        "comp_bench_1.json"
        )
    fig_dir = Path("/home/mtdodson/NLDAS-3/benchmarks/compression/figures")
    comp_read_scatter_path = fig_dir.joinpath(
        "nldas3_compression_netcdf_comp-vs-read.png"
        )
    exclude_vars = ["lon", "lat", "time"]

    """ ------------( end normal configuration )------------ """

    """
    to plot:

    - scatter read times vs compression ratio
    - bar plot compression ratios w/ minmax whiskers
      - group by compression level
    - bar plot write times
      - group by compression level
    """

    ## maps combo strings to dicts containing the parameters, dates for which
    ## that parameters are run, write times and compression ratios for each
    ## date, and read times
    results = json.load(res_json_path.open("r"))

    wtimes = []
    wtbounds = []
    rtimes = []
    rtbounds = []
    cratios = []
    crbounds = []
    css = []
    clvl = []
    for cs,cv in results.items():
        css.append(cs)
        clvl.append(cv["params"]["complevel"])
        rt = np.array([
            cv["read_times"][k] for k in cv["read_times"].keys()
            if not k in exclude_vars
            ]).ravel()
        rtimes.append(np.average(rt))
        rtbounds.append((
            rtimes[-1] - np.amin(rt),
            np.amax(rt) - rtimes[-1],
            ))
        cr = np.array(cv["comp_ratio"])
        cratios.append(np.average(cr))
        crbounds.append((
            cratios[-1] - np.amin(cr),
            np.amax(cr) - cratios[-1],
            ))
        wt = np.array(cv["write_time"])
        wtimes.append(np.average(wt))
        wtbounds.append((
            wtimes[-1] - np.amin(wt),
            np.amax(wt) - wtimes[-1],
            ))

    plot_scatter(
        x=cratios,
        y=rtimes,
        #size=np.array(sizes)/30000,
        #color=color,
        yerr=list(zip(*rtbounds)),
        xerr=list(zip(*crbounds)),
        labels=css,
        plot_spec={
            "title":"NLDAS-3 netCDF Compression vs Access Speed",
            "ylabel":"Load Throughput (B/s) and min/max",
            "xlabel":"Compression Ratio (default/comp)",
            #"xscale":"log",
            "point_label_fontsize":6,
            "point_label_rotation":-20,
            "tight_layout":True,
            "avoid_label_overlap":False,
            "errorbar_elinewidth":1,
            "errorbar_capsize":2,
            "errorbar_ecolor":"gray",
            "label_ha":"center",
            "label_va":"center",
            "dpi":200,
            "fig_size":(11,9),
            "use_colorbar":True,
            #"cbar_label":[
            #    "Mean Absolute Error", "Compression Category"
            #    ][isall],
            #"cmap":"plasma",
            "norm":"linear",
            },
        fig_path=comp_read_scatter_path,
        )
    print("finished")

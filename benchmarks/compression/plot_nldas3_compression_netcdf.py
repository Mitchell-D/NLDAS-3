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
    #res_json_path = Path(
    #    "/discover/nobackup/mtdodson/nldas3/" + \
    #    "comp_bench_1.json"
    #    )
    res_json_path = Path("data/comp_bench_1.json")
    #fig_dir = Path("/home/mtdodson/NLDAS-3/benchmarks/compression/figures")
    fig_dir = Path("figures")
    comp_read_scatter_path = fig_dir.joinpath(
        "nldas3_compression_netcdf_comp-vs-read.png"
        )
    comp_bar_path = fig_dir.joinpath(
        "nldas3_compression_netcdf_comp.png"
        )
    write_bar_path = fig_dir.joinpath(
        "nldas3_compression_netcdf_write.png"
        )
    read_bar_path = fig_dir.joinpath(
        "nldas3_compression_netcdf_read.png"
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
        p10,p90 = np.percentile(rt, [10, 90])
        rtbounds.append((
            rtimes[-1] - p10,
            p90 - rtimes[-1],
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

    #'''
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
            "ylabel":"Average array load time (s) and 10/90 pctl",
            "xlabel":"Compression Ratio (default/comp) with min/max",
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
    #'''

    bars_comp = {}
    bars_write = {}
    bars_read = {}
    for i,cs in enumerate(css):
        alglvl,*other = cs.split("_")
        other = "_".join(other)
        if not other in bars_comp.keys():
            bars_comp[other] = {}
            bars_write[other] = {}
            bars_read[other] = {}
        bars_comp[other][alglvl] = [cratios[i], crbounds[i][1]+crbounds[i][0]]
        bars_write[other][alglvl] = [wtimes[i], wtbounds[i][1]+wtbounds[i][0]]
        bars_read[other][alglvl] = [rtimes[i], rtbounds[i][1]+rtbounds[i][0]]
        #print(alglvl, other)

    bars_comp["none"] = bars_comp.pop("")
    bars_write["none"] = bars_write.pop("")
    bars_read["none"] = bars_read.pop("")

    plot_nested_bars(
        data_dict=bars_comp,
        labels={},
        plot_error_bars=True,
        bar_colors=None,
        plot_spec={
            "title":"NLDAS-3 netCDF compression ratios vs current settings",
            "cmap":"rainbow",
            "hline":1,
            "ylabel":"compression ratio (cur size / new size)",
            "hline_kwargs":{
                "zorder":-5,
                },
            },
        show=False,
        fig_path=comp_bar_path,
        group_order=None,
        bar_order=list(sorted(bars_comp[other].keys())),
        )

    plot_nested_bars(
        data_dict=bars_read,
        labels={},
        plot_error_bars=True,
        bar_colors=None,
        plot_spec={
            "title":"NLDAS-3 netCDF per-variable read times (10-90 pctl)",
            "cmap":"rainbow",
            "ylabel":"read time (seconds)",
            #"hline":1,
            "hline_kwargs":{
                "zorder":-5,
                },
            },
        show=True,
        fig_path=read_bar_path,
        group_order=None,
        bar_order=list(sorted(bars_comp[other].keys())),
        )

    plot_nested_bars(
        data_dict=bars_write,
        labels={},
        plot_error_bars=True,
        bar_colors=None,
        plot_spec={
            "title":"NLDAS-3 full netCDF write times",
            "cmap":"rainbow",
            "ylabel":"write time (seconds)",
            #"hline":1,
            "hline_kwargs":{
                "zorder":-5,
                },
            },
        show=False,
        fig_path=write_bar_path,
        group_order=None,
        bar_order=list(sorted(bars_comp[other].keys())),
        )
    print("finished")

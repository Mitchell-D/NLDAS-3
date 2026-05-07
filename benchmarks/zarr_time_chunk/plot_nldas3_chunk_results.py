import numpy as np
import json
import matplotlib.pyplot as plt
import netCDF4 as nc
from matplotlib.colors import ListedColormap
from pathlib import Path

from ChunkConfig import ChunkConfig

def plot_hists(counts:list, labels:list, bin_coords:np.array, normalize=False,
        line_colors:list=None, plot_spec:dict={}, show=False, fig_path=None):
    """
    Plot one or more histograms on a single pane

    :@param counts: List of 1D arrays representing the binned counts
    :@param labels: List of string labels corresponding to each histogram
    :@param bin_mins: List of 2-tuple (min, max) data coordinate values for
        each histogram. The minimum should be the minimum value of the first
        bin, and the maximum should be the upper value of the last bin.
    :@param plot_spec: Dict of configuration options for the plot
    """
    ps = {"xlabel":"", "ylabel":"", "linewidth":2, "text_size":12,
            "title":"", "dpi":80, "norm":None,"figsize":(12,12),
            "legend_ncols":1, "line_opacity":1, "cmap":"hsv",
            "label_fontsize":14, "title_fontsize":20, "legend_fontsize":14,
            "xscale":"linear", "yscale":"linear", "tick_fontsize":14,
            }
    ps.update(plot_spec)
    fig,ax = plt.subplots()
    cm = matplotlib.cm.get_cmap(ps.get("cmap"), len(counts))

    if not ps.get("hlines") is None:
        for hl in ps.get("hlines"):
            hlparams = {"color":"black", "linewidth":ps.get("linewidth")}
            if isinstance(hl, (list,tuple)) and len(hl)==2:
                hl,hlpnew = hl
                hlparams.update(hlpnew)
            ax.axhline(hl, **hlparams)
    if not ps.get("vlines") is None:
        for vl in ps.get("vlines"):
            vlparams = {"color":"black", "linewidth":ps.get("linewidth")}
            if isinstance(vl, (list,tuple)) and len(vl)==2:
                vl,vlpnew = vl
                vlparams.update(vlpnew)
            ax.axvline(vl, **vlparams)

    for i,(carr,label,bins) in enumerate(zip(counts,labels,bin_coords)):
        assert len(carr.shape) == 1, f"counts array must be 1D, {carr.shape=}"
        #bins = (np.arange(carr.size)+.5)/carr.size * (bmax-bmin) + bmin
        color = cm(i) if not line_colors else line_colors[i]
        if normalize:
            carr = carr / np.sum(carr)
        ax.plot(bins, carr, label=label, linewidth=ps.get("linewidth"),
                color=color, alpha=ps.get("line_opacity"))

    if not ps.get("ylim") is None:
        ax.set_ylim(*ps.get("ylim"))
    if not ps.get("xlim") is None:
        ax.set_xlim(*ps.get("xlim"))

    ax.set_xlabel(ps.get("xlabel"), fontsize=ps.get("label_fontsize"))
    ax.set_ylabel(ps.get("ylabel"), fontsize=ps.get("label_fontsize"))
    ax.set_title(ps.get("title"), fontsize=ps.get("title_fontsize"))
    ax.legend(ncol=ps.get("legend_ncols"), fontsize=ps.get("legend_fontsize"))
    ax.set_xscale(ps.get("xscale"))
    ax.set_yscale(ps.get("yscale"))
    ax.tick_params(axis="both",which="major",labelsize=ps.get("tick_fontsize"))
    ax.tick_params(axis="both",which="minor",labelsize=ps.get("tick_fontsize"))

    if show:
        plt.show()
    if fig_path:
        fig.set_size_inches(*ps.get("figsize"))
        fig.savefig(fig_path.as_posix(),bbox_inches="tight",dpi=ps.get("dpi"))
    plt.close()
    return

def plot_nested_bars(data_dict:dict, labels:dict={}, plot_error_bars=False,
        bar_colors:list=None, plot_spec:dict={}, show=False, fig_path=None,
        group_order:list=None, bar_order:list=None):
    """
    Plot a bar graph of metrics nested 2 levels deep, with optional error bars.

    :@param data_dict: Dict nested 2 layers deep, where the first layer
        identifies the bar grouping, and the second layer identifies the
        subcategory of a data point within each bar grouping. The second layer
        should map to a number if plot_error_bars is False, or a 2-tuple of
        [data, error_bar_magnitude] if plot_error_bars is True.
    :@param labels: dict of optional labels to replace data_dict keys in the
        legend or x-axis, if the data_dict key matches a labels key
    :@param plot_error_bars: Determines whether to expect a 2-tuple including
        error bar data per bar, as specified above
    :@param plot_spec: Dict of configuration options for the plot
    """
    ps = {"xlabel":"", "ylabel":"", "text_size":12, "title":"", "dpi":80,
            "figsize":(12,12), "legend_ncols":1, "line_opacity":1,
            "cmap":"hsv", "label_fontsize":14, "title_fontsize":20,
            "legend_fontsize":14, "bar_spacing":1}
    ps.update(plot_spec)
    fig,ax = plt.subplots()

    ## group keys
    gkeys = list(data_dict.keys()) if group_order is None else group_order
    ngroups = len(gkeys)
    group_starts = np.arange(ngroups)
    assert all(set(data_dict[k])==set(data_dict[gkeys[0]]) for k in gkeys[1:])
    ## bar keys
    bkeys = list(data_dict[gkeys[0]]) if bar_order is None else bar_order
    cm = matplotlib.cm.get_cmap(ps.get("cmap"), len(bkeys))

    bwidth = ps.get("bar_width", 1/(len(bkeys)+ps.get("bar_spacing")))

    bar_plots = []
    err_plots = []
    offset = 0
    for bix,bk in enumerate(bkeys):
        if plot_error_bars:
            tmp_data = [data_dict[gk][bk][0] for gk in gkeys]
            tmp_err = [data_dict[gk][bk][1] for gk in gkeys]
        else:
            tmp_data = [data_dict[gk][bk] for gk in gkeys]
            tmp_err = None
        bar_plots.append(ax.bar(
                group_starts + offset,
                tmp_data,
                color=cm(bix) if bar_colors is None else bar_colors[bix],
                width=bwidth,
                label=labels.get(bk,bk),
                ))
        if plot_error_bars:
            err_plots.append(ax.errorbar(
                    group_starts + offset,
                    tmp_data,
                    yerr=tmp_err,
                    fmt=ps.get("err_fmt","o"),
                    color=ps.get("err_color","black"),
                    ))
        offset += bwidth

    ax.set_xticks(
            group_starts+bwidth/2, [labels.get(gk,gk) for gk in gkeys],
            rotation=ps.get("xtick_rotation", 0),
            fontsize=ps.get("xtick_fontsize", ps.get("label_fontsize")),
            )
    plt.yticks(fontsize=ps.get("label_fontsize"))

    ax.set_xlabel(ps.get("xlabel"), fontsize=ps.get("label_fontsize"))
    ax.set_ylabel(ps.get("ylabel"), fontsize=ps.get("label_fontsize"))
    if not ps.get("ylim") is None:
        ax.set_ylim(*ps.get("ylim"))
    ax.set_title(ps.get("title"), fontsize=ps.get("title_fontsize"))
    ax.legend(ncol=ps.get("legend_ncols"), fontsize=ps.get("legend_fontsize"))

    if show:
        plt.show()
    if fig_path:
        fig.set_size_inches(*ps.get("figsize"))
        fig.savefig(fig_path.as_posix(),bbox_inches="tight",dpi=ps.get("dpi"))
    plt.close()
    return

def plot_scatter(x, y, size=None, color=None, xerr=None, yerr=None,
        labels=None, plot_spec={}, fig_path=None, show=False):
    """
    scatter plot with optional error bars and point labels, including optional
    label collision avoidance.
    """
    ps = {
        "xlabel":"", "ylabel":"", "marker_size":4, "dpi":200, "cmap":"jet",
        "text_size":12, "title":"", "norm":"linear", "marker":"o",
        "cbar_shrink":1., "map_linewidth":2, "title_fontsize":14,
        "legend_fontsize":14, "tick_fontsize":10, "legend_ncols":1,
        ## error bar defaults
        "errorbar_fmt":"none", "errorbar_ecolor":"black",
        "errorbar_elinewidth":1, "errorbar_capsize":2,
        ## label defaults
        "label_offset":(2,2), "label_fontsize":10, "label_color":"black",
        "label_ha":"right", "label_va":"bottom",
        ## collision avoidance
        "avoid_label_overlap":True, "adjust_text_expand":(1.05,1.2),
        "adjust_text_force":(0.1, 0.25),
        "adjust_text_arrowprops":{"arrowstyle":"-", "color":"gray", "lw":0.5},
        ## legend
        "use_point_legend":False, "legend_loc":"best",
        "legend_marker_scale":1.5,
        ## color bar
        "use_colorbar":False, "cbar_label":None, "cbar_label_fontsize":12,
        "cbar_tick_fontsize":10,
        }

    ps.update(plot_spec)
    plt.rcParams.update({"font.size": ps["text_size"]})

    fig, ax = plt.subplots(figsize=ps.get("fig_size"))

    sc = ax.scatter(
        x=x, y=y,
        s=size,
        c=color,
        marker=ps.get("marker"),
        cmap=ps.get("cmap"),
        vmin=ps.get("vmin"),
        vmax=ps.get("vmax"),
        norm=ps.get("norm"),
        linewidths=ps.get("linewidths"),
        )

    if ps.get("use_colorbar") and color is not None:
        try:
            ## check if numeric
            c_array = np.asarray(color)
            if np.issubdtype(c_array.dtype, np.number):
                cbar = fig.colorbar(sc, ax=ax, shrink=ps.get("cbar_shrink"))
                if ps.get("cbar_label"):
                    cbar.set_label(
                        ps.get("cbar_label"),
                        fontsize=ps.get("cbar_label_fontsize")
                        )
                cbar.ax.tick_params(
                    labelsize=ps.get("cbar_tick_fontsize")
                    )
            else:
                print("Colorbar skipped: 'color' is not numeric.")
        except Exception:
            print("Colorbar skipped: could not interpret 'color'.")

    ## optional error bars
    if xerr is not None or yerr is not None:
        ax.errorbar(
            x, y,
            xerr=xerr,
            yerr=yerr,
            fmt=ps.get("errorbar_fmt"),
            ecolor=ps.get("errorbar_ecolor"),
            elinewidth=ps.get("errorbar_elinewidth"),
            capsize=ps.get("errorbar_capsize"),
            zorder=0
            )

    texts = []

    ## optional labels
    if labels is not None:
        dx,dy = ps.get("label_offset")
        for xi,yi,lab in zip(x,y,labels):
            txt = ax.annotate(
                str(lab),
                (xi, yi),
                textcoords="offset points",
                xytext=(dx, dy),
                fontsize=ps.get("point_label_fontsize", 10),
                color=ps.get("label_color"),
                ha=ps.get("label_ha"),
                va=ps.get("label_va"),
                )
            texts.append(txt)

    # optional collision avoidance
    if ps.get("avoid_label_overlap") and texts:
        try:
            from adjustText import adjust_text

            adjust_text(
                texts,
                ax=ax,
                expand=ps.get("adjust_text_expand"),
                force_text=ps.get("adjust_text_force"),
                arrowprops=ps.get("adjust_text_arrowprops"),
                )
        except ImportError:
            print("adjustText not installed; skipping collision avoidance.")

    ## point legend
    if ps.get("use_point_legend") and labels is not None:
        handles = []
        seen = {}

        # normalize color handling
        if hasattr(color, "__len__"):
            colors = color
        else:
            colors = [color] * len(x)

        for xi, yi, lab, ci in zip(x, y, labels, colors):
            key = (lab, ci, ps.get("marker"))

            if key in seen:
                continue

            handle = Line2D(
                [0], [0],
                marker=ps.get("marker"),
                color="none",
                markerfacecolor=ci,
                markersize=ps.get("legend_marker_scale")*ps.get("marker_size"),
                label=str(lab)
                )

            handles.append(handle)
            seen[key] = True

        ax.legend(
            handles=handles,
            fontsize=ps.get("legend_fontsize"),
            ncol=ps.get("legend_ncols"),
            loc=ps.get("legend_loc")
            )

    ax.set_xlabel(ps.get("xlabel"), fontsize=ps.get("label_fontsize"))
    ax.set_ylabel(ps.get("ylabel"), fontsize=ps.get("label_fontsize"))
    ax.set_title(ps.get("title"), fontsize=ps.get("title_fontsize"))

    ax.tick_params(
        axis="both", which="major", labelsize=ps.get("tick_fontsize")
        )
    ax.tick_params(axis="both", which="minor",
                   labelsize=ps.get("tick_fontsize"))

    if show:
        plt.show()

    if fig_path:
        fig.savefig(
            fig_path.as_posix(),
            bbox_inches="tight",
            dpi=ps.get("dpi")
            )

    plt.close()

def get_listed_cmap(size, cmap="gist_rainbow", truncate_extremes=0):
    """ """
    ref_cmap = plt.get_cmap(cmap, size + 2*truncate_extremes)
    return ListedColormap([ref_cmap(i+truncate_extremes) for i in range(size)])

if __name__=="__main__":
    data_dir = Path("data")
    fig_dir = Path("figures")
    res_json_path = data_dir.joinpath("nldas3_chunk_bench_results.json")

    nldas3_param_path = data_dir.joinpath("nldas3_params.nc")
    results = json.load(res_json_path.open("r"))
    tlabels = results.keys() ## tests performed
    all_clabels = list(set(kl for tl in tlabels for kl in results[tl].keys()))

    plot_variables = ["Tair"]

    plot_StSxy_bitrate_scatter = True
    plot_land_per_chunk = True
    plot_pareto = True

    cl_cmap = get_listed_cmap(
            size=len(all_clabels),
            cmap="gist_rainbow",
            truncate_extremes=0,
            )
    cl_colors = {cl:cl_cmap(i) for i,cl in enumerate(all_clabels)}

    npz_dir_path = data_dir.joinpath("polys")
    int_fill = 999999999

    ## scatter plot St/Sxy vs time/point for each experiment
    if plot_StSxy_bitrate_scatter:
        for tl in tlabels:
            sratios,br_pct,dt_pct,ccs,sizes,dt_loads = [],[],[],[],[],[]
            clabels = list(sorted(results[tl].keys()))
            for cl in clabels:
                vstr,cstr = cl.split("-")
                if vstr not in plot_variables:
                    continue
                stime,slat,slon = tuple(map(int, cstr.split(".")))
                cc = ChunkConfig(ntime=stime, nlat=slat, nlon=slon)

                #sratios.append(stime / (slat*slon))
                sratios.append(stime / (slat*slon)**(1/2))
                sizes.append(stime*slat*slon*4/1000**2)
                pcount = np.array(results[tl][cl]["point_count"])
                dt_load = np.array(results[tl][cl]["dt_load"])
                br_pct.append(np.percentile(pcount/dt_load, [25, 50, 75]))
                dt_pct.append(np.percentile(dt_load, [25, 50, 75]))
                ccs.append(cc)

            br_p25,br_p50,br_p75 = map(np.asarray, zip(*br_pct))
            dt_p25,dt_p50,dt_p75 = map(np.asarray, zip(*dt_pct))

            plot_scatter(
                x=sratios,
                y=br_p50,
                #size=np.array(sizes)/30000,
                color=sizes,
                yerr=(br_p50-br_p25, br_p75-br_p50),
                labels=[cc.as_tuple() for cc in ccs],
                plot_spec={
                    "title":f"Download Efficiency per Chunk Aspect ({tl})" + \
                            "\nColored by Size in MB",
                    "ylabel":"Median Download (px/sec) (25-75 pct)",
                    "xlabel":"Timesteps per Area of a Chunk " + \
                            "[St Sxy^(-1/2)]",
                    "xscale":"log",
                    "point_label_fontsize":3,
                    "avoid_label_overlap":False,
                    "errorbar_elinewidth":1,
                    "errorbar_capsize":2,
                    "label_ha":"center",
                    "label_va":"center",
                    "use_colorbar":True,
                    "cbar_label":"Chunk Size (MB)",
                    "cmap":"rainbow",
                    "norm":"log",
                    },
                fig_path=fig_dir.joinpath(
                    f"chunk-bench_StSxy-bitrate_{tl}.png"),
                )

            plot_scatter(
                #x=np.log(sizes),
                x=sizes,
                #x=np.array(sizes)**(1/3),
                y=br_p50,
                #size=np.array(sizes)/30000,
                color=sratios,
                yerr=(br_p50-br_p25, br_p75-br_p50),
                labels=[cc.as_tuple() for cc in ccs],
                plot_spec={
                    "title":f"Download Efficiency wrt Chunk Size ({tl})" + \
                            "\nColored by Times per Area",
                    "ylabel":"Median Download (px/sec) (25-75pct)",
                    "xlabel":"Chunk Size (MB)",
                    "xscale":"log",
                    "point_label_fontsize":3,
                    "avoid_label_overlap":False,
                    "errorbar_elinewidth":1,
                    "errorbar_capsize":2,
                    "label_ha":"center",
                    "label_va":"center",
                    "use_colorbar":True,
                    "cbar_label":"Chunk St/Sxy",
                    "cmap":"rainbow",
                    #"norm":"log",
                    },
                fig_path=fig_dir.joinpath(
                    f"chunk-bench_csize-bitrate{tl}.png"),
                )

            plot_scatter(
                x=sizes,
                #x=np.array(sizes)**(1/3),
                y=dt_p50,
                #size=np.array(sizes)/30000,
                color=sratios,
                yerr=(dt_p50-dt_p25, dt_p75-dt_p50),
                labels=[cc.as_tuple() for cc in ccs],
                plot_spec={
                    "title":f"Download Time wrt Chunk Size ({tl})" + \
                            "\nColored by Times per Area",
                    "ylabel":"Median Download Time (sec) (25-75pct)",
                    "xlabel":"Chunk Size (MB)",
                    "xscale":"log",
                    "point_label_fontsize":3,
                    "avoid_label_overlap":False,
                    "errorbar_elinewidth":1,
                    "errorbar_capsize":2,
                    "label_ha":"center",
                    "label_va":"center",
                    "use_colorbar":True,
                    "cbar_label":"Chunk St/Sxy",
                    "cmap":"rainbow",
                    #"norm":"log",
                    },
                fig_path=fig_dir.joinpath(
                    f"chunk-bench_csize-dltime_{tl}.png"),
                )

    if plot_pareto:
        pdict = {}
        ## make a dict with all relevant collected results from each experiment
        for tl in tlabels:
            psubdict = {
                "sizes":[],
                "size_ratios":[],
                "bitrate_pct":[],
                "dltime_pct":[],
                "bitrate_full_pct":[],
                "ccs":[],
                }
            br_pct,br_full_pct,dt_pct = [],[],[]
            clabels = list(sorted(results[tl].keys()))
            for cl in clabels:
                vstr,cstr = cl.split("-")
                if vstr not in plot_variables:
                    continue
                stime,slat,slon = tuple(map(int, cstr.split(".")))

                psubdict["sizes"].append(stime*slat*slon*4/1000**2)
                psubdict["size_ratios"].append(stime / (slat*slon)**(1/2))
                pcount = np.array(results[tl][cl]["point_count"])
                dt_load = np.array(results[tl][cl]["dt_load"])
                cc = ChunkConfig(ntime=stime, nlat=slat, nlon=slon)
                psubdict["ccs"].append(cc)
                br_pct.append(np.percentile(pcount/dt_load, [25, 50, 75]))
                br_full_pct.append(np.percentile(
                    np.prod(cc.as_tuple())/dt_load,
                    [25, 50, 75]
                    ))
                dt_pct.append(np.percentile(dt_load, [25, 50, 75]))

            psubdict["bitrate_full_pct"] = list(map(
                np.asarray, zip(*br_full_pct)))
            psubdict["bitrate_pct"] = list(map(np.asarray, zip(*br_pct)))
            psubdict["dltime_pct"]  = list(map(np.asarray, zip(*dt_pct)))
            pdict[tl] = psubdict

        ## pixel vs timestep

        pres = pdict["pixel"]
        tres = pdict["timestep"]
        shared_ccs = [cc for cc in pres["ccs"] if cc in tres["ccs"]]

        pp25,pp50,pp75,csize = list(map(np.asarray, zip(*[
            (p25,p50,p75,float(np.prod(cc.as_tuple())))
            for p25,p50,p75,cc in sorted(
                zip(*pres["bitrate_full_pct"], pres["ccs"]),
                key=lambda t:t[-1].as_tuple(),
                )
            if cc in shared_ccs
            ])))
        tp25,tp50,tp75 = list(map(np.asarray, zip(*[
            (p25,p50,p75)
            for p25,p50,p75,cc in sorted(
                zip(*tres["bitrate_full_pct"], tres["ccs"]),
                key=lambda t:t[-1].as_tuple()
                )
            if cc in shared_ccs
            ])))

        plot_scatter(
            x=tp50,
            #x=np.array(sizes)**(1/3),
            y=pp50,
            #size=np.array(sizes)/30000,
            color=np.array(csize)*4/1000**2,
            xerr=(tp50-tp25, tp75-tp50),
            yerr=(pp50-pp25, pp75-pp50),
            labels=[cc.as_tuple() for cc in shared_ccs],
            plot_spec={
                "title":f"Chunk Efficiency wrt Area and Time\n" + \
                    "(assuming full chunk utilization)",
                "ylabel":"Efficiency Across Time (px/s) (25-50pct)",
                "xlabel":"Efficiency Across Space (px/s (25-50pct))",
                "xscale":"linear",
                "point_label_fontsize":3,
                "avoid_label_overlap":False,
                "errorbar_elinewidth":1,
                "errorbar_capsize":2,
                "label_ha":"center",
                "label_va":"center",
                "use_colorbar":True,
                "cbar_label":"Chunk Size (MB)",
                "cmap":"rainbow",
                "norm":"log",
                },
            fig_path=fig_dir.joinpath(
                f"chunk-bench_pareto-timestep-pixel_full-chunk.png"),
            )

        pp25,pp50,pp75,csize = list(map(np.asarray, zip(*[
            (p25,p50,p75,float(np.prod(cc.as_tuple())))
            for p25,p50,p75,cc in sorted(
                zip(*pres["bitrate_pct"], pres["ccs"]),
                key=lambda t:t[-1].as_tuple(),
                )
            if cc in shared_ccs
            ])))
        tp25,tp50,tp75 = list(map(np.asarray, zip(*[
            (p25,p50,p75)
            for p25,p50,p75,cc in sorted(
                zip(*tres["bitrate_pct"], tres["ccs"]),
                key=lambda t:t[-1].as_tuple()
                )
            if cc in shared_ccs
            ])))

        plot_scatter(
            x=tp50,
            #x=np.array(sizes)**(1/3),
            y=pp50,
            #size=np.array(sizes)/30000,
            color=np.array(csize)*4/1000**2,
            xerr=(tp50-tp25, tp75-tp50),
            yerr=(pp50-pp25, pp75-pp50),
            labels=[cc.as_tuple() for cc in shared_ccs],
            plot_spec={
                "title":f"Chunk Efficiency wrt Area and Time\n" + \
                    "(only count single timestep/pixel column)",
                "ylabel":"Efficiency Across Time (px/s) (25-50pct)",
                "xlabel":"Efficiency Across Space (px/s (25-50pct))",
                "xscale":"linear",
                "point_label_fontsize":3,
                "avoid_label_overlap":False,
                "errorbar_elinewidth":1,
                "errorbar_capsize":2,
                "label_ha":"center",
                "label_va":"center",
                "use_colorbar":True,
                "cbar_label":"Chunk Size (MB)",
                "cmap":"rainbow",
                "norm":"log",
                },
            fig_path=fig_dir.joinpath(
                f"chunk-bench_pareto-timestep-pixel_valid-pixels.png"),
            )

    if plot_land_per_chunk:
        ## download the parameter file if it doesn't exist already
        if not nldas3_param_path.exists():
            s3 = boto3.client("s3")
            s3.download_file(
                "nasa-waterinsight",
                "NLDAS3/static/NLDAS-3_dominant-soil-vegetation.nc",
                nldas3_param_path.as_posix(),
                )
        ## extract geo coords and land mask from the parameter file
        with nc.Dataset(nldas3_param_path, "r") as param_ds:
            nldas3_lats = param_ds["lat"][...]
            nldas3_lons = param_ds["lon"][...]
            ## class 14 corresponds to water
            nldas3_land_mask = ~(param_ds["Soiltype_inst"][...] == 14)

        ## Land pixels per chunk
        all_spatial_ccs = list(set([
            tuple(map(int, rk.split("-")[-1].split(".")))[1:]
            for tl in tlabels
            for rk in results[tl].keys()
            ]))
        poly_npz_paths = [
            (p,tuple(map(int, p.stem.split("_")[-1].split("-"))))
            for p in npz_dir_path.iterdir()
            if p.name.endswith(".npz")
            ]
        chunk_efficiency = {}
        for p,ct in filter(lambda pt:pt[1] in all_spatial_ccs, poly_npz_paths):
            chunks = np.load(p, allow_pickle=True)
            cinfo = chunks["chunk_info"]
            cmasks = np.where(nldas3_land_mask, chunks["chunk_masks"], int_fill)
            cunq = np.unique(cmasks, return_counts=True)
            #print(cunq)
            chunk_efficiency[ct] = np.delete(
                    cunq[-1],
                    int(np.where(cunq[0]==int_fill)[0][0])
                    )

        ctups,ceffs = zip(*chunk_efficiency.items())
        csizes = [cy*cx for cy,cx in ctups]
        eff_p25,eff_p50,eff_p75 = map(np.asarray, zip(*[
                np.percentile(a/s, [25, 50, 75])
                for a,s in zip(ceffs, csizes)
                ]))
        nchunks = [
            int(np.prod(ChunkConfig(1,cy,cx).chunk_layout(1,6500,11700)))
            for cy,cx in ctups
            ]
        layout_eff = [
                chunk_efficiency[(cy,cx)].size / n
                for (cy,cx),n in zip(ctups, nchunks)
                ]


        plot_scatter(
            x=csizes,
            #x=np.array(sizes)**(1/3),
            y=eff_p50,
            #size=np.array(sizes)/30000,
            color=layout_eff,
            yerr=(eff_p50-eff_p25, eff_p75-eff_p50),
            labels=ctups,
            plot_spec={
                "title":f"Spatial Efficiency of NLDAS-3 Chunk Shapes",
                "ylabel":"Median Land Pixels per Valid Chunk Pixel (25-75pct)",
                "xlabel":"Chunk Size (px)",
                "xscale":"log",
                "point_label_fontsize":5,
                "avoid_label_overlap":False,
                "errorbar_elinewidth":1,
                "errorbar_capsize":2,
                "label_ha":"center",
                "label_va":"center",
                "use_colorbar":True,
                "cbar_label":"Ratio of Chunks with Land Total Chunks",
                "cmap":"rainbow",
                #"norm":"log",
                },
            fig_path=fig_dir.joinpath(
                f"chunk-bench_csize-ceff.png"),
            )

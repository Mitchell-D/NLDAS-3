"""
Script for plotting benchmark results, mainly for 'pixel' and 'timestep'
experiments. I realized after running these tests that the results were
seriously affected by chunks that were smaller than the full size.
"""
import numpy as np
import json
import matplotlib.pyplot as plt
import netCDF4 as nc
from matplotlib.colors import ListedColormap
from pathlib import Path

from ChunkConfig import ChunkConfig
from plotting import plot_scatter,get_listed_cmap

if __name__=="__main__":
    data_dir = Path("data")
    fig_dir = Path("figures/full-chunks")
    #res_json_path = data_dir.joinpath("nldas3_chunk_bench_results.json")
    res_json_path = data_dir.joinpath(
        "nldas3_chunk_bench_results_fullchunk_1.json")
    run_str = "fc-1"

    nldas3_param_path = data_dir.joinpath("nldas3_params.nc")
    results = json.load(res_json_path.open("r"))
    tlabels = results.keys() ## tests performed
    all_clabels = list(set(kl for tl in tlabels for kl in results[tl].keys()))

    plot_variables = ["Tair"]

    plot_StSxy_bitrate_scatter = True
    plot_land_per_chunk = False
    plot_pareto = False
    plot_size_aspect = False
    plot_size_nchunks = False

    cl_cmap = get_listed_cmap(
            size=len(all_clabels),
            cmap="gist_rainbow",
            truncate_extremes=0,
            )
    cl_colors = {cl:cl_cmap(i) for i,cl in enumerate(all_clabels)}

    npz_dir_path = data_dir.joinpath("polys")
    int_fill = 999999999

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
                cc = ChunkConfig(stime, slat, slon)

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
                #labels=[cc.as_tuple() for cc in ccs],
                plot_spec={
                    "title":f"Download Efficiency wrt Chunk Aspect ({tl})" + \
                            "\nColored by Size in MB",
                    "ylabel":"Median Selected Pixels per Second (25-75 pct)",
                    "xlabel":"Timesteps per Area of a Chunk " + \
                            "[St Sxy^(-1/2)]",
                    "xscale":"log",
                    "point_label_fontsize":3.5,
                    "point_label_rotation":-30,
                    "tight_layout":True,
                    "avoid_label_overlap":False,
                    "errorbar_elinewidth":1,
                    "errorbar_capsize":2,
                    "errorbar_ecolor":"gray",
                    "label_ha":"center",
                    "label_va":"center",
                    "use_colorbar":True,
                    "cbar_label":"Chunk Size (log(size); MB)",
                    "cmap":"rainbow",
                    "norm":"log",
                    },
                fig_path=fig_dir.joinpath(
                    f"chunk-bench_StSxy-bitrate_{tl}_{run_str}.png"),
                )

            plot_scatter(
                #x=np.log(sizes),
                x=sizes,
                #x=np.array(sizes)**(1/3),
                y=br_p50,
                #size=np.array(sizes)/30000,
                color=sratios,
                yerr=(br_p50-br_p25, br_p75-br_p50),
                #labels=[cc.as_tuple() for cc in ccs],
                plot_spec={
                    "title":f"Download Efficiency wrt Chunk Size ({tl})" + \
                            "\nColored by Times per Area [St Sxy^(-1/2)]",
                    "ylabel":"Median Selected Pixels per Second (25-75pct)",
                    "xlabel":"Chunk Size (MB)",
                    "xscale":"log",
                    "point_label_fontsize":3.5,
                    "point_label_rotation":-30,
                    "tight_layout":True,
                    "avoid_label_overlap":False,
                    "errorbar_elinewidth":1,
                    "errorbar_capsize":2,
                    "errorbar_ecolor":"gray",
                    "label_ha":"center",
                    "label_va":"center",
                    "use_colorbar":True,
                    "cbar_label":"Chunk St/Sxy",
                    "cmap":"rainbow",
                    "ylim":{
                        "chunk":[0,1.2e7],
                        "multichunk":[0,1.2e7],
                        "volume":[0,6e7],
                        }.get(tl),
                    #"norm":"log",
                    },
                fig_path=fig_dir.joinpath(
                    f"chunk-bench_csize-bitrate_{tl}_{run_str}.png"),
                )

            plot_scatter(
                x=sizes,
                #x=np.array(sizes)**(1/3),
                y=dt_p50,
                #size=np.array(sizes)/30000,
                color=sratios,
                yerr=(dt_p50-dt_p25, dt_p75-dt_p50),
                #labels=[cc.as_tuple() for cc in ccs],
                plot_spec={
                    "title":f"Download Time wrt Chunk Size ({tl})" + \
                            "\nColored by Times per Area",
                    "ylabel":"Median Download Time (sec) (25-75pct)",
                    "xlabel":"Chunk Size (MB)",
                    "xscale":"log",
                    "point_label_fontsize":3.5,
                    "point_label_rotation":-45,
                    "tight_layout":True,
                    "avoid_label_overlap":False,
                    "errorbar_elinewidth":1,
                    "errorbar_capsize":2,
                    "errorbar_ecolor":"gray",
                    "label_ha":"center",
                    "label_va":"center",
                    "use_colorbar":True,
                    "cbar_label":"Chunk St/Sxy",
                    "cmap":"rainbow",
                    "ylim":{
                        "chunk":[0,4],
                        "multichunk":[0,125],
                        "volume":[0,80],
                        }.get(tl),
                    #"norm":"log",
                    },
                fig_path=fig_dir.joinpath(
                    f"chunk-bench_csize-dltime_{tl}_{run_str}.png"),
                )

    if plot_pareto or plot_size_aspect or plot_size_nchunks:
        pdict = {}
        ## make a dict with all relevant collected results from each experiment
        for tl in tlabels:
            psubdict = {
                "sizes":[],
                "size_ratios":[],
                "bitrate_pct":[],
                "dltime_pct":[],
                "point_count":[],
                "bitrate_full_pct":[],
                "unq_sizes":{},
                "ccs":[],
                "nchunks":[],
                "nchunks_partial":[],
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
                cc = ChunkConfig(stime, slat, slon)
                psubdict["ccs"].append(cc)
                psubdict["point_count"].append(np.sum(pcount))

                ## currently using ratio to simulate average
                if tl == "chunk":
                    nchunks = 1
                    nchunks_partial = 1
                elif tl == "pixel":
                    nchunks = cc.intersections(
                            grid_shape=(1826,1000,1800),
                            subset_shape=(1,1000,1800),
                            )[2]
                    nchunks_partial = 1826 / cc.cvec[0]
                elif tl == "timestep":
                    nchunks = cc.intersections(
                            grid_shape=(1826,1000,1800),
                            subset_shape=(1826,1,1),
                            )[2]
                    nchunks_partial = 1000 / cc.cvec[1] * 1800 / cc.cvec[2]
                elif tl == "multichunk":
                    nchunks = np.average([
                        v["nchunks"] for v in results[tl][cl]["test_kwargs"]
                        ])
                    nchunks_partial = nchunks
                elif tl == "volume":
                    nchunks = np.average([
                        np.array(v).shape[0] for v in results[tl][cl]["query"]
                        ])
                    nchunks_partial = nchunks
                else:
                    raise ValueError(f"test label not supported: {tl}")

                psubdict["nchunks"].append(nchunks)
                psubdict["nchunks_partial"].append(nchunks_partial)

                ## percentiles of bit rate
                br_pct.append(np.percentile(pcount/dt_load, [25, 50, 75]))
                br_full_pct.append(np.percentile(
                    nchunks_partial*np.prod(cc.as_tuple()) / dt_load,
                    [25, 50, 75]
                    ))
                ## percentiles of elapsed time
                dt_pct.append(np.percentile(dt_load, [25, 50, 75]))
                ## percentiles of elapsed time per payload size
                #for p,n in zip(*np.unique(pcount, return_counts=True)):
                #    print(p, n)

            psubdict["bitrate_full_pct"] = list(map(
                np.asarray, zip(*br_full_pct)))
            psubdict["bitrate_pct"] = list(map(np.asarray, zip(*br_pct)))
            psubdict["dltime_pct"]  = list(map(np.asarray, zip(*dt_pct)))
            pdict[tl] = psubdict

        ## pixel column vs timesteps pareto front

        ## percentiles for all chunk configs shared between
        shared_ccs = sorted([
            cc for cc in pdict["pixel"]["ccs"]
            if cc in pdict["timestep"]["ccs"] and cc in pdict["chunk"]["ccs"]
            ],key=lambda cc:cc.as_tuple())
        pp25,pp50,pp75,csize = list(map(np.asarray, zip(*[
            (p25,p50,p75,float(np.prod(cc.as_tuple())))
            for p25,p50,p75,cc in sorted(
                zip(*pdict["pixel"]["bitrate_full_pct"],pdict["pixel"]["ccs"]),
                key=lambda t:t[-1].as_tuple(),
                )
            if cc in shared_ccs
            ])))
        tp25,tp50,tp75 = list(map(np.asarray, zip(*[
            (p25,p50,p75)
            for p25,p50,p75,cc in sorted(
                zip(*pdict["timestep"]["bitrate_full_pct"],
                    pdict["timestep"]["ccs"]),
                key=lambda t:t[-1].as_tuple()
                )
            if cc in shared_ccs
            ])))
        cp25,cp50,cp75 = list(map(np.asarray, zip(*[
            (p25,p50,p75)
            for p25,p50,p75,cc in sorted(
                zip(*pdict["chunk"]["bitrate_full_pct"],pdict["chunk"]["ccs"]),
                key=lambda t:t[-1].as_tuple()
                )
            if cc in shared_ccs
            ])))
        mp25,mp50,mp75 = list(map(np.asarray, zip(*[
            (p25,p50,p75)
            for p25,p50,p75,cc in sorted(
                zip(*pdict["multichunk"]["bitrate_full_pct"],
                    pdict["multichunk"]["ccs"]),
                key=lambda t:t[-1].as_tuple()
                )
            if cc in shared_ccs
            ])))

        if plot_pareto:
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
                    "title":f"Chunk Throughput wrt Aspect\n" + \
                        "(assuming full chunk utilization)",
                    "ylabel":"Efficiency Across Time (px/s) (25-50pct)",
                    "xlabel":"Efficiency Across Space (px/s (25-50pct))",
                    "point_label_fontsize":3.5,
                    "point_label_rotation":-30,
                    "tight_layout":True,
                    "avoid_label_overlap":False,
                    "adjust_text_expand":(2.0,2.0),
                    "adjust_text_force":(4.05, 4.05),
                    "adjust_text_arrowprops":{
                        "arrowstyle":"-", "color":"gray", "lw":0.5},
                    "errorbar_elinewidth":1,
                    "errorbar_capsize":2,
                    "errorbar_ecolor":"gray",
                    "label_ha":"center",
                    "label_va":"center",
                    "use_colorbar":True,
                    "cbar_label":"Chunk Size (MB)",
                    "cmap":"rainbow",
                    "norm":"log",
                    },
                fig_path=fig_dir.joinpath(
                    f"chunk-bench_pareto-timestep-pixel" + \
                    f"_full-chunk_{run_str}.png"),
                )

        if plot_size_aspect:
            plot_scatter(
                x=[np.log10(cc.size*4/1000**2) for cc in shared_ccs], ## mb
                y=[cc.cvec[0]*np.prod(cc.cvec[1:3])**(-0.5)
                   for cc in shared_ccs],
                color=tp50 / 1e6,
                size=128,
                #size=pp50 / 500,
                #xerr=(tp50-tp25, tp75-tp50),
                #yerr=(pp50-pp25, pp75-pp50),
                labels=[cc.as_tuple() for cc in shared_ccs],
                plot_spec={
                    "title":f"Chunk Size and Aspect wrt Throughput (px/s)" + \
                        "\n(assumes full chunk utilization)",
                    "ylabel":"Chunk Aspect (St (SxSy)^-0.5)",
                    "xlabel":"Chunk log(size) in MB",
                    "point_label_fontsize":3.5,
                    "tight_layout":True,
                    "avoid_label_overlap":False,
                    "adjust_text_expand":(2.0,2.0),
                    "adjust_text_force":(4.05, 4.05),
                    "adjust_text_arrowprops":{
                        "arrowstyle":"-", "color":"gray", "lw":0.5},
                    "errorbar_elinewidth":1,
                    "errorbar_capsize":2,
                    "errorbar_ecolor":"gray",
                    "label_ha":"center",
                    "label_va":"center",
                    "use_colorbar":True,
                    #"cbar_label":"Spatial throughput (px/s)\n" + \
                    #    "Point size indicates time throughput",
                    "cbar_label":"Spatial throughput (1e6 px/s)",
                    "cbar_orient":"horizontal",
                    "cmap":"rainbow",
                    "vmin":0,
                    #"vmax":1,
                    },
                fig_path=fig_dir.joinpath(
                    f"chunk-bench_pareto-size-aspect_" + \
                    f"full-chunk_area_{run_str}.png"),
                )
            plot_scatter(
                x=[np.log10(cc.size*4/1000**2) for cc in shared_ccs], ## mb
                y=[cc.cvec[0]*np.prod(cc.cvec[1:3])**(-0.5)
                   for cc in shared_ccs],
                color=pp50 / 1e6,
                size=128,
                #size=tp50 / 500,
                #xerr=(tp50-tp25, tp75-tp50),
                #yerr=(pp50-pp25, pp75-pp50),
                labels=[cc.as_tuple() for cc in shared_ccs],
                plot_spec={
                    "title":f"Chunk Size and Aspect wrt Throughput (px/s)" + \
                        "\n(assumes full chunk utilization)",
                    "ylabel":"Chunk Aspect (St (SxSy)^-0.5)",
                    "xlabel":"Chunk log(size) in MB",
                    "point_label_fontsize":3.5,
                    "tight_layout":True,
                    "avoid_label_overlap":False,
                    "adjust_text_expand":(2.0,2.0),
                    "adjust_text_force":(4.05, 4.05),
                    "adjust_text_arrowprops":{
                        "arrowstyle":"-", "color":"gray", "lw":0.5},
                    "errorbar_elinewidth":1,
                    "errorbar_capsize":2,
                    "errorbar_ecolor":"gray",
                    "label_ha":"center",
                    "label_va":"center",
                    "use_colorbar":True,
                    "cbar_label":"Temporal throughput (1e6 px/s)",
                    "cbar_orient":"horizontal",
                    "cmap":"rainbow",
                    "vmin":0,
                    #"vmax":.3,
                    },
                fig_path=fig_dir.joinpath(
                    f"chunk-bench_pareto-size-aspect_full-chunk_" + \
                    f"time_{run_str}.png"),
                )
            plot_scatter(
                x=[np.log10(cc.size*4/1000**2) for cc in shared_ccs], ## mb
                y=[cc.cvec[0]*np.prod(cc.cvec[1:3])**(-0.5)
                   for cc in shared_ccs],
                color=cp50 / 1e6,
                size=128,
                #size=tp50 / 500,
                #xerr=(tp50-tp25, tp75-tp50),
                #yerr=(pp50-pp25, pp75-pp50),
                labels=[cc.as_tuple() for cc in shared_ccs],
                plot_spec={
                    "title":f"Chunk Size and Aspect wrt Throughput (px/s)" + \
                        "\n(assumes full chunk utilization)",
                    "ylabel":"Chunk Aspect (St (SxSy)^-0.5)",
                    "xlabel":"Chunk log(size) in MB",
                    "point_label_fontsize":3.5,
                    "tight_layout":True,
                    "avoid_label_overlap":False,
                    "adjust_text_expand":(2.0,2.0),
                    "adjust_text_force":(4.05, 4.05),
                    "adjust_text_arrowprops":{
                        "arrowstyle":"-", "color":"gray", "lw":0.5},
                    "errorbar_elinewidth":1,
                    "errorbar_capsize":2,
                    "errorbar_ecolor":"gray",
                    "label_ha":"center",
                    "label_va":"center",
                    "use_colorbar":True,
                    "cbar_label":"Chunk throughput (1e6 px/s)",
                    "cbar_orient":"horizontal",
                    "cmap":"rainbow",
                    "vmin":0,
                    #"vmax":1.,
                    },
                fig_path=fig_dir.joinpath(
                    f"chunk-bench_pareto-size-aspect_full-chunk_chunk"
                    f"_{run_str}.png"),
                )

        if plot_size_nchunks:
            all_p50 = np.concatenate([cp50, pp50, tp50, mp50], axis=0)
            nchunks = [
                *pdict["chunk"]["nchunks_partial"],
                *pdict["pixel"]["nchunks_partial"],
                *pdict["timestep"]["nchunks_partial"],
                *pdict["multichunk"]["nchunks_partial"],
                ]
            plot_scatter(
                x=[np.log10(cc.size*4/1000**2) for cc in shared_ccs]*4, ## mb
                y=np.log10(nchunks),
                color=all_p50 / 1e6,
                size=128,
                #size=pp50 / 500,
                #xerr=(tp50-tp25, tp75-tp50),
                #yerr=(pp50-pp25, pp75-pp50),
                labels=[cc.as_tuple() for cc in shared_ccs]*4,
                plot_spec={
                    "title":f"Chunk Size and Count wrt Throughput (px/s)" + \
                        "\n(assumes full chunk utilization)",
                    "ylabel":"Chunks log(count)",
                    "xlabel":"Chunk log(size) in MB",
                    "point_label_fontsize":3.5,
                    "point_label_rotation":30,
                    "tight_layout":True,
                    "avoid_label_overlap":False,
                    "adjust_text_expand":(2.0,2.0),
                    "adjust_text_force":(4.05, 4.05),
                    "adjust_text_arrowprops":{
                        "arrowstyle":"-", "color":"gray", "lw":0.5},
                    "errorbar_elinewidth":1,
                    "errorbar_capsize":2,
                    "errorbar_ecolor":"gray",
                    "label_ha":"center",
                    "label_va":"center",
                    "use_colorbar":True,
                    #"cbar_label":"Spatial throughput (px/s)\n" + \
                    #    "Point size indicates time throughput",
                    "cbar_label":"Spatial throughput (1e6 px/s)",
                    "cbar_orient":"horizontal",
                    "cmap":"rainbow",
                    "vmin":0,
                    #"vmax":1,
                    },
                fig_path=fig_dir.joinpath(
                    f"chunk-bench_pareto-size-nchunk_full-chunk_all" + \
                    f"_{run_str}.png"),
                )

        pp25,pp50,pp75,csize = list(map(np.asarray, zip(*[
            (p25,p50,p75,float(np.prod(cc.as_tuple())))
            for p25,p50,p75,cc in sorted(
                zip(*pdict["pixel"]["bitrate_pct"], pdict["pixel"]["ccs"]),
                key=lambda t:t[-1].as_tuple(),
                )
            if cc in shared_ccs
            ])))
        tp25,tp50,tp75 = list(map(np.asarray, zip(*[
            (p25,p50,p75)
            for p25,p50,p75,cc in sorted(
                zip(*pdict["timestep"]["bitrate_pct"],
                    pdict["timestep"]["ccs"]),
                key=lambda t:t[-1].as_tuple()
                )
            if cc in shared_ccs
            ])))

        if plot_pareto:
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
                    "title":f"Chunk Efficiency wrt Aspect\n" + \
                        "(only count single timestep/pixel column)",
                    "ylabel":"Efficiency Across Time (px/s) (25-50pct)",
                    "xlabel":"Efficiency Across Space (px/s (25-50pct))",
                    "xscale":"linear",
                    "point_label_fontsize":3.5,
                    "point_label_rotation":-30,
                    "tight_layout":True,
                    "avoid_label_overlap":False,
                    "errorbar_elinewidth":1,
                    "errorbar_capsize":2,
                    "errorbar_ecolor":"gray",
                    "label_ha":"center",
                    "label_va":"center",
                    "use_colorbar":True,
                    "cbar_label":"Chunk Size (log(size); MB)",
                    "cmap":"rainbow",
                    "norm":"log",
                    },
                fig_path=fig_dir.joinpath(
                    f"chunk-bench_pareto-timestep-pixel_" + \
                    f"valid-pixels_{run_str}.png"),
                )

        if plot_size_aspect:
            plot_scatter(
                x=[np.log10(cc.size*4/1000**2) for cc in shared_ccs],
                y=[cc.cvec[0]*np.prod(cc.cvec[1:3])**(-0.5)
                   for cc in shared_ccs],
                #color=pp50 / 1e6,
                color=pp50,
                #size=tp50 / 500,
                size=128,
                #xerr=(tp50-tp25, tp75-tp50),
                #yerr=(pp50-pp25, pp75-pp50),
                labels=[cc.as_tuple() for cc in shared_ccs],
                plot_spec={
                    "title":f"Chunk Size and Aspect wrt Throughput (px/s)" + \
                        "\n(only counts indexed pixels)",
                    "ylabel":"Chunk Aspect (St (SxSy)^-0.5)",
                    "xlabel":"Chunk log(size) in MB",
                    "xscale":"linear",
                    "point_label_fontsize":3.5,
                    "tight_layout":True,
                    "avoid_label_overlap":False,
                    "adjust_text_expand":(2.0,2.0),
                    "adjust_text_force":(4.05, 4.05),
                    "adjust_text_arrowprops":{
                        "arrowstyle":"-", "color":"gray", "lw":0.5},
                    "errorbar_elinewidth":1,
                    "errorbar_capsize":2,
                    "errorbar_ecolor":"gray",
                    "label_ha":"center",
                    "label_va":"center",
                    "use_colorbar":True,
                    "cbar_label":"Temporal throughput (px/s)",
                    #"cbar_label":"Temporal throughput (px/s)\n" + \
                    #    "Point size indicates area throughput",
                    "cbar_orient":"horizontal",
                    "cmap":"rainbow",
                    #"norm":"log",
                    },
                fig_path=fig_dir.joinpath(
                    f"chunk-bench_pareto-size-aspect_valid-pixels " + \
                    f"_area_{run_str}.png"),
                )
            plot_scatter(
                x=[np.log10(cc.size*4/1000**2) for cc in shared_ccs],
                y=[cc.cvec[0]*np.prod(cc.cvec[1:3])**(-0.5)
                   for cc in shared_ccs],
                color=tp50 / 1e6,
                #size=pp50 / 500,
                size=128,
                #xerr=(tp50-tp25, tp75-tp50),
                #yerr=(pp50-pp25, pp75-pp50),
                labels=[cc.as_tuple() for cc in shared_ccs],
                plot_spec={
                    "title":f"Chunk Size and Aspect wrt Throughput (px/s)" + \
                        "\n(only counts indexed pixels)",
                    "ylabel":"Chunk Size",
                    "xlabel":"Chunk Aspect",
                    "xscale":"linear",
                    "point_label_fontsize":3.5,
                    "tight_layout":True,
                    "avoid_label_overlap":False,
                    "adjust_text_expand":(2.0,2.0),
                    "adjust_text_force":(4.05, 4.05),
                    "adjust_text_arrowprops":{
                        "arrowstyle":"-", "color":"gray", "lw":0.5},
                    "errorbar_elinewidth":1,
                    "errorbar_capsize":2,
                    "errorbar_ecolor":"gray",
                    "label_ha":"center",
                    "label_va":"center",
                    "use_colorbar":True,
                    "cbar_label":"Area throughput (1e6 px/s)",
                    "cbar_orient":"horizontal",
                    "cmap":"rainbow",
                    #"norm":"log",
                    },
                fig_path=fig_dir.joinpath(
                    f"chunk-bench_pareto-size-aspect_valid-pixels" + \
                    f"_time_{run_str}.png"),
                )

    if plot_land_per_chunk:
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
        #for p,ct in filter(lambda pt:pt[1] in all_spatial_ccs, poly_npz_paths):
        for p,ct in poly_npz_paths:
            chunks = np.load(p, allow_pickle=True)
            cinfo = chunks["chunk_info"]
            cmasks = np.where(nldas3_land_mask,chunks["chunk_masks"],int_fill)
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
                "point_label_fontsize":3.5,
                "tight_layout":True,
                "avoid_label_overlap":False,
                "errorbar_elinewidth":1,
                "errorbar_capsize":2,
                "errorbar_ecolor":"gray",
                "label_ha":"center",
                "label_va":"center",
                "use_colorbar":True,
                "cbar_label":"Ratio of Chunks with Land Total Chunks",
                "cmap":"rainbow",
                #"norm":"log",
                },
            fig_path=fig_dir.joinpath(
                f"chunk-bench_csize-ceff_{run_str}.png"),
            )


"""
Script for plotting chunk benchmark results, mainly from full-chunk
'chunk', 'multichunk', and 'volume' experiments. This second version of
the script was created after I realized that it made a lot more sense to
guaruntee full chunks were being retrieved at a time.
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
        "nldas3_chunk_bench_results_fullchunk_3.json")
    run_str = "fc-3"

    compression_json_path = data_dir.joinpath(
            "nldas3_chunk_compression-ratio.json")

    subset_shape = (1826, 1000, 1800)
    dtype_size = 4 ## bytes

    nldas3_param_path = data_dir.joinpath("nldas3_params.nc")
    results = json.load(res_json_path.open("r"))
    tlabels = results.keys() ## tests performed
    all_clabels = list(set(kl for tl in tlabels for kl in results[tl].keys()))

    plot_variables = ["Tair"]

    plot_StSxy_bitrate_scatter = True
    plot_compression_ratio = True
    plot_throughput_wrt_volume = False

    ## get a color map with unique colors for each chunk configuration
    cl_cmap = get_listed_cmap(
            size=len(all_clabels),
            cmap="gist_rainbow",
            truncate_extremes=0,
            )
    cl_colors = {cl:cl_cmap(i) for i,cl in enumerate(all_clabels)}

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
                #color=sratios,
                color=[cc.cvec[0] for cc in ccs],
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
                    #"cbar_label":"Chunk St/Sxy",
                    "cbar_label":"Number of Times per Chunk",
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
                #color=sratios,
                color=[cc.cvec[0] for cc in ccs],
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
                    "cbar_label":"Number of Times per Chunk",
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
    if plot_compression_ratio:
        ccs,dsize = zip(*json.load(compression_json_path.open("r")).items())
        ccs = [tuple(map(int, cc.split(","))) for cc in ccs]
        sratios = [St/(Sx*Sy)**(1/2) for St,Sx,Sy in ccs]
        sts = [cc[0] for cc in ccs]
        plot_scatter(
            x=[np.prod(cc)*dtype_size / 1000**2 for cc in ccs],
            y=np.prod(subset_shape) * dtype_size / dsize,
            #color=sratios,
            color=sts,
            plot_spec={
                "title":"Chunk compression ratio wrt chunk size",
                "xlabel":"Chunk Size (MB)",
                "ylabel":"Compression Ratio (uncomp/comp)",
                "use_colorbar":True,
                #"cbar_label":"Chunk Aspect St (Sx Sy)^(-1/2)",
                "cbar_label":"Number of Time Steps per Chunk",
                "tight_layout":True,
                "ylim":[1.3, 1.55],
                },
            fig_path=Path("figures/chunk-bench_compression-ratio.png")
            )

    if plot_throughput_wrt_volume:
        for tl in tlabels:
            sratios,br_pct,dt_pct,ccs,sizes,dt_loads = [],[],[],[],[],[]
            clabels = list(sorted(results[tl].keys()))
            for cl in clabels:
                vstr,cstr = cl.split("-")
                if vstr not in plot_variables:
                    continue
                stime,slat,slon = tuple(map(int, cstr.split(".")))
                cc = ChunkConfig(stime, slat, slon)

    print(f"finished")

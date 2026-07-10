"""
"""
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
from pprint import pprint

from plotting import plot_scatter,get_listed_cmap

if __name__=="__main__":
    fig_dir = Path("figures_chunkshape/")
    #res_json_path = data_dir.joinpath("nldas3_chunk_bench_results.json")

    #plot_feats = ["Tair", "PSurf", "SWdown"]
    #plot_feats = ["Tair"]
    plot_feats = ["PSurf", "SWdown", "Rainf"]
    plot_regions = ["midwest", "desertw", "alaska", "mountainw"]
    #plot_chunks = ["32,325,450", "8,650,900", "64,325,225"]
    plot_chunks = ["32,325,450", "8,650,900", "64,325,225",
            "64,325,450", "64,650,450"]
    #plot_chunks = ["32,325,450", "64,650,450"]
    exclude_pipelines = [
        "dtype:f4_bitround:6_zstd:bitshuffle", ## error way too high
        "dtype:f4_bitround:6", ## no cr since no compression
        "dtype:f4_bitround:9", ## no cr since no compression
        "dtype:f4_bitround:10", ## no cr since no compression
        "dtype:f4_bitround:12", ## no cr since no compression
        "dtype:f2_lz4hc", ## way too much truncation error
        "dtype:f2_zstd", ## way too much truncation error

        ## essentially no difference from higher zstd compression level
        "dtype:f4_bitround:12_zstd:8,bitshuffle",
        "dtype:f4_bitround:11_zstd:8,bitshuffle",
        "dtype:f4_bitround:10_zstd:8,bitshuffle",
        "dtype:f4_bitround:9_zstd:8,bitshuffle",
        "dtype:f4_bitround:8_zstd:8,bitshuffle",
        "dtype:f4_bitround:6_zstd:8,bitshuffle",

        "dtype:f4_bitround:12_zfpy_zstd:8,bitshuffle",
        "dtype:f4_bitround:11_zfpy_zstd:8,bitshuffle",
        "dtype:f4_bitround:10_zfpy_zstd:8,bitshuffle",
        "dtype:f4_bitround:9_zfpy_zstd:8,bitshuffle",
        "dtype:f4_bitround:8_zfpy_zstd:8,bitshuffle",

        "dtype:f4_bitround:12_pcodec_zstd:8,bitshuffle",
        "dtype:f4_bitround:11_pcodec_zstd:8,bitshuffle",
        "dtype:f4_bitround:10_pcodec_zstd:8,bitshuffle",
        "dtype:f4_bitround:9_pcodec_zstd:8,bitshuffle",
        "dtype:f4_bitround:8_pcodec_zstd:8,bitshuffle",

        ## bizzarely high maximum error
        "dtype:f4_zfpy:0.05_zstd:bitshuffle",
        "dtype:f4_zfpy:0.01_zstd:bitshuffle",
        ]

    pipeline_groups = {
        "comp":[
            "dtype:f2_lz4hc",
            "dtype:f2_zstd",
            "dtype:f4_lz4hc",
            "dtype:f4_zstd",

            "dtype:f4_zstd:bitshuffle",
            "dtype:f4_zstd:8,bitshuffle",

            "dtype:f4_zfpy_zstd:bitshuffle",
            "dtype:f4_pcodec_zstd:bitshuffle",
            "dtype:f4_zfpy:0.05_zstd:bitshuffle",
            "dtype:f4_zfpy:0.01_zstd:bitshuffle",
            ],
        "bitround":[
            "dtype:f4_bitround:12",
            "dtype:f4_bitround:10",
            "dtype:f4_bitround:8",
            "dtype:f4_bitround:6",

            "dtype:f4_bitround:12_zstd:8,bitshuffle",
            "dtype:f4_bitround:11_zstd:8,bitshuffle",
            "dtype:f4_bitround:10_zstd:8,bitshuffle",
            "dtype:f4_bitround:9_zstd:8,bitshuffle",
            "dtype:f4_bitround:8_zstd:8,bitshuffle",
            "dtype:f4_bitround:6_zstd:8,bitshuffle",

            "dtype:f4_bitround:12_zfpy_zstd:8,bitshuffle",
            "dtype:f4_bitround:11_zfpy_zstd:8,bitshuffle",
            "dtype:f4_bitround:10_zfpy_zstd:8,bitshuffle",
            "dtype:f4_bitround:9_zfpy_zstd:8,bitshuffle",
            "dtype:f4_bitround:8_zfpy_zstd:8,bitshuffle",

            "dtype:f4_bitround:12_pcodec_zstd:8,bitshuffle",
            "dtype:f4_bitround:11_pcodec_zstd:8,bitshuffle",
            "dtype:f4_bitround:10_pcodec_zstd:8,bitshuffle",
            "dtype:f4_bitround:9_pcodec_zstd:8,bitshuffle",
            "dtype:f4_bitround:8_pcodec_zstd:8,bitshuffle",

            "dtype:f4_bitround:14_zstd:bitshuffle",
            "dtype:f4_bitround:13_zstd:bitshuffle",
            "dtype:f4_bitround:12_zstd:bitshuffle",
            "dtype:f4_bitround:11_zstd:bitshuffle",
            "dtype:f4_bitround:10_zstd:bitshuffle",
            "dtype:f4_bitround:9_zstd:bitshuffle",
            "dtype:f4_bitround:8_zstd:bitshuffle",
            "dtype:f4_bitround:6_zstd:bitshuffle",

            "dtype:f4_bitround:14_zfpy_zstd:bitshuffle",
            "dtype:f4_bitround:13_zfpy_zstd:bitshuffle",
            "dtype:f4_bitround:12_zfpy_zstd:bitshuffle",
            "dtype:f4_bitround:11_zfpy_zstd:bitshuffle",
            "dtype:f4_bitround:10_zfpy_zstd:bitshuffle",
            "dtype:f4_bitround:9_zfpy_zstd:bitshuffle",
            "dtype:f4_bitround:8_zfpy_zstd:bitshuffle",

            "dtype:f4_bitround:14_pcodec_zstd:bitshuffle",
            "dtype:f4_bitround:13_pcodec_zstd:bitshuffle",
            "dtype:f4_bitround:12_pcodec_zstd:bitshuffle",
            "dtype:f4_bitround:11_pcodec_zstd:bitshuffle",
            "dtype:f4_bitround:10_pcodec_zstd:bitshuffle",
            "dtype:f4_bitround:9_pcodec_zstd:bitshuffle",
            "dtype:f4_bitround:8_pcodec_zstd:bitshuffle",
            ],
        "norm-custom":[
            "dtype:f4_intnorm:custom",
            "dtype:f4_intnorm:custom_delta:u2,u2_zstd:5,bitshuffle",
            "dtype:f4_intnorm:custom_zstd:bitshuffle",
            "dtype:f4_intnorm:custom_zstd:shuffle",
            "dtype:f4_intnorm:custom,i4_zfpy_zstd:5,bitshuffle",
            "dtype:f4_intnorm:custom_pcodec_zstd:5,bitshuffle",
            "dtype:f4_intnorm:custom,u4_pcodec_zstd:5,bitshuffle",
            ],
        "norm-4096":[
            "dtype:f4_intnorm:4096",
            "dtype:f4_intnorm:4096_delta:u2,u2_zstd:5,bitshuffle",
            "dtype:f4_intnorm:4096_zstd",
            "dtype:f4_intnorm:4096_zstd:bitshuffle",
            "dtype:f4_intnorm:4096_zstd:shuffle",
            "dtype:f4_intnorm:4096,i4_zfpy_zstd:5,bitshuffle",
            "dtype:f4_intnorm:4096_pcodec_zstd:5,bitshuffle",
            "dtype:f4_intnorm:4096,u4_pcodec_zstd:5,bitshuffle",
            ],
        "norm-2048":[
            "dtype:f4_intnorm:2048",
            "dtype:f4_intnorm:2048_delta:u2,u2_zstd:5,bitshuffle",
            "dtype:f4_intnorm:2048_zstd:bitshuffle",
            "dtype:f4_intnorm:2048_zstd:shuffle",
            "dtype:f4_intnorm:2048,i4_zfpy_zstd:5,bitshuffle",
            "dtype:f4_intnorm:2048_pcodec_zstd:5,bitshuffle",
            "dtype:f4_intnorm:2048,u4_pcodec_zstd:5,bitshuffle"
            ],
        "bitround-13":[
            "dtype:f4_bitround:13_pcodec_zstd:bitshuffle",
            ],
        }
    pipeline_groups["norm-all"] = pipeline_groups["norm-custom"] \
        + pipeline_groups["norm-4096"] + pipeline_groups["norm-2048"]

    group_colors = {
        "comp":"#8da0cb", ## purple
        "bitround":"#fc8d62", ## orange
        #"norm-custom":"#66c2a5", ## cyan
        #"norm-4096":"#e78ac3", ## pink
        #"norm-2048":"#a6d854", ## lime
        "norm-custom":"#b2e2e2", ## lightgreen
        "norm-4096":"#66c2a4", ## midgreen
        "norm-2048":"#238b45", ## darkgreen
        }

    #plot_groups = ["all", "bitround", "comp"]
    plot_groups = ["bitround-13"]

    plot_scatter_cratio_rtime = True
    plot_scatter_cratio_error = True
    plot_bars_cratio = True
    plot_bars_rtime = True
    plot_bars_error = True

    tres = "daily"

    res_json_path = Path(f"data/results_compression_{tres}.json")

    """ ------------( end normal configuration )------------ """

    results = json.load(res_json_path.open("r"))
    combos = []
    for fk in results.keys():
        for pk in results[fk].keys():
            for rk in results[fk][pk].keys():
                for ct in results[fk][pk][rk].keys():
                    combos.append((fk,pk,rk,ct))
    cout = list(map(
        lambda t:sorted(list(set(t))),
        zip(*combos)
        ))
    all_feats,all_pipelines,all_regions,all_chunks = cout
    pipeline_groups["all"] = all_pipelines
    print(all_feats, all_pipelines, all_regions)

    for fk,pk,rk,ct in sorted(combos, key=lambda c:(c[0],c[2],c[1],c[3])):
        r = results[fk][pk][rk][ct]
        print(f"{r['total_size']/r['chunk_size']:>6.2f}",
            f"    {r['error_stats']['absmean']:>4.2f}    ",
            fk, rk, pk, ct)

    if plot_scatter_cratio_rtime:
        plot_combos = {}
        for fk in all_feats:
            if fk not in plot_feats:
                continue
            for gk in plot_groups:
                for pk in results[fk]:
                    if pk in exclude_pipelines:
                        continue
                    if pk not in pipeline_groups[gk]:
                        continue
                    for rk in results[fk][pk].keys():
                        if rk not in plot_regions:
                            continue
                        if (gk,fk,rk) not in plot_combos.keys():
                            plot_combos[(gk,fk,rk)] = []
                        for ct in results[fk][pk][rk].keys():
                            if ct not in plot_chunks:
                                continue
                            plot_combos[(gk,fk,rk)].append((pk, ct))

        for (gk,fk,rk),pks_cts in plot_combos.items():
            pdata = [results[fk][pk][rk][ct] for pk,ct in pks_cts]
            cr = np.array([d["total_size"]/d["chunk_size"] for d in pdata])
            pprint(pdata)
            if len(pdata) == 0:
                print("no data found for", gk, fk, rk)
                continue
            ## (pipeline, timstep) MB/s
            rt = [
                d["total_size"] / len(d["load_time"]) / 1000**2 \
                        / np.array(d["load_time"])
                for d in pdata
                ]
            print(rt)
            rt25,rt50,rt75 = map(np.squeeze, np.split(np.array([
                np.percentile(v, [25, 50, 75])
                for v in rt
                ]), 3, axis=-1))
            print(rt25.shape, rt50.shape, rt75.shape)
            rt = np.array([np.average(v) for v in rt])
            print(rt)
            #rt = np.array(rt)
            #rt25,rt50,rt75 = np.percentile(rt, [25,50,75], axis=-1)
            #rt = np.average(rt, axis=-1)

            ## color "all" group by inclusion in other groups
            isall = gk == "all"
            if isall:
                color = []
                for tpk,tct in pks_cts:
                    color_found = False
                    for tgk,tgv in pipeline_groups.items():
                        if tgk == "all" or not tpk in tgv:
                            continue
                        if tgk not in group_colors.keys():
                            print(f"no color for {tgk}; skipping")
                            continue
                        color.append(group_colors[tgk])
                        color_found = True
                        break
                    if not color_found:
                        raise ValueError(f"group or color not found for ", tpk)
            else:
                color = [p["error_stats"]["absmean"] for p in pdata]


            fpath = fig_dir.joinpath(
                    f"scatter_{gk}_{fk}_{rk}_{tres}_cratio-rtime.png")

            plot_scatter(
                x=cr,
                y=rt,
                #size=np.array(sizes)/30000,
                color=color,
                yerr=(rt50-rt25, rt75-rt50),
                labels=[f"{tpk} {tct}" for tpk,tct in pks_cts],
                plot_spec={
                    "title":"Compression vs Access Speed " + \
                            f"\n({tres} {fk} {rk} {gk})",
                    "ylabel":"Load Throughput (MB/s) w/ IQR",
                    "xlabel":"Compression Ratio (uncomp/comp)",
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
                    "cbar_label":[
                        "Mean Absolute Error", "Compression Category"
                        ][isall],
                    "cmap":"plasma",
                    "norm":"linear",
                    },
                fig_path=fpath,
                )
            print(f"Generated {fpath}")

    if plot_scatter_cratio_error:
        plot_combos = {}
        for fk in all_feats:
            if fk not in plot_feats:
                continue
            for gk in plot_groups:
                for pk in results[fk]:
                    if pk in exclude_pipelines:
                        continue
                    if pk not in pipeline_groups[gk]:
                        continue
                    for rk in results[fk][pk].keys():
                        if rk not in plot_regions:
                            continue
                        if (gk,fk,rk) not in plot_combos.keys():
                            plot_combos[(gk,fk,rk)] = []
                        for ct in results[fk][pk][rk].keys():
                            if ct not in plot_chunks:
                                continue
                            plot_combos[(gk,fk,rk)].append((pk, ct))

        for (gk,fk,rk),pks_cts in plot_combos.items():
            pdata = [results[fk][pk][rk][ct] for pk,ct in pks_cts]
            cr = np.array([d["total_size"]/d["chunk_size"] for d in pdata])
            er = np.array([d["error_stats"]["absmean"] for d in pdata])
            er_stdv = np.array([
                d["error_stats"]["absstdv"] / 2
                for d in pdata
                ])
            er_max = np.array([
                max(d["error_stats"]["max"], abs(d["error_stats"]["min"])) \
                        - d["error_stats"]["absmean"]
                for d in pdata
                ])
            rt = [
                d["total_size"]/1000**2 / \
                        np.array(d["load_time"])/len(d["load_time"])
                for d in pdata
                ]
            rt = np.array([np.average(v) for v in rt])

            ## color "all" group by inclusion in other groups
            isall = gk == "all"
            if isall:
                color = []
                for tpk,tct in pks_cts:
                    color_found = False
                    for tgk,tgv in pipeline_groups.items():
                        if tgk == "all" or not tpk in tgv:
                            continue
                        if tgk not in group_colors.keys():
                            print(f"no color for {tgk}; skipping")
                            continue
                        color.append(group_colors[tgk])
                        color_found = True
                        break
                    if not color_found:
                        raise ValueError(f"group or color not found for ", tpk)
            else:
                color = rt

            fpath = fig_dir.joinpath(
                    f"scatter_{gk}_{fk}_{rk}_{tres}_cratio-error.png")
            plot_scatter(
                x=cr,
                y=er,
                #size=np.array(sizes)/30000,
                color=color,
                yerr=((0,)*er_max.size, er_max),
                labels=[f"{tpk} {tct}" for tpk,tct in pks_cts],
                plot_spec={
                    "title":"Compression vs Trunc Error " + \
                            f"\n({tres} {fk} {rk} {gk})",
                    "ylabel":"Absolute Error",
                    "xlabel":"Compression Ratio (uncomp/comp)",
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
                    "use_colorbar":True,
                    "ylim":(0, None),
                    "dpi":200,
                    "fig_size":(11,9),
                    "cbar_label":[
                        "Mean Throughput (MB/s)", "Compression Category"
                        ][isall],
                    "cmap":"plasma",
                    "norm":"linear",
                    },
                fig_path=fpath,
                )
            print(f"Generated {fpath}")

    print("finished")

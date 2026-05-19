"""
Script for plotting chunk benchmark results, mainly from full-chunk
'chunk', 'multichunk', and 'volume' experiments. This second version of
the script was created after I realized that it made a lot more sense to
guaruntee full chunks were being retrieved at a time.
"""
import json
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
from ChunkConfig import ChunkConfig

from plotting import plot_scatter

def throughput_model_linear(chunk_size, latency, speed):
    """
    simple functional form for throughput that assumes the amount of time
    it takes to load a chunk is a constant latency plus the size of the
    chunk times the transmission speed...

    chunk_time = speed * chunk_size + latency
    throughput = chunk_size / chunk_time
    throughput = chunk_size / (speed * chunk_size + latency)
    """
    return chunk_size / (chunk_size * speed + latency)

if __name__=="__main__":
    data_dir = Path("data")
    fig_dir = Path("figures/full-chunks")
    #res_json_path = data_dir.joinpath("nldas3_chunk_bench_results.json")
    res_json_path = data_dir.joinpath(
        "nldas3_chunk_bench_results_fullchunk_3.json")
    dtype_size = 4

    model_domain = np.linspace(0, 60, 256)

    run_str = "fc-3-serial"
    exclude_test_types = ["volume", "multichunk"]
    #exclude_test_types = ["volume"]
    #run_str = "fc-3-concurrent"
    #exclude_test_types = ["chunk", "multichunk"]

    results = json.load(res_json_path.open("r"))
    tlabels = results.keys() ## tests performed
    all_clabels = list(set(kl for tl in tlabels for kl in results[tl].keys()))

    chunk_sizes = []
    chunk_sizes_all = []
    throughputs = []
    throughputs_summary = []
    for tl in tlabels:
        if tl in exclude_test_types:
            continue
        for cl in results[tl].keys():
            vstr,cstr = cl.split("-")
            stime,slat,slon = tuple(map(int, cstr.split(".")))
            cc = ChunkConfig(stime, slat, slon)

            tmpd = results[tl][cl]
            tp = np.array(tmpd["point_count"]) \
                    * dtype_size / np.array(tmpd["dt_load"]) / 1000**2
            #'''
            throughputs.append(tp)
            chunk_sizes_all.append([
                np.array(np.prod(cc.cvec) * dtype_size / 1000**2)
                for i in range(len(tmpd["dt_load"]))
                ])
            #'''
            throughputs_summary.append(np.percentile(tp, [25, 50, 75]))
            chunk_sizes.append(np.prod(cc.cvec) * dtype_size / 1000**2)

    chunk_sizes_all = np.concatenate(chunk_sizes_all)
    throughputs = np.concatenate(throughputs)
    tp25,tp50,tp75 = map(np.asarray, zip(*throughputs_summary))

    #'''
    params,cov = curve_fit(
            throughput_model_linear,
            #chunk_sizes,
            #tp50,
            chunk_sizes_all,
            throughputs,
            p0=[.6, .02], ## start w/ 0.1 sec latency, 5 MB/s
            maxfev=4000,
            )

    print("params:", params)
    stderr = np.sqrt(np.diag(cov))
    params_lower = params - 1.96 * stderr
    params_upper = params + 1.96 * stderr
    print("95% CI for latency:", params_lower[0], params_upper[0])
    print("95% CI for speed:", params_lower[1], params_upper[1])
    model_trend = throughput_model_linear(model_domain, *params)
    model_lower = throughput_model_linear(model_domain, *params_lower)
    model_upper = throughput_model_linear(model_domain, *params_upper)
    #'''

    plot_scatter(
        x=chunk_sizes,
        y=tp50,
        yerr=(tp50-tp25, tp75-tp50),
        lines = [
            {"x":model_domain, "y":model_trend,
                "label":"best fit", "color":"black"},
            {"x":model_domain, "y":model_upper,
                "label":"Upper CI 95%", "color":"red", "linestyle":"--"},
            {"x":model_domain, "y":model_lower,
                "label":"Lower CI 95%", "color":"red", "linestyle":"--"},
            ],
        #color=sratios,
        #color=sts,
        plot_spec={
            "title":"throughput wrt chunk size",
            "xlabel":"Chunk Size (MB)",
            "ylabel":"Throughput (MB/s)",
            #"use_colorbar":True,
            #"cbar_label":"Chunk Aspect St (Sx Sy)^(-1/2)",
            #"cbar_label":"Number of Time Steps per Chunk",
            "tight_layout":True,
            #"ylim":[1.3, 1.55],
            },
        fig_path=Path("figures/tmp.png")
        )

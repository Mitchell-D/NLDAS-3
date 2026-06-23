import dask
import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing as mp
import json
from pprint import pprint
from pathlib import Path

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

    pprint(data_dict)
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
            group_starts+offset/2-bwidth/2,
            [labels.get(gk,gk) for gk in gkeys],
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

if __name__=="__main__":
    #store_dir = Path("data/comp-tests")
    store_dir = Path("/rstor/mdodson/nldas3/comp-tests")
    comp_json_path = Path("data/results_compression.json")
    fig_dir = Path("figures")
    fig_path_template = "nldas3_comp_{feat_label}.png"
    nprocs = 8

    '''
    stores = []
    for p in store_dir.iterdir():
        *_,cl,vl,dl = p.stem.split("_")
        stores.append((p, cl, vl, dl))
        ds = xr.open_dataset(p)
        if dl=="int16":
            print(cl, vl, dl, ds.min(), ds.max())
    '''
    comp_res_raw = json.load(comp_json_path.open("r"))
    clabels,flabels,dlabels = zip(*map(
        lambda k:k.split("."),
        comp_res_raw.keys(),
        ))

    comp_res = {}
    comp_res["all-feats"] = {
            dl:{cl:0 for cl in clabels if "lz4" not in cl}
            for dl in dlabels}
    for k,v in comp_res_raw.items():
        cl,fl,dl = k.split(".")
        if "lz4" in cl:
            continue
        if fl not in comp_res.keys():
            comp_res[fl] = {}
        if dl not in comp_res[fl].keys():
            comp_res[fl][dl] = {}
        vnew = v * 365.25 * 23 / 10**12
        comp_res[fl][dl][cl] = vnew
        comp_res["all-feats"][dl][cl] += vnew

    for tl in comp_res.keys():
        #pprint(comp_res[tl])
        plot_nested_bars(
            data_dict=comp_res[tl],
            labels={},
            plot_error_bars=False,
            plot_spec={
                "ylabel":"Size of 23-year PoR (TB)",
                "title":f"NLDAS-3 {tl} PoR Size wrt Compression Regime",
                },
            show=False,
            group_order=["int16", "float16", "float32"],
            fig_path=fig_dir.joinpath(fig_path_template.format(feat_label=tl)),
            bar_colors=[
                "#D46A6A", "#AA3939", "#801515",
                "#77B75B", "#4E9231", "#2E6E12",
                ],
            bar_order=[
                "blosclz-1", "blosclz-5", "blosclz-8",
                "zstd-1", "zstd-5", "zstd-8",
                ]
            )

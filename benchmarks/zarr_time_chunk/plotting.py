import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from pathlib import Path
from matplotlib.colors import ListedColormap
from cartopy.mpl.ticker import LatitudeFormatter,LongitudeFormatter

def plot_point_cloud_3d(coords, cvalues=None, svalues=None, annotations=None,
        axis_labels:list=None, plot_spec:dict={}, fig_path=None, show=False):
    """
    :@param coords: (N,3) array of int coords
    :@param cvalues: (N,) array of floats from which color values are derived
    :@param svalues: (N,) array of floats from which sizes are derived.
    :@param annotations: (N,) list/array of strings to annotate points
    :@param axis_labels: list of arrays/lists providing labels along each axis
    """
    ps = {
        "fig_size":(8,8), "cmap":"magma", "title_size":14, "size_scale":1,
        "annotation_fontsize": 8, "annotation_color": "black",
        "annotation_alpha": 0.9, "annotation_offset": (0.02, 0.02, 0.02),
        }
    ps.update(plot_spec)
    assert coords.shape[-1] == 3
    if not cvalues is None:
        assert cvalues.shape[0]==coords.shape[0]
    if not svalues is None:
        assert svalues.shape[0]==coords.shape[0]

    fig = plt.figure(figsize=ps.get("fig_size"))
    ax = fig.add_subplot(111, projection="3d")
    marker_size = ps.get("marker_size", svalues)
    if marker_size is None:
        marker_size = ps.get("size_scale")
    else:
        marker_size = marker_size * ps.get("size_scale")
    scatter = ax.scatter(
        coords[...,0], coords[...,1], coords[...,2],
        c=ps.get("marker_colors", cvalues),
        s=marker_size,
        vmin=ps.get("vmin"),
        vmax=ps.get("vmax"),
        cmap=ps.get("cmap"),
        alpha=ps.get("alpha",.6),
        marker=ps.get("marker_style", "o"),
        )
    if annotations is not None:
        dx,dy,dz = ps.get("annotation_offset")
        for (x,y,z),label in zip(coords, point_annotations):
            ax.text(
                x + dx, y + dy, z + dz,
                str(label),
                fontsize=ps.get("annotation_fontsize"),
                color=ps.get("annotation_color"),
                alpha=ps.get("annotation_alpha"),
                )
    if axis_labels:
        assert len(axis_labels)==3
        assert all(isinstance(al,(list,tuple)) for al in axis_labels)
        ax.set_xticks(range(len(axis_labels[0])))
        ax.set_yticks(range(len(axis_labels[1])))
        ax.set_zticks(range(len(axis_labels[2])))

        ax.set_xticklabels(axis_labels[0])
        ax.set_yticklabels(axis_labels[1])
        ax.set_zticklabels(axis_labels[2])
    fig.colorbar(scatter, ax=ax, label=ps.get("cb_label"),
                 shrink=ps.get("cb_shrink"))

    plt.title(ps.get("title"),fontweight="bold",fontsize=ps.get("title_size"))

    if ps.get("xtick_rotation"):
        plt.tick_params(axis="x", **{"labelrotation":ps.get(
            "xtick_rotation")})
    if ps.get("ytick_rotation"):
        plt.tick_params(axis="y", **{"labelrotation":ps.get(
            "ytick_rotation")})
    if ps.get("ztick_rotation"):
        plt.tick_params(axis="z", **{"labelrotation":ps.get(
            "ztick_rotation")})

    if not fig_path is None:
        fig.savefig(fig_path.as_posix(), bbox_inches="tight", dpi=80)
    if show:
        plt.show()
    plt.close()

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
        "label_offset":(2,2), "label_fontsize":10,
        "point_label_color":"black", "label_ha":"right", "label_va":"bottom",
        ## collision avoidance
        "avoid_label_overlap":True, "adjust_text_expand":(1.05,1.2),
        "adjust_text_force":(0.1, 0.25),
        "adjust_text_arrowprops":{"arrowstyle":"-", "color":"gray", "lw":0.5},
        ## legend
        "use_point_legend":False, "legend_loc":"best",
        "legend_marker_scale":1.5,
        ## color bar
        "use_colorbar":False, "cbar_label":None, "cbar_label_fontsize":12,
        "cbar_tick_fontsize":10, "cbar_orient":"vertical",
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
                cbar = fig.colorbar(
                        sc, ax=ax,
                        shrink=ps.get("cbar_shrink"),
                        orientation=ps.get("cbar_orient"),
                        )
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
                color=ps.get("point_label_color"),
                rotation=ps.get("point_label_rotation"),
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

    if not ps.get("ylim") is None:
        ax.set_ylim(*ps.get("ylim"))
    if not ps.get("xlim") is None:
        ax.set_xlim(*ps.get("xlim"))


    ax.tick_params(
        axis="both", which="major", labelsize=ps.get("tick_fontsize")
        )
    ax.tick_params(axis="both", which="minor",
                   labelsize=ps.get("tick_fontsize"))

    if ps.get("tight_layout"):
        plt.tight_layout()

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

def plot_geo_ints(int_data, lat, lon, shapes=None,
    geo_bounds=None, latlon_ticks=True,
    int_labels=None, fig_path=None, cbar_ticks=False, colors=None,
    show=False, plot_spec={}):
    """
    Plots a map with pixels colored according to a 2D array of integer values.

    :@param int_data: 2D numpy array of integer values to be visualized
    :@param latitudes: 1D array of latitudes corresponding to rows in `data`
    :@param longitudes: 1D array of longitudes corresponding to columns in`data`
    :@param colors: list or dict mapping indeces present in int_data to
        matplotlib-valid colors
    """
    ps = {
        "xlabel":"", "ylabel":"", "title":"", "dpi":80, "norm":None,
        "figsize":(12,12), "legend_ncols":1, "line_opacity":1, "cmap":"hsv",
        "label_size":14, "title_size":20, "cbar_disable":True,
        "cartopy_feats":["land", "borders", "states"],
        "shape_params":{},
        }
    ps.update(plot_spec)
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})

    if shapes:
        tmp_fc = ps.get("shape_params", {}).get("facecolor", "blue")
        tmp_ec = ps.get("shape_params", {}).get("edgecolor", "black")
        tmp_lw = ps.get("shape_params", {}).get("linewidth", 1)
        tmp_ls = ps.get("shape_params", {}).get("linestyle", "solid")
        tmp_zo = ps.get("shape_params", {}).get("zorder", 10)
        tmp_aa = ps.get("shape_params", {}).get("alpha", .5)
        if not isinstance(tmp_fc, (list, tuple)):
            tmp_fc = [tmp_fc for i in range(len(shapes))]
        if not isinstance(tmp_ec, (list, tuple)):
            tmp_ec = [tmp_ec for i in range(len(shapes))]
        if not isinstance(tmp_lw, (list, tuple, np.ndarray)):
            tmp_lw = [tmp_lw for i in range(len(shapes))]
        if not isinstance(tmp_ls, (list, tuple)):
            tmp_ls = [tmp_ls for i in range(len(shapes))]
        if not isinstance(tmp_zo, (list, tuple)):
            tmp_zo = [tmp_zo for i in range(len(shapes))]
        if not isinstance(tmp_aa, (list, tuple, np.ndarray)):
            tmp_aa = [tmp_aa for i in range(len(shapes))]
        for i,s in enumerate(shapes):
            ax.add_geometries(
                [s],
                ccrs.PlateCarree(),
                facecolor=tmp_fc[i],
                edgecolor=tmp_ec[i],
                linewidth=tmp_lw[i],
                linestyle=tmp_ls[i],
                zorder=tmp_zo[i],
                alpha=tmp_aa[i],
                )

    if "land" in ps.get("cartopy_feats"):
        ax.add_feature(
                cfeature.LAND,
                #linestyle=ps.get("border_style", "-"),
                #linewidth=ps.get("border_linewidth", 2),
                #edgecolor=ps.get("border_color", "black"),
                )
    if "borders" in ps.get("cartopy_feats"):
        ax.add_feature(
                cfeature.BORDERS,
                linestyle=ps.get("border_style", "-"),
                linewidth=ps.get("border_linewidth", 2),
                edgecolor=ps.get("border_color", "black"),
                )
    if "states" in ps.get("cartopy_feats"):
        ax.add_feature(
                cfeature.STATES,
                linestyle=ps.get("border_style", "-"),
                linewidth=ps.get("border_linewidth", 2),
                edgecolor=ps.get("border_color", "black"),
                )
    if geo_bounds is None:
        geo_bounds = [np.amin(lon), np.amax(lon), np.amin(lat), np.amax(lat)]
    ax.set_extent(geo_bounds, crs=ccrs.PlateCarree())

    if not ps.get("ylim") is None:
        ax.set_ylim(*ps.get("ylim"))
    if not ps.get("xlim") is None:
        ax.set_xlim(*ps.get("xlim"))

    m_invalid = ~np.isfinite(int_data)
    int_data[m_invalid] = int_data[~m_invalid][0]
    int_data = int_data.astype(int)

    ## assign each unique integer to an index
    unq_ints = np.unique(int_data)
    val_to_ix = {v:ix for ix,v in enumerate(unq_ints)}
    if colors is None:
        ref_cmap = plt.get_cmap(ps.get("cmap", "tab20"), unq_ints.size)
        cmap = ListedColormap([ref_cmap(i) for i in range(unq_ints.size)])
    else:
        cmap = ListedColormap([colors[v] for v in unq_ints])
    if int_labels is None:
        ix_labels = list(unq_ints)
    else:
        ix_labels = [int_labels[v] for v in unq_ints]
    if isinstance(int_data, np.ma.MaskedArray):
        int_data = int_data.data
    ix_data = np.vectorize(val_to_ix.get)(int_data).astype(float)
    ix_data[m_invalid] = np.nan

    im = ax.imshow(
            ix_data,
            origin=ps.get("origin", "upper"),
            cmap=cmap,
            extent=geo_bounds,
            interpolation=ps.get("interpolation")
            )

    if latlon_ticks:
        lonmin,lonmax,latmin,latmax = geo_bounds
        frq = ps.get("tick_frequency", 1)
        ax.set_yticks(np.linspace(latmin,latmax,ix_data.shape[0])[::frq],
                crs=ccrs.PlateCarree())
        ax.set_xticks(np.linspace(lonmin,lonmax,ix_data.shape[1])[::frq],
                crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.tick_params(rotation=ps.get("tick_rotation", 0))


    if not ps.get("cbar_disable"):
        cbar = plt.colorbar(
                im, ax=ax,
                orientation=ps.get("cbar_orient", "vertical"),
                pad=ps.get("cbar_pad", 0.05),
                shrink=ps.get("cbar_shrink", 1.)
                )

        ## make a scale that centers ticks on their color bar increments
        if cbar_ticks:
            nunq = unq_ints.size
            ticks = np.linspace(0, nunq-1, nunq*2+1)[1::2]
            #ticks = np.array(list(range(nunq))) * (nunq-1)/nunq + .5
            cbar.set_ticks(ticks)
            cbar.ax.tick_params(rotation=ps.get("cbar_tick_rotation", 0))
            cbar.set_ticklabels(ix_labels)
            cbar.ax.tick_params(labelsize=ps.get("cbar_fontsize", 14))

        cbar.set_label(ps.get("cbar_label"))
    ax.set_title(ps.get("title", ""), fontsize=ps.get("title_fontsize", 18))
    if not fig_path is None:
        fig.set_size_inches(*ps.get("figsize"))
        fig.savefig(
            fig_path.as_posix(),
            bbox_inches="tight",
            dpi=ps.get("dpi", 100),
            )
    if show:
        plt.show()
    plt.close()
    return

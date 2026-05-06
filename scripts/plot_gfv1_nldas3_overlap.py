"""
Plot the number of chunks overlapped by each GFv1 polygon given the raster
created by create_nldas3_chunk_polygons.py and the adjacency matrix calculated
by calc_gfv1_nldas3_overlap.py
"""
import numpy as np
import netCDF4 as nc
import pickle as pkl
import boto3
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from shapely.geometry import Polygon
from matplotlib.colors import ListedColormap
from cartopy.mpl.ticker import LatitudeFormatter,LongitudeFormatter

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

if __name__=="__main__":
    data_dir = Path("data")
    fig_dir = Path("figures")
    nldas3_param_path = data_dir.joinpath("nldas3_params.nc")

    #gfv1_raster_pkl_path = data_dir.joinpath("nldas3_gfv1_land.pkl")
    #adj_npz_path = data_dir.joinpath("adjacency_gfv1_nldas3_land.npz")
    #out_png_overlap_path = fig_dir.joinpath("overlap_gfv1_nldas3_land.png")

    gfv1_raster_pkl_path = data_dir.joinpath("nldas3_gfv1_all.pkl")
    adj_npz_path = data_dir.joinpath("adjacency_gfv1_nldas3_all.npz")
    out_png_overlap_path = fig_dir.joinpath("overlap_gfv1_nldas3_all.png")

    oob_val_gfv1 = -1
    oob_val_nldas3 = 65535
    oob_val_out_land = -1 ## oob wrt gfv1 but valid nldas3 pixel
    oob_val_out_both = -2 ## oob wrt gfv1 and nldas3
    oob_color_land = "#1b1d26"
    oob_color_both = "black"
    cmap_name = "rainbow"
    lon_bounds = (-127, -64)
    lat_bounds = (23, 53.5)

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
        ## construct shapely polygons and index slices for each NLDAS-3 chunk

    slc_lat = slice(np.argmin((nldas3_lats-lat_bounds[0])**2),
            np.argmin((nldas3_lats-lat_bounds[1])**2))
    slc_lon = slice(np.argmin((nldas3_lons-lon_bounds[0])**2),
            np.argmin((nldas3_lons-lon_bounds[1])**2))

    ## load the adjacency matrix and info
    adj = np.load(adj_npz_path, allow_pickle=True)

    ## count nonzero over chunk axis to get chunks per gfv1 polygon
    pchunks = np.count_nonzero(adj["adj_gfv1_nldas3"], axis=1)

    ## get a mapping from polygon coords to chunk overlap counts
    pcoords_to_count = {
        int(c):int([v,oob_val_out_land][int(c==oob_val_gfv1)])
        for c,v in zip(adj["poly_coords"], pchunks)
        }
    f_pcoords_to_count = np.vectorize(pcoords_to_count.get)

    ## load the GFv1 polygon raster
    praster,_,_ = pkl.load(gfv1_raster_pkl_path.open("rb"))

    ## map polygon indices to their counts
    ccounts = f_pcoords_to_count(praster.ravel()).reshape(praster.shape)

    ## where OOB for both, assign a new number
    m_oob_both = (~nldas3_land_mask) & (ccounts == oob_val_out_land)
    ccounts[m_oob_both] = oob_val_out_both

    ## make a mapping between unique ints and their colors
    max_count = np.amax(ccounts)
    unq_count = np.unique(ccounts)
    cmap = plt.get_cmap(cmap_name)
    colors = {
        **{v:cmap(v/max_count) for v in unq_count},
        oob_val_out_land:oob_color_land,
        oob_val_out_both:oob_color_both,
        }

    ## plot the raster-based chunk inclusions of unmasked pixels
    plot_geo_ints(
        int_data=ccounts[slc_lat, slc_lon],
        lat=nldas3_lats[slc_lat],
        lon=nldas3_lons[slc_lon],
        shapes=None,
        geo_bounds=None,
        latlon_ticks=True,
        int_labels=None,
        fig_path=out_png_overlap_path,
        cbar_ticks=True,
        colors=colors,
        show=False,
        plot_spec={
            "title":"Number of NLDAS-3 chunks intersected by GFv1 polygons",
            "title_fontsize":16,
            "cbar_label":"NLDAS-3 chunks count (-2,-1 -> OOB)",
            "cbar_orient":"horizontal",
            "cbar_pad":.1,
            "cbar_shrink":.9,
            "cartopy_feats":["borders", "states"],
            "cbar_disable":False,
            "origin":"lower",
            "tick_frequency":500,
            "tick_rotation":45,
            "border_linewidth":1,
            "dpi":200,
            }
        )

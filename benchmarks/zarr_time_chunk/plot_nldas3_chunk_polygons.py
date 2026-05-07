import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import netCDF4 as nc
import multiprocessing as mp

from pathlib import Path
from matplotlib.colors import ListedColormap
from cartopy.mpl.ticker import LatitudeFormatter,LongitudeFormatter
from shapely.geometry import Polygon

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

def plot_nldas3_chunk_polygons(nldas3_param_path, poly_npz_path,
        out_path_polys=None, out_path_ints=None,
        plot_spec_polys={}, plot_spec_ints={},
        randomize_cmap=False, seed=None, cmap_str="prism",
        water_color="#1b1d26", oob_color="black", oob_value=65535):
    """
    """
    ps_polys = {
        "title":"NLDAS-3 chunks and land mask",
        "cbar_label":"land mask",
        "cartopy_feats":["borders", "states"],
        "cbar_disable":True,
        "origin":"lower",
        "tick_frequency":500,
        "tick_rotation":45,
        "border_linewidth":1,
        "dpi":500,
        "shape_params":{
            "facecolor":"from_cmap",
            "edgecolor":"black",
            "linewidth":.5,
            "linestyle":"solid",
            "zorder":10,
            "alpha":.66,
            },
        }
    ps_polys.update(plot_spec_polys)
    ps_ints = {
        "title":"NLDAS-3 chunks and land mask",
        "cbar_label":"land mask",
        "cartopy_feats":["borders", "states"],
        "cbar_disable":True,
        "origin":"lower",
        "tick_frequency":500,
        "tick_rotation":45,
        "border_linewidth":1,
        "dpi":500,
        }
    ps_ints.update(plot_spec_ints)

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

    chunks = np.load(poly_npz_path, allow_pickle=True)

    cinfo = chunks["chunk_info"]
    rng = np.random.default_rng(seed)

    if not out_path_polys is None:
        cpolys = [Polygon(cd["geometry"][0]) for cd in cinfo]
        cmap = plt.get_cmap(cmap_str)

        cmap_index = np.arange(len(cinfo))
        if randomize_cmap:
            rng.shuffle(cmap_index)

        if ps_polys["shape_params"]["facecolor"] == "from_cmap":
            ps_polys["shape_params"]["facecolor"]  = [
                oob_color if not c["has_valid_points"]
                else cmap(i/(len(cinfo)-1))
                for i,c in zip(cmap_index, cinfo)
                ]

        plot_geo_ints(
            int_data=nldas3_land_mask.astype(np.uint8),
            lat=nldas3_lats,
            lon=nldas3_lons,
            shapes=cpolys,
            geo_bounds=None,
            latlon_ticks=True,
            int_labels=None,
            fig_path=out_path_polys,
            cbar_ticks=False,
            colors=["black", "white"],
            show=False,
            plot_spec=ps_polys,
            )

    if not out_path_ints is None:
        cmasks = chunks["chunk_masks"]
        cunq = np.unique(cmasks)
        cvalid = cunq[cunq != oob_value]
        cmap = plt.get_cmap(cmap_str)
        ## formerly un-used int for pixels not in land mask, but in a chunk
        ## that has some valid land pixels
        water_value = cvalid[-1] + 1
        cmasks = np.where(
            (~nldas3_land_mask) & (cmasks != oob_value),
            water_value,
            cmasks,
            )

        cmap_index = np.arange(cvalid.size)
        if randomize_cmap:
            rng.shuffle(cmap_index)

        ## make a mapping between unique ints and their colors
        colors = {
            **{v:cmap(i/(cvalid.size-1)) for i,v in zip(cmap_index, cvalid)},
            oob_value:oob_color,
            water_value:water_color,
            }

        ## plot the raster-based chunk inclusions of unmasked pixels
        plot_geo_ints(
            int_data=cmasks,
            lat=nldas3_lats,
            lon=nldas3_lons,
            shapes=None,
            geo_bounds=None,
            latlon_ticks=True,
            int_labels=None,
            fig_path=out_path_ints,
            cbar_ticks=False,
            colors=colors,
            show=False,
            plot_spec=ps_ints,
            )
    return [p for p in [out_path_polys, out_path_ints] if not p is None]

def mp_plot_nldas3_chunk_polygons(args):
    return args,plot_nldas3_chunk_polygons(**args)

if __name__=="__main__":
    data_dir = Path("data")
    fig_dir = Path("figures")

    nprocs =  8
    nldas3_path = data_dir.joinpath("nldas3_params.nc")
    poly_dir_path = data_dir.joinpath("polys")

    plot_npz_paths = [
        fp for fp in poly_dir_path.iterdir() if fp.name.endswith(".npz")
        ]

    args = [{
        "nldas3_param_path":nldas3_path,
        "poly_npz_path":nzp,
        "out_path_ints":fig_dir.joinpath(f"poly_{nzp.stem}.png"),
        "out_path_polys":fig_dir.joinpath(f"int_{nzp.stem}.png"),
        "plot_spec_ints":{
            "title":f"NLDAS-3 {cc}",
            },
        "plot_spec_polys":{
            "title":f"NLDAS-3 {cc}",
            },
        "randomize_cmap":True,
        "cmap_str":"terrain",
        "water_color":"#1b1d26",
        "oob_color":"black",
        "oob_value":65535,
        } for nzp,cc in map(
            lambda p: (p,tuple(map(int, p.stem.split("_")[-1].split("-")))),
            plot_npz_paths,
            )
        ]

    with mp.Pool(nprocs) as pool:
        for a,p in pool.imap_unordered(mp_plot_nldas3_chunk_polygons, args):
            print(f"Generated:", p)

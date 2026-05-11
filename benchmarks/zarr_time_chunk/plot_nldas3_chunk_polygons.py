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

from plotting import plot_geo_ints

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
        "out_path_ints":fig_dir.joinpath(f"int_{nzp.stem}.png"),
        "out_path_polys":fig_dir.joinpath(f"poly_{nzp.stem}.png"),
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

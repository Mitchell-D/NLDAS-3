import numpy as np
import xarray as xr
import multiprocessing as mp
from pprint import pprint
from pathlib import Path
from zarr.codecs import BloscCodec

def create_float_zarr(src_path, out_path, var_key, blosc_args=None,
        encoding_args={}):
    assert not out_path.exists()
    da = xr.open_dataset(src_path.as_posix())[var_key]
    if blosc_args is None:
        comp = None
    else:
        comp = BloscCodec(**blosc_args)
    da.to_dataset().to_zarr(
        store=out_path.as_posix(),
        mode="w",
        encoding={
            var_key:{
                "compressors":[comp],
                "dtype":"float32",
                **encoding_args,
                },
            }
        )
    return out_path

def mp_create_float_zarr(args):
    return args,create_float_zarr(**args)

def create_int_zarr(src_path, out_path, var_key, blosc_args=None,
        encoding_args={}):
    assert not out_path.exists()
    da = xr.open_dataset(src_path.as_posix())[var_key]
    m = (da.vmax - da.vmin) / 65535 ## max for uint16
    b = da.vmin
    if "dtype" in encoding_args.keys():
        if encoding_args["dtype"] != "uint16":
            raise ValueError(f"int dtype must be uint16")
    if blosc_args is None:
        comp = None
    else:
        comp = BloscCodec(**blosc_args)
    da.to_dataset().to_zarr(
        store=out_path.as_posix(),
        mode="w",
        encoding={
            var_key:{
                "compressors":[comp],
                "dtype":"uint16",
                "scale_factor":np.float16(m),
                "add_offset":np.float16(b),
                **encoding_args,
                },
            }
        )
    return out_path

def mp_create_int_zarr(args):
    return args,create_int_zarr(**args)

if __name__=="__main__":
    source_nc_path = Path(
        "/rstor/mdodson/nldas3/NLDAS_FOR0010_H.A20130930.030.nc")
    out_dir = Path("data/comp-tests/")
    vlabels = [
        k for k in xr.open_dataset(source_nc_path).variables.keys()
        if k not in ("lat", "lon", "time")
        ]
    nprocs = 6

    int_encoding_args = {
            "int16":{"chunks":(6,500,900)},
            }
    float_encoding_args = {
            "float16":{"chunks":(6,500,900), "dtype":"float16"},
            "float32":{"chunks":(6,500,900), "dtype":"float32"},
            }

    int_blosc_variations = {
        "zstd-1":{"cname":"zstd", "clevel":1, "shuffle":"bitshuffle"},
        "zstd-5":{"cname":"zstd", "clevel":5, "shuffle":"bitshuffle"},
        "zstd-8":{"cname":"zstd", "clevel":8, "shuffle":"bitshuffle"},

        "lz4-1":{"cname":"lz4", "clevel":1, "shuffle":"bitshuffle"},
        "lz4-5":{"cname":"lz4", "clevel":5, "shuffle":"bitshuffle"},
        "lz4-8":{"cname":"lz4", "clevel":8, "shuffle":"bitshuffle"},

        "blosclz-1":{"cname":"blosclz", "clevel":1, "shuffle":"bitshuffle"},
        "blosclz-5":{"cname":"blosclz", "clevel":5, "shuffle":"bitshuffle"},
        "blosclz-8":{"cname":"blosclz", "clevel":8, "shuffle":"bitshuffle"},
        }

    float_blosc_variations = {
        "zstd-1":{"cname":"zstd", "clevel":1, "shuffle":"bitshuffle"},
        "zstd-5":{"cname":"zstd", "clevel":5, "shuffle":"bitshuffle"},
        "zstd-8":{"cname":"zstd", "clevel":8, "shuffle":"bitshuffle"},

        "lz4-1":{"cname":"lz4", "clevel":1, "shuffle":"bitshuffle"},
        "lz4-5":{"cname":"lz4", "clevel":5, "shuffle":"bitshuffle"},
        "lz4-8":{"cname":"lz4", "clevel":8, "shuffle":"bitshuffle"},

        "blosclz-1":{"cname":"blosclz", "clevel":1, "shuffle":"bitshuffle"},
        "blosclz-5":{"cname":"blosclz", "clevel":5, "shuffle":"bitshuffle"},
        "blosclz-8":{"cname":"blosclz", "clevel":8, "shuffle":"bitshuffle"},
        }

    int_args = [{
        "src_path":source_nc_path,
        "out_path":out_dir.joinpath(
            f"{source_nc_path.stem}_{bk}_{l}_{ek}.zarr"),
        "var_key":l,
        "blosc_args":bv,
        "encoding_args":ev,
        }
        for l in vlabels
        for ek,ev in int_encoding_args.items()
        for bk,bv in int_blosc_variations.items()
        ]

    pprint(int_args)

    with mp.Pool(nprocs) as pool:
        for a,r in pool.imap(mp_create_int_zarr, int_args):
            print(f"Finished {r}")

    float_args = [{
        "src_path":source_nc_path,
        "out_path":out_dir.joinpath(
            f"{source_nc_path.stem}_{bk}_{l}_{ek}.zarr"),
        "var_key":l,
        "blosc_args":bv,
        "encoding_args":ev,
        }
        for l in vlabels
        for ek,ev in float_encoding_args.items()
        for bk,bv in float_blosc_variations.items()
        ]

    pprint(int_args)

    with mp.Pool(nprocs) as pool:
        for a,r in pool.imap(mp_create_float_zarr, float_args):
            print(f"Finished {r}")


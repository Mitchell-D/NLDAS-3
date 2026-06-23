import numpy as np
import xarray as xr
import multiprocessing as mp
import zarr
import json
from time import perf_counter
from pprint import pprint
from pathlib import Path
from zarr.registry import register_codec

import compression_config as config
from ClipScaleQuantizeCodec import ClipScaleQuantizeCodec
from ChunkConfig import ChunkConfig
register_codec(ClipScaleQuantizeCodec.codec_id, ClipScaleQuantizeCodec)

def get_codec_pipeline(pipeline, feat, time_res, dtype):
    """
    parse a pipeline configuration into arguments for zarr.create_array
    """
    conditional_norm_feats = ["Rainf"]

    dtype = np.dtype(dtype)

    filters = []
    serializer = "auto"
    compressors = []
    cfg = {"filters":[], "serializer":None, "compressors":[]}
    for ak,aargs in pipeline:
        a = {**config.default_codecs[ak], **aargs}
        ctype = a["_ctype"]
        del a["_ctype"]
        if ctype == "filter":
            if ak == "bitround":
                filters.append(zarr.codecs.BitRound(**a))
            elif ak == "intnorm":
                ## set custom resolution if requested
                if a["resolution"] == "custom":
                    a["resolution"] = config.get_custom_norm_res(feat,time_res)
                imin = np.iinfo(np.dtype(a["store_dtype"])).min
                imax = np.iinfo(np.dtype(a["store_dtype"])).max
                a["source_dtype"] = str(dtype)
                if a["resolution"] > imax-imin:
                    raise ValueError(
                        f"{a['resolution']=} can't be >{imax-imin}")
                if imin <= a["mask_val"] < a["resolution"]:
                    raise ValueError(
                        f"{a['mask_val']=} is in norm range for resolution",
                        a["resolution"], (imin, imax)
                        )
                ## set norm bounds from config
                bounds = config.norm_bounds[feat]
                if feat in config.conditional_norm_feats:
                    bounds = bounds[time_res]
                a["bounds"] = bounds
                filters.append(ClipScaleQuantizeCodec(**a))
            elif ak == "delta":
                filters.append(zarr.codecs.Delta(**a))
            else:
                raise ValueError(f"unrecognized filter:", ak)
            cfg["filters"].append((ak,a))
        elif ctype == "serializer":
            if ak == "pcodec":
                serializer = zarr.codecs.PCodec(**a)
            elif ak == "zfpy":
                serializer = zarr.codecs.ZFPY(**a)
            else:
                raise ValueError(f"unrecognized serializer:", ak)
            cfg["serializer"] = (ak,a)
        elif ctype == "compressor":
            if ak in config.compressors.keys():
                compressors.append(zarr.codecs.BloscCodec(cname=ak, **a))
            else:
                raise ValueError(f"unrecognized compressor:", ak)
            cfg["compressors"].append((ak,a))
    return {
        "dtype":dtype,
        "filters":filters,
        "serializer":serializer,
        "compressors":compressors,
        }, cfg


def mp_benchmark_compression(args):
    return args,benchmark_compression(**args)
def benchmark_compression(feat, zarr_path, region_group, out_group,
        out_zarr_array_name, chunks, shards, dtype, filters=[],
        serializer=None, compressors=[], delete_after_finished=True,
        overwrite_if_exists=False, **kwargs):
    """ """
    zreg = zarr.open(zarr_path, path=region_group, mode="r")
    if feat not in zreg["baseline"].keys():
        raise ValueError(
            f"feat must be one of ",
            list(zgreg["baseline"].keys())
            )
    zout = zarr.open(zarr_path, path=out_group, mode="a")
    new_array = True
    if out_zarr_array_name in zout.keys():
        if overwrite_if_exists:
            print("overwriting existing zarr array {out_zarr_array_name}")
            del zout[out_zarr_array_name]
        else:
            new_array = False

    total_size = zreg["baseline"][feat].size \
            * zreg["baseline"][feat].dtype.itemsize

    cc = ChunkConfig(chunks)
    cslcs = cc.gen_chunks(zreg["baseline"][feat].shape)
    write_times = []
    if new_array:
        zout.create_array(
            out_zarr_array_name,
            shape=zreg["baseline"][feat].shape,
            chunks=chunks,
            shards=shards,
            filters=filters,
            serializer=serializer,
            compressors=compressors,
            dtype=dtype,
            )
        for cs in cslcs:
            x = zreg["baseline"][feat][*cs]
            t0 = perf_counter()
            zout[out_zarr_array_name][*cs] = x
            tf = perf_counter()
            write_times.append(tf-t0)

    load_times = []
    error_stats = None
    error_stats = {
        "max":[], "min":[], "mean":[], "absmean":[], "absstdv":[], "count":[],
        }
    cslcs = cc.gen_chunks(zreg["baseline"][feat].shape)
    print(out_zarr_array_name)
    for cs in cslcs:
        x = zreg["baseline"][feat][*cs]
        m_valid = np.where(np.isfinite(x))
        t0 = perf_counter()
        y = zout[out_zarr_array_name][*cs]
        tf = perf_counter()
        #print(x.dtype, x[m_valid])
        #print(y.dtype, y[m_valid])
        load_times.append(tf-t0)
        dv = y[m_valid] - x[m_valid]
        adv = np.abs(dv)

        error_stats["count"].append(dv.size)
        error_stats["mean"].append(float(np.average(dv)))
        error_stats["absmean"].append(float(np.average(adv)))
        error_stats["absstdv"].append(float(np.std(adv)))
        error_stats["min"].append(float(np.amin(dv)))
        error_stats["max"].append(float(np.amax(dv)))

    counts = np.array([error_stats["count"]])
    avgs = np.array([error_stats["mean"]])
    aavgs = np.array([error_stats["absmean"]])
    stdvs = np.array([error_stats["absstdv"]])

    count = np.sum(counts)
    aavg = np.sum(counts*aavgs) / count
    avg = np.sum(counts*avgs) / count
    stdv = np.sum(counts*(stdvs**2 + (aavgs-aavg)**2)) / count
    all_stats = {
        "max":float(np.amax(error_stats["max"])),
        "min":float(np.amin(error_stats["min"])),
        "absmean":float(aavg),
        "mean":float(avg),
        "absstdv":float(stdv),
        "count":float(count),
        }

    ## query the file system for compressed chunk sizes
    ddir = zarr_path.joinpath(zout.path).joinpath(f"{out_zarr_array_name}/c")
    comp_size = float(np.sum([
        p.stat().st_size for p in ddir.rglob("*") if p.is_file()
        ]))

    if delete_after_finished:
        del zout[out_zarr_array_name]

    return write_times,load_times,comp_size,total_size,all_stats


if __name__=="__main__":
    out_dir = Path("data/comp-tests/")
    nworkers = 12
    extract_feats = config.extract_feats

    time_res = "hourly"

    zarr_path_out = Path(f"data/store/compression_{time_res}.zarr")
    out_json = Path(f"data/results_compression_{time_res}.json")

    out_chunks = config.encoding_args["chunks"]
    out_shards = config.encoding_args["shards"]

    delete_after_finished = True
    overwrite_if_exists = False

    run_feats = [
        "Tair",
        "PSurf",
        "SWdown",
        "Wind_E",
        "Rainf",
        ]

    run_regions = [
        "midwest",
        "alaska",
        "desertw",
        "yucatan",
        "southeast",
        "mountainw",
        ]

    run_pipelines = [
        "dtype:f2_zstd",
        "dtype:f2_lz4hc",
        "dtype:f4_zstd",
        "dtype:f4_lz4hc",

        "dtype:f4_intnorm:4096",
        "dtype:f4_intnorm:4096_zstd",
        "dtype:f4_intnorm:4096_zstd:shuffle",
        "dtype:f4_intnorm:4096_zstd:bitshuffle",
        "dtype:f4_intnorm:4096_delta:u2,u2_zstd:5,bitshuffle",
        "dtype:f4_intnorm:4096,i4_zfpy_zstd:5,bitshuffle",
        "dtype:f4_intnorm:4096,u4_pcodec_zstd:5,bitshuffle",

        "dtype:f4_intnorm:2048",
        "dtype:f4_intnorm:2048_zstd:shuffle",
        "dtype:f4_intnorm:2048_zstd:bitshuffle",
        "dtype:f4_intnorm:2048_delta:u2,u2_zstd:5,bitshuffle",
        "dtype:f4_intnorm:2048,i4_zfpy_zstd:5,bitshuffle",
        "dtype:f4_intnorm:2048,u4_pcodec_zstd:5,bitshuffle",

        "dtype:f4_intnorm:custom",
        "dtype:f4_intnorm:custom_zstd:shuffle",
        "dtype:f4_intnorm:custom_zstd:bitshuffle",
        "dtype:f4_intnorm:custom_delta:u2,u2_zstd:5,bitshuffle",
        "dtype:f4_intnorm:custom,i4_zfpy_zstd:5,bitshuffle",
        "dtype:f4_intnorm:custom,u4_pcodec_zstd:5,bitshuffle",

        "dtype:f4_bitround:12_zfpy_zstd:bitshuffle",
        "dtype:f4_bitround:10_zfpy_zstd:bitshuffle",
        "dtype:f4_bitround:9_zfpy_zstd:bitshuffle",
        "dtype:f4_bitround:8_zfpy_zstd:bitshuffle",

        "dtype:f4_bitround:12_pcodec_zstd:bitshuffle",
        "dtype:f4_bitround:10_pcodec_zstd:bitshuffle",
        "dtype:f4_bitround:9_pcodec_zstd:bitshuffle",
        "dtype:f4_bitround:8_pcodec_zstd:bitshuffle",

        "dtype:f4_bitround:12_zstd:bitshuffle",
        "dtype:f4_bitround:10_zstd:bitshuffle",
        "dtype:f4_bitround:9_zstd:bitshuffle",
        "dtype:f4_bitround:8_zstd:bitshuffle",
        #"dtype:f4_bitround:12",
        #"dtype:f4_bitround:6",
        #"dtype:f4_bitround:6_zstd:bitshuffle",
        #"dtype:f4_bitround:3",
        #"dtype:f4_bitround:3_zstd:bitshuffle"
        ]

    ## feature -> pipeline -> region -> (max, min, mean, stdv, count)
    pprint(list(config.pipelines.keys()))

    skip = []
    if not overwrite_if_exists and out_json.exists():
        cur = json.load(out_json.open("r"))
        for fk in cur.keys():
            for pk in cur[fk].keys():
                for rk in cur[fk][pk].keys():
                    print(f"skipping:",fk, pk, rk)
                    skip.append((fk,pk,rk))

    missing_pipelines = []
    for pk in run_pipelines:
        if pk not in config.pipelines.keys():
            missing_pipelines.append(pk)
    if len(missing_pipelines) != 0:
        raise ValueError("requested pipelines not configured:",
                missing_pipelines)

    args = []
    settings = {}
    zreg = zarr.open(zarr_path_out, mode="r", path="/regions")
    for fk in extract_feats:
        if fk not in run_feats:
            continue
        for pk,pd in config.pipelines.items():
            if pk not in run_pipelines:
                continue
            pout,pset = get_codec_pipeline(
                    pipeline=pd["pipeline"],
                    dtype=pd["dtype"],
                    feat=fk,
                    time_res=time_res
                    )

            for rk in zreg.keys():
                if rk not in run_regions:
                    continue
                if (fk,pk,rk) in skip:
                    continue
                args.append({
                    "feat":fk,
                    "zarr_path":zarr_path_out,
                    "region_group":f"/regions/{rk}",
                    "out_group":f"variants",
                    "out_zarr_array_name":f"{fk}|{pk}|{rk}",
                    "dtype":pout["dtype"],
                    "chunks":out_chunks,
                    "shards":out_shards,
                    "filters":pout["filters"],
                    "serializer":pout["serializer"],
                    "compressors":pout["compressors"],
                    "delete_after_finished":delete_after_finished,
                    "overwrite_if_exists":overwrite_if_exists,
                    "pipeline_name":pk,
                    "region_name":rk,
                    })
                settings[(fk,pk,rk)] = pset

    args = sorted(
        args,
        key=lambda a:(a["feat"], a["region_name"], a["pipeline_name"]),
        )

    out_fields = [
        "write_time",
        "load_time",
        "chunk_size",
        "total_size",
        "error_stats",
        ]
    zres = zarr.open(zarr_path_out, mode="a")
    if "variants" not in zres.keys():
        zres.create_group("variants")

    all_results = zres.attrs.get("benchmarks", {})
    '''
    for a,r in map(mp_benchmark_compression, args):
        fk = a["feat"]
        pk = a["pipeline_name"]
        rk = a["region_name"]
        if fk not in all_results.keys():
            all_results[fk] = {}
        if pk not in all_results[fk].keys():
            all_results[fk][pk] = {}
        all_results[fk][pk][rk] = dict(zip(out_fields, r))
        zres.attrs.update({"benchmarks":all_results})
        print(f"Finished {fk} {pk} {rk}")
        print(r)
        print()
    '''

    #'''
    with mp.Pool(nworkers) as pool:
        for a,r in pool.imap(mp_benchmark_compression, args):
            fk = a["feat"]
            pk = a["pipeline_name"]
            rk = a["region_name"]
            if fk not in all_results.keys():
                all_results[fk] = {}
            if pk not in all_results[fk].keys():
                all_results[fk][pk] = {}
            all_results[fk][pk][rk] = dict(zip(out_fields, r))
            zres.attrs.update({"benchmarks":all_results})
            json.dump(all_results, out_json.open("w"))
            print(f"Finished {fk} {pk} {rk}")
            print(r)
            print()

    #'''

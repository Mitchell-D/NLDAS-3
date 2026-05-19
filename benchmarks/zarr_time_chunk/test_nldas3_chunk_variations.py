"""
Script for generating a subset of nldas3 data using various chunk layouts,
and benchmarking access across space and time.
"""
import time
import json
import random
import multiprocessing as mp
import numpy as np
import fsspec
import xarray as xr
import zarr
import dask
from pathlib import Path
from pprint import pprint

from ChunkConfig import ChunkConfig

def nldas3_subset_to_zarr(time_slice, lat_slice, lon_slice, times_per_load,
        acquire_var, out_zarr_path, out_chunks, out_dtype):
    """
    :@param time_slice: slice of `%Y-%m-%d` strings indicating date ranges to
        acquire for the subset
    :@param lat_slice: index slice of latitudes to acquire for subset
    :@param lon_slice: index slice of longitudes to acquire for subset
    :@param times_per_load: Number of timesteps to acquire for each call to
        xarray.Dataset.load()
    :@param acquire_var: String label of nldas3 variable to download
    :@param out_zarr_path: non-existing zarr directory store path to dump to
    :@param out_chunks: chunk configuration of output array.
    """
    ## set up a virtual file system using the kerchunk references
    ref_fs = fsspec.filesystem(
        "reference",
        fo="s3://nasa-waterinsight/virtual/nldas3_daily.parq",
        remote_protocol="s3",
        asynchronous=True,
        remote_options={"asynchronous":True, "anon":True},
        target_options={"anon":True},
        lazy=True
        )

    ## create a xarray Dataset object based on the virtual chunks.
    ds = xr.open_zarr(
            ref_fs.get_mapper(""),
            consolidated=False,
            )

    ## declare the overall subset and determine time slices
    sub = ds.sel(time=time_slice).isel(lat=lat_slice, lon=lon_slice)
    ntimes = sub.time.size
    nlats = sub.lat.size
    nlons = sub.lon.size
    nslices = ntimes // times_per_load + int(ntimes % times_per_load != 0)
    time_slices = [
        slice(int(ix0), int(min(ix0+times_per_load, ntimes)))
        for ix0 in np.arange(nslices) * times_per_load
        ]

    ## initialize the dataset with a lazy-loaded empty dask array
    default_array_label = acquire_var + "-" + ".".join(map(str,out_chunks))
    default_array = dask.array.zeros(
        (ntimes, nlats, nlons),
        chunks=out_chunks,
        dtype=out_dtype,
        )
    ds = xr.Dataset(
        {default_array_label:(("time", "lat", "lon"), default_array)},
        coords={
            "time":sub.time.load().to_numpy(),
            "lat":sub.lat.load().to_numpy(),
            "lon":sub.lon.load().to_numpy(),
            },
        )
    ds.to_zarr(out_zarr_path, compute=False)

    ## load data into the file one slice at a time
    for s in time_slices:
        sub_array = sub[acquire_var].isel(time=s).load().to_numpy()
        ds_slice = xr.Dataset({
            default_array_label:(("time", "lat", "lon"), sub_array)
            })
        ds_slice.to_zarr(out_zarr_path, region={"time":s})
        print(f"Loaded {s}")

    return out_zarr_path

def mp_run_benchmark(args):
    print(f"running benchmark", args["test_type"], args["var_label"])
    return args,run_benchmark(**args)

def run_benchmark(zarr_url, test_type, var_label,
        test_kwargs:dict={}, seed=None, debug=False, dtype_size=4):
    """
    :@param zarr_url: path or remote url
    :@param test_type: determines what kind of subset to extract and time;
        may be 'pixel', 'timestep', 'chunk', or 'multichunk'
    """
    test_settings = {
        "nchunks":2, ## multichunk
        "full_chunk_only":False, ## chunk,multichunk,volume
        "volume_cutoff_range_mb":4000, ## volume
        }
    test_settings.update(test_kwargs)

    assert test_type in ["pixel", "timestep", "chunk", "multichunk", "volume"]
    rng = np.random.default_rng(seed)
    xr_kwargs = [{},{"storage_options":{"anon":True}}][zarr_url[:3]=="s3:"]
    if debug:
        print(f"opening zarr at {zarr_url}")
    t0_init = time.perf_counter()
    arr = xr.open_zarr(
            zarr_url,
            consolidated=False,
            **xr_kwargs,
            )[var_label]
    tf_init = time.perf_counter()
    if debug:
        print(arr)

    if test_type=="pixel":
        ixy = rng.integers(arr.shape[1])
        ixx = rng.integers(arr.shape[2])
        if debug:
            print(f"Extracting pixel ({ixy}, {ixx})")
        t0_load = time.perf_counter()
        sub_out = arr[:,ixy,ixx].load().to_numpy()
        tf_load = time.perf_counter()
        point_count = sub_out.size
        query = (ixy, ixx)

    elif test_type=="timestep":
        ixt = rng.integers(arr.shape[1])
        if debug:
            print(f"Extracting timestep {ixt}")
        t0_load = time.perf_counter()
        sub_out = arr[ixt,:,:].load().to_numpy()
        tf_load = time.perf_counter()
        point_count = sub_out.size
        query = (ixt,)

    elif test_type in ["chunk", "multichunk", "volume"]:
        ## list the chunk boundary indices
        csdict = arr.chunksizes
        cb_time = np.cumsum(np.concatenate([[0], csdict["time"]]))
        cb_lat = np.cumsum(np.concatenate([[0], csdict["lat"]]))
        cb_lon = np.cumsum(np.concatenate([[0], csdict["lon"]]))

        base_shape = [csdict["time"][0], csdict["lat"][0], csdict["lon"][0]]
        ## remove the final chunk from the listing if only full chunks allowed
        if test_settings["full_chunk_only"]:
            if base_shape[0] != csdict["time"][-1]:
                cb_time = cb_time[:-1]
            if base_shape[1] != csdict["lat"][-1]:
                cb_lat = cb_lat[:-1]
            if base_shape[2] != csdict["lon"][-1]:
                cb_lon = cb_lon[:-1]

        if test_type=="chunk":
            ## choose a random chunk
            ixc_0 = rng.integers(cb_time.size-1)
            ixc_1 = rng.integers(cb_lat.size-1)
            ixc_2 = rng.integers(cb_lon.size-1)
            query = (tuple(map(int, (ixc_0, ixc_1, ixc_2))),)

            cslc = [
                slice(cb_time[ixc_0], cb_time[ixc_0+1]),
                slice(cb_lat[ixc_1], cb_lat[ixc_1+1]),
                slice(cb_lon[ixc_2], cb_lon[ixc_2+1]),
                ]
            if debug:
                print(f"Extracting chunk {cslc}")
            ## record time to reference and download subset
            t0_load = time.perf_counter()
            sub_out = arr[*cslc].load().to_numpy()
            tf_load = time.perf_counter()
            point_count = sub_out.size

        if test_type=="volume":
            csize = np.prod(base_shape) * dtype_size / 1000**2
            assert test_settings["volume_cutoff_range_mb"][0] \
                < test_settings["volume_cutoff_range_mb"][1]
            assert csize < test_settings["volume_cutoff_range_mb"][0], \
                "default chunk size is too large given volume_cutoff_range_mb"
            #print(cb_time.size, cb_lat.size, cb_lon.size)
            vrange = np.stack([
                sorted(rng.choice(cb_time.size, 2, replace=False)),
                sorted(rng.choice(cb_lat.size, 2, replace=False)),
                sorted(rng.choice(cb_lon.size, 2, replace=False)),
                ], axis=0)

            ## randomize the cutoff within the requested bounds
            cutoff_size = rng.random() \
                    * np.diff(test_settings["volume_cutoff_range_mb"]) \
                    + test_settings["volume_cutoff_range_mb"][0]

            ## iterate on randomly removing chunks until within the cutoff
            vdiff = np.diff(vrange, axis=1)
            vsize = np.prod(vdiff) * csize
            trunc_last = True
            while vsize > cutoff_size:
                ## weight probability of truncating axis by size in num chunks
                probs = np.squeeze((vdiff-1)/np.sum(vdiff-1))
                trunc_ix = rng.choice(3, p=probs)
                vrange[trunc_ix][int(trunc_last)] += [1,-1][int(trunc_last)]
                vdiff = np.diff(vrange, axis=1)
                vsize = np.prod(vdiff) * csize
                trunc_last = not trunc_last

            cslc = [
                slice(cb_time[vrange[0][0]], cb_time[vrange[0][1]]),
                slice(cb_lat[vrange[1][0]], cb_lat[vrange[1][1]]),
                slice(cb_lon[vrange[2][0]], cb_lon[vrange[2][1]]),
                ]

            query = np.stack(np.meshgrid(
                range(vrange[0][0], vrange[0][1]),
                range(vrange[1][0], vrange[1][1]),
                range(vrange[2][0], vrange[2][1]),
                indexing="ij",
                ), axis=-1).reshape(-1,3)
            query = tuple(map(tuple, query.astype(int).tolist()))

            t0_load = time.perf_counter()
            sub_out = arr[*cslc].load().to_numpy()
            tf_load = time.perf_counter()
            point_count = sub_out.size

        if test_type=="multichunk":
            ## choose multiple random chunks w/o replacement
            ixc_0 = rng.integers(0, cb_time.size-1, test_settings["nchunks"])
            ixc_1 = rng.integers(0, cb_lat.size-1, test_settings["nchunks"])
            ixc_2 = rng.integers(0, cb_lon.size-1, test_settings["nchunks"])
            cslcs = [(
                slice(cb_time[ixc_0[i]], cb_time[ixc_0[i]+1]),
                slice(cb_lat[ixc_1[i]], cb_lat[ixc_1[i]+1]),
                slice(cb_lon[ixc_2[i]], cb_lon[ixc_2[i]+1]),
                ) for i in range(test_settings["nchunks"])
                ]
            query = tuple([
                tuple(map(int, (ixc_0[i], ixc_1[i], ixc_2[i])))
                for i in range(test_settings["nchunks"])
                ])
            if debug:
                print(f"Extracting chunks: {cslcs}")
            ## record the total time and number of points downloaded
            point_count = 0
            t0_load = time.perf_counter()
            for cslc in cslcs:
                sub_out = arr[*cslc].load().to_numpy()
                point_count += sub_out.size
            tf_load = time.perf_counter()

    return {
        "point_count":int(point_count),
        "time_start":t0_init,
        "dt_init":tf_init-t0_init,
        "dt_load":tf_load-t0_load,
        "query":query,
        }

def collect_benchmark_result(cur_args, cur_results, all_results):
    """ Adds new results from run_benchmark to a dict of old results """
    if cur_args["test_type"] not in all_results.keys():
        all_results[cur_args["test_type"]] = {}
    if cur_args["var_label"] not in all_results[cur_args["test_type"]].keys():
        all_results[cur_args["test_type"]][cur_args["var_label"]] = {
            "point_count":[],
            "time_start":[],
            "dt_init":[],
            "dt_load":[],
            "test_kwargs":[],
            "query":[],
            "seed":[],
            }
    tmp_res_dict = {
        **cur_results,
        "test_kwargs":cur_args["test_kwargs"],
        "seed":cur_args["seed"],
        }
    for k,v in tmp_res_dict.items():
        all_results[cur_args["test_type"]][cur_args["var_label"]][k].append(v)
    print("Finished benchmark:",
        cur_args["test_type"],
        cur_args["var_label"],
        len(all_results[cur_args["test_type"]][cur_args["var_label"]]["seed"]),
        )
    return all_results


def get_chunk_size_combos(time_sizes, lat_sizes, lon_sizes,
        dtype_bytesize=4, chunk_size_bounds_mb=None, area_aspect_bounds=None,
        exclude_permutations=True):
    """
    """
    ## enumerate all chunk size combos
    layouts = np.stack(np.meshgrid(
        time_sizes, lat_sizes, lon_sizes, indexing="ij"
        ), axis=0).reshape(3, -1)

    ## resrtrict by chunk size per chunk_size_bounds_mb
    if not chunk_size_bounds_mb is None:
        chunk_sizes_mb = np.prod(layouts, axis=0) * dtype_bytesize / 1000**2
        m_size = (chunk_sizes_mb > chunk_size_bounds_mb[0]) \
                & (chunk_sizes_mb < chunk_size_bounds_mb[1])
        layouts = layouts[:,m_size]

    ## restrict by area aspect ratio via area_aspect_bounds
    if not area_aspect_bounds is None:
        chunk_area_asp = layouts[1] / layouts[2]
        m_asp = (chunk_area_asp > area_aspect_bounds[0]) \
                & (chunk_area_asp < area_aspect_bounds[1])
        layouts = layouts[:,m_asp]

    ## rule out chunks with dimensions that are permutations of other configs
    if exclude_permutations:
        a,ixs,cnts = np.unique(
                np.sort(layouts, axis=0),
                axis=1,
                return_index=True,
                return_counts=True,
                )
        layouts = layouts[:,ixs]

    return list(map(tuple, layouts.T.tolist()))

if __name__=="__main__":

    ## switchboard
    print_table = False
    download_new_subset = False
    load_chunk_variations = False
    run_benchmarks = True

    ## table printing settings
    full_grid_shape = (6500, 11700, 8400)
    dtype_size_bytes = 4

    ## subset zarr storage settings
    out_zarr_path = Path("/rgroup/airmettle/nldas3_chunk_benchmarking.zarr")
    sub_time_slice=slice("2014-01-01", "2018-12-31")
    sub_lat_slice=slice(2500, 3500)
    sub_lon_slice=slice(7200, 9000)
    times_per_load=64
    sub_acquire_var="Tair"
    sub_out_chunks=(1, 500, 900)
    sub_out_dtype = np.float32

    ## benchmark settings
    run_benchmark_tests = [
            #"pixel", "timestep",
            "chunk", "multichunk", "volume",
            ]
    benchmark_iterations = 64
    random_seed = 200007221750
    zarr_url = "s3://nasa-waterinsight/.test/nldas3_chunk_benchmarking.zarr"
    #zarr_url = out_zarr_path.as_posix()
    benchmark_var = "Tair"
    multi_chunk_range = (2, 32)
    volume_cutoff_range_mb = (200, 4000)
    full_chunk_only = True
    #json_out_path = Path("nldas3_chunk_bench_results_local.json")
    #json_out_path = Path("data/nldas3_chunk_bench_results.json")
    json_out_path = Path("data/nldas3_chunk_bench_results_fullchunk_3.json")
    save_results_frequency = 32

    nprocs = 24 ## number of concurrent processes for downloading
    #nprocs = 1

    ## run default
    chunking_cands = [ChunkConfig(500, 900, 1)]

    ## daily
    '''
    chunking_cands = [
        ChunkConfig(1,500,900), ChunkConfig(1,325,650),
        ChunkConfig(1,250,450), ChunkConfig(1, 500, 300),
        ChunkConfig(1, 260, 260), ChunkConfig(1, 130, 260),

        ChunkConfig(8, 500, 900), ChunkConfig(8, 325, 650),
        ChunkConfig(8, 250, 450), ChunkConfig(8, 500, 300),
        ChunkConfig(8, 260, 260), ChunkConfig(8, 130, 260),

        ChunkConfig(16, 500, 900), ChunkConfig(16, 325, 650),
        ChunkConfig(16, 250, 450), ChunkConfig(16, 500, 300),
        ChunkConfig(16, 260, 260), ChunkConfig(16, 130, 260),

        ChunkConfig(24, 500, 900), ChunkConfig(24, 325, 650),
        ChunkConfig(24, 250, 450), ChunkConfig(24, 500, 300),
        ChunkConfig(24, 260, 260), ChunkConfig(24, 130, 260),

        ChunkConfig(32, 500, 900), ChunkConfig(32, 325, 650),
        ChunkConfig(32, 250, 450), ChunkConfig(32, 500, 300),
        ChunkConfig(32, 260, 260), ChunkConfig(32, 130, 260),
        ]
    '''

    ## generate ChunkConfigs
    '''
    #time_sizes = [1, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    #latlon_common = [1, 4, 10, 25, 50, 65, 100, 130, 260, 325, 650]
    time_sizes = [4, 8, 16, 24, 32, 48, 96, 128, 256]
    latlon_common = [25, 65, 100, 130, 260, 325, 650]
    layouts = get_chunk_size_combos(
        time_sizes=time_sizes,
        lat_sizes=[*latlon_common, 500],
        lon_sizes=[*latlon_common, 180, 450],
        dtype_bytesize=4,
        chunk_size_bounds_mb=(0.1, 32.),
        area_aspect_bounds=(1/5, 5.),
        exclude_permutations=True,
        )
    chunking_cands = list(filter(
            lambda cc:cc not in chunking_cands,
            [ChunkConfig(*lt) for lt in layouts],
            ))
    '''

    ## derive ChunkConfigs from what's available on the s3 bucket
    #'''
    ds = xr.open_zarr(zarr_url, consolidated=False)
    chunking_cands = [
        ChunkConfig(*map(int, vl.split("-")[-1].split(".")))
        for vl in ds.variables.keys()
        if vl not in ("lat", "lon", "time")
        ]
    print(list(ds.variables.keys()))
    #'''

    """ ------------------( END NORMAL CONFIGURATION )-----------------  """

    if print_table:
        ## print a markdown table of the chunk configuration information
        col_labels = [
            "lat", "lon", "time", "size/chunk (MB)",
            "N<sub>t</sub>", "N<sub>xy</sub>",
            "S<sub>t</sub>/S<sub>xy</sub> (x 10<sup>3</sup>)"
            #"S<sub>t</sub> S<sub>xy</sub><sup>-1/2</sup> "
            ]
        print(" | ".join(col_labels))
        print(" | ".join(["---" for _ in range(len(col_labels))]))
        for cc in chunking_cands:
            nct,ncx,ncy = cc.chunk_layout(*full_grid_shape)
            sct,scx,scy = cc.as_tuple()
            print(" | ".join(map(str, [
                *tuple(cc.cvec),
                cc.chunk_size_mb(dtype_size_bytes),
                nct, ncy*ncx,
                f"{sct/(scy*scx)*1000:.3f}"
                #f"{sct*(scy*scx)**(-0.5):.4f}"
                ])))

    """ initialize the zarr store and load the initial subset """

    if download_new_subset:
        assert not out_zarr_path.exists()
        nldas3_subset_to_zarr(
            time_slice=sub_time_slice,
            lat_slice=sub_lat_slice,
            lon_slice=sub_lon_slice,
            times_per_load=times_per_load,
            acquire_var=sub_acquire_var,
            out_zarr_path=out_zarr_path,
            out_chunks=sub_out_chunks,
            out_dtype=sub_out_dtype,
            )

    """ update the zarr store with variations on chunk configuration """

    if load_chunk_variations:
        ds = xr.open_zarr(out_zarr_path)
        x = ds["Tair-1.500.900"].load().to_numpy()
        for cc in chunking_cands:
            clayout = tuple(cc)
            tmp_label = sub_acquire_var + "-" + ".".join(map(str,clayout))
            tmp_array = dask.array.from_array(x, chunks=clayout)
            ds = xr.Dataset({tmp_label:(("time", "lat", "lon"), tmp_array)})
            ds.to_zarr(out_zarr_path, mode="a", compute=True)
            print(f"Loaded {clayout}")

    """ run benchmarks and store results """

    if run_benchmarks:
        ## load any existing results
        if json_out_path.exists():
            results = json.load(json_out_path.open("r"))
        else:
            results = {}

        rng = np.random.default_rng(random_seed)
        bench_runs = [
            (cc.as_tuple(),tl)
            for cc in chunking_cands
            for tl in run_benchmark_tests
            for _ in range(benchmark_iterations)
            ]
        rng.shuffle(bench_runs)
        pprint(bench_runs)

        default_test_kwargs = {"full_chunk_only":full_chunk_only}
        args = [{
            "zarr_url":zarr_url,
            "test_type":tl,
            "var_label":benchmark_var+"-"+".".join(map(str, cctup)),
            "test_kwargs":{
                "multichunk":{
                    **default_test_kwargs,
                    "nchunks":int(rng.integers(*multi_chunk_range))
                    },
                "volume":{
                    **default_test_kwargs,
                    "volume_cutoff_range_mb":volume_cutoff_range_mb,
                    },
                }.get(tl, default_test_kwargs),
            "seed":random_seed+i,
            "debug":False,
            } for i,(cctup,tl) in enumerate(bench_runs)]

        print(f"Running {len(args)} experiments")
        if nprocs>1:
            with mp.Pool(nprocs) as pool:
                for i,(a,r) in enumerate(pool.imap_unordered(
                        mp_run_benchmark, args)):
                    results = collect_benchmark_result(
                        cur_args=a,
                        cur_results=r,
                        all_results=results,
                        )
                    if i%save_results_frequency==0:
                        json.dump(results, json_out_path.open("w"))
        else:
            for i,(a,r) in enumerate(map(mp_run_benchmark, args)):
                results = collect_benchmark_result(
                    cur_args=a,
                    cur_results=r,
                    all_results=results,
                    )
                if i%save_results_frequency==0:
                    json.dump(results, json_out_path.open("w"))
        json.dump(results, json_out_path.open("w"))

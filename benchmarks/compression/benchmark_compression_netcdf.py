#import os
import time
import netCDF4
import json
import itertools
from multiprocessing import Pool
from datetime import datetime
from pathlib import Path

def mp_benchmark_netcdf(args):
    return args,benchmark_netcdf(**args)

def benchmark_netcdf(src_nc_path:Path, dest_nc_path:Path, params:dict, **_):
    """
    recreate a netCDF file with variations on its compression strategy

    :@param input_path: path to source netCDF file
    :@param params: dict of storage parameters
        - compression (zlib, zstd, bzip2, None)
        - complevel (1-9)
        - shuffle (True/False)
        - significant_digits: (int or None)
        - least_significant_digits (int or None)
        - read_iterations (int)
    :@return: tuple (write_time, file_size, read_times, status)
    """
    ## start write timer
    start_time = time.perf_counter()
    read_times = {}

    try:
        with netCDF4.Dataset(src_nc_path, "r") as src, \
             netCDF4.Dataset(dest_nc_path, "w", format="NETCDF4") as dst:

            ## copy global attributes
            dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})

            ## copy dimensions
            for name, dim in src.dimensions.items():
                dimsize = len(dim) if not dim.isunlimited() else None
                dst.createDimension(name, dimsize)

            ## copy variables
            for name, var in src.variables.items():
                kwargs = {
                    "datatype": var.datatype,
                    "dimensions": var.dimensions,
                    "compression": params.get("compression", "zlib"),
                    "complevel": params.get("complevel", 4),
                    "shuffle": params.get("shuffle", True)
                }

                ## Apply quantization if specified
                if params.get("significant_digits") is not None:
                    kwargs["significant_digits"] = params["significant_digits"]
                if params.get("least_significant_digit") is not None:
                    kwargs["least_significant_digit"] = \
                        params["least_significant_digit"]

                ## Handle _FillValue passed on creation
                if hasattr(var, "_FillValue"):
                    kwargs["fill_value"] = var._FillValue

                ## create this variable
                dst_var = dst.createVariable(name, **kwargs)

                ## copy variable metadata
                attrs = {
                    k: var.getncattr(k)
                    for k in var.ncattrs()
                    if k != "_FillValue"
                }
                dst_var.setncatts(attrs)

                ## copy actual data
                dst_var[:] = var[:]

        write_time = time.perf_counter() - start_time
        file_size = dest_nc_path.stat().st_size

        ## benchmark read performance
        read_iterations = params.get("read_iterations", 5)

        with netCDF4.Dataset(dest_nc_path, "r") as nc:
            for name, var in nc.variables.items():

                ## warm-up read
                _ = var[:]

                ## timed reads
                times = []

                for _ in range(read_iterations):
                    start_read = time.perf_counter()
                    data = var[:]
                    elapsed = time.perf_counter() - start_read
                    times.append(elapsed)
                read_times[name] = times

        status = (True, None)

    except Exception as e:
        write_time = None
        file_size = None
        read_times = {}
        status = (False, str(e))

    return write_time, file_size, read_times, status

if __name__=="__main__":
    src_dir = Path("/discover/nobackup/projects/eis_nldas3/DATA/DA/daily/")
    src_template = "{YYYYmm}/NLDAS_DA0010_D.A{YYYYmmdd}.030.beta.nc"
    out_json_path = Path(
        "/discover/nobackup/mtdodson/nldas3/" + \
        "comp_bench_1.json"
        )
    out_dir = Path("/discover/nobackup/mtdodson/nldas3/tmp_comp_bench_1")

    dates = [
        #datetime(2012, 10, 13),
        #datetime(2019, 5, 21),
        #datetime(2001, 9, 8),
        #datetime(2013, 2, 14),
        #datetime(2016, 7, 29),
        #datetime(2007, 12, 6),
        #datetime(2005, 6, 3),
        #datetime(2020, 3, 27),
        datetime(2011, 5, 14),
        datetime(2005, 12, 3),
        datetime(2019, 8, 18),
        datetime(2003, 2, 26),
        ]

    ## define parameter space to test
    algorithms = ["zlib", "zstd"]
    #levels = [1, 3, 5, 7, 9]
    levels = [ 3, 5, 7]
    shuffles = [True, False]
    quantization_options = [
        {"significant_digits": None, "least_significant_digit": None},
        {"significant_digits": 6, "least_significant_digit": None},
        {"significant_digits": None, "least_significant_digit": 5},
        ]

    ## number of times to test reading variables to get average
    read_iterations = 5

    nworkers = 16

    ## generate parameter combinations
    args = []
    combos = itertools.product(
        algorithms, levels, shuffles, quantization_options, dates
        )
    for cid,(algo,lvl,shuf,quant,date) in enumerate(combos):
        cs = f"{algo}:{lvl}"
        if shuf:
            cs += "_shuf"
        if quant.get("significant_digits"):
            cs += f"_sd:{quant["significant_digits"]}"
        if quant.get("least_significant_digit"):
            cs += f"_ld:{quant["least_significant_digit"]}"
        dstr = date.strftime("%Y%m%d")
        args.append({
            "src_nc_path":src_dir.joinpath(src_template.format(
                YYYYmm=date.strftime("%Y%m"),
                YYYYmmdd=date.strftime("%Y%m%d"),
                )),
            "dest_nc_path":out_dir.joinpath(f"nldas3_da_{dstr}_{cid}.nc"),
            "params":{
                "compression":algo,
                "complevel":lvl,
                "shuffle":shuf,
                "significant_digits":quant["significant_digits"],
                "least_significant_digit":quant["least_significant_digit"]
                },
            "datetime":date,
            "combo_string":cs,
            })

    print(f"Starting benchmark over {len(args)} configs...")

    results = {}
    with Pool(nworkers) as pool:
        for a,r in pool.imap_unordered(mp_benchmark_netcdf, args):
            src_fsize = a['src_nc_path'].stat().st_size
            cs = a["combo_string"]
            pout = a["dest_nc_path"]
            dt = a["datetime"]
            wtime,fsize,rtimes,status = r
            if not status[0]:
                print(f"Failed {cs}: {status[1]}")
                continue
            if not cs in results.keys():
                results[cs] = {
                    "params":a["params"],
                    "write_time":[],
                    "comp_ratio":[],
                    "dates":[],
                    "read_times":{
                        k:[] for k in rtimes.keys()
                        },
                    }
            results[cs]["write_time"].append(wtime)
            results[cs]["comp_ratio"].append(src_fsize / fsize)
            results[cs]["dates"].append(dt.strftime("%Y%m%d"))
            for k in rtimes.keys():
                results[cs]["read_times"][k].append(rtimes[k])
            json.dump(results, out_json_path.open("w"))

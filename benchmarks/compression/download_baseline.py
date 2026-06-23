import numpy as np
import xarray as xr
import fsspec
import dask.array as da
import dask.dataframe as dd
import zarr
from datetime import datetime,timedelta
from pprint import pprint
from pathlib import Path

import compression_config as config

if __name__=="__main__":
    out_dir = Path("data/store/")
    nprocs = 6

    extract_regions = [
        "midwest",
        "alaska",
        "desertw",
        "yucatan",
        "southeast",
        "mountainw",
        ]
    compute_regions = [
        "midwest",
        "alaska",
        "desertw",
        "yucatan",
        "southeast",
        "mountainw",
        ]
    time_resolution = "hourly"
    compute_feats = ["Tair", "PSurf", "SWdown", "Wind_E", "Rainf"]
    #compute_feats = ["Tair"]

    zstor_out = out_dir.joinpath(f"compression_{time_resolution}.zarr")
    skip_existing_regions = True
    delete_existing_regions = False
    download_time_chunk_size = 128

    out_shape = config.extract_shape
    region_origins = config.region_origins
    extract_feats = config.extract_feats
    baseline_chunks = config.encoding_args["chunks"]
    baseline_shards = config.encoding_args["shards"]


    start_time = datetime(2015, 1, 1, 0, 0)
    #end_time = start_time + timedelta(hours=out_shape[0]-1)
    nt = out_shape[0]-1
    end_time = start_time + timedelta(
            **[{"days":nt},{"hours":nt}][
                ["daily","hourly"].index(time_resolution)
                ])

    #percentiles = [10, 25, 50, 75, 90]

    """ ------------( end normal configuration )------------ """

    zgrp_out = zarr.open(zstor_out, mode="a")
    if "regions" not in zgrp_out.keys():
        zgrp_out.create_group("regions")

    for rk in extract_regions:
        new_region = rk not in zgrp_out["regions"].keys()
        if not new_region:
            if skip_existing_regions:
                print("skipping existing region:", rk)
                continue
            else:
                if delete_existing_regions:
                    print("deleting existing region:", rk)
                    del zgrp_out["regions"][rk]
                    new_region = True

        if not new_region:
            continue
        ## set up a virtual file system using the kerchunk references
        ref_fs = fsspec.filesystem(
            "reference",
            fo=f"s3://nasa-waterinsight/virtual/nldas3_{time_resolution}.parq",
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

        zgrp_out["regions"].create_group(rk)

        ## get a subset of the virtual dataset for this region
        sub = ds.sel(time=slice(start_time,end_time))
        sub = sub.isel(
            lat=slice(
                region_origins[rk][0],
                region_origins[rk][0]+out_shape[1]
                ),
            lon=slice(
                region_origins[rk][1],
                region_origins[rk][1]+out_shape[2]
                ),
            )

        ## load the coordinate values
        times = sub.time.load().to_numpy()
        lats = sub.lat.load().to_numpy()
        lons = sub.lon.load().to_numpy()

        zreg = zgrp_out[f"/regions/{rk}"]
        zreg.create_array(name="time", data=times)
        zreg.create_array(name="lat", data=lats)
        zreg.create_array(name="lon", data=lons)

        zreg.create_group("baseline")
        tslcs = [
            slice(i, min(i+download_time_chunk_size, out_shape[0]))
            for i in range(0, out_shape[0], download_time_chunk_size)
            ]

        ## declare an uncompressed baseline data array for each variable
        for fk in extract_feats:
            zreg["baseline"].create_array(
                fk,
                shape=out_shape,
                dtype=np.float32,
                chunks=baseline_chunks,
                shards=baseline_shards,
                filters=[],
                compressors=[],
                )
            ## iteratively load the baseline from the s3 bucket
            print(f"downloading {fk}", out_shape)

            for ts in tslcs:
                zreg["baseline"][fk][ts] = sub[fk].isel(
                        time=ts
                        ).load().to_numpy()

    baseline_stats = {}
    for rk in compute_regions:
        ## use dask to calculate the baseline statistics
        print(f"computing bulk statistics")
        baseline_stats[rk] = {}
        zreg = zarr.open(zstor_out, mode="r", path=f"/regions/{rk}/baseline")
        for fk in compute_feats:
            baseline_stats[rk][fk] = {}
            x = da.from_zarr(zreg[fk]).astype(np.float32)
            x = da.ma.masked_invalid(x)
            #pctl = dd.from_dask_array(
            #    x.ravel(), columns=["v"],
            #    )["v"].quantile(
            #        q=[p/100 for p in percentiles],
            #        method="tdigest",
            #        )
            stats = {"min":x.min(), "max":x.max(),
                    "mean":x.mean(), "std":x.std(),
                    "count":x.size,
                    #"pctl":pctl,
                    }
            stats = da.compute(stats)[0]
            #pctl = stats["pctl"]
            #print(pctl)
            #del stats["pctl"]
            #print(stats)
            #stats.update({f"p{p}":v for p,v in zip(percentiles, pctl)})
            baseline_stats[rk][fk] = {k:float(v) for k,v in stats.items()}

    print(baseline_stats)
    zgrp_out.attrs.update({"baseline_stats":baseline_stats})
    print("finished")

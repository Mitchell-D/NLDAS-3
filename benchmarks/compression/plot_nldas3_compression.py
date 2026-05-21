import dask
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import multiprocessing as mp
from pathlib import Path

if __name__=="__main__":
    #store_dir = Path("data/comp-tests")
    store_dir = Path("/rstor/mdodson/nldas3/comp-tests")
    nprocs = 8

    stores = []
    for p in store_dir.iterdir():
        *_,cl,vl,dl = p.stem.split("_")
        stores.append((p, cl, vl, dl))
        ds = xr.open_dataset(p)
        if dl=="int16":
            print(cl, vl, dl, ds.min(), ds.max())

    #with mp.Pool(nprocs) as pool:
    #    for a,r in pool.imap_unordered(args, )

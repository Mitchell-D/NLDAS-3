import numpy as np
import math
import json
from pathlib import Path

from ChunkConfig import ChunkConfig,calculate_chunk_intersections
from plotting import plot_colored_lines

if __name__=="__main__":
    grid_shape = (8400, 6500, 11700)
    fig_dir = Path("figures")
    dtype_size = 4
    chunk_sizes_mb = np.arange(1, 81, 5) / 2
    time_sizes = np.arange(1, 513, 8)
    volume_shape = (365, 256, 256)
    sub_shapes = [
        (1, *grid_shape[1:]),
        (grid_shape[0], 1, 1),
        (365, 256, 256),
        (365*5, 256, 256),
        (31, 1000, 1800),
        (14, 2400, 6000),
        ]

    npoints = chunk_sizes_mb * 1000**2 / dtype_size

    intersections = {}
    for i in range(npoints.size):
        Smb = chunk_sizes_mb[i]
        C = npoints[i] ## chunk size in points
        intersections[Smb] = {
            "aspect":[],
            "ccounts":{k:[] for k in sub_shapes},
            }
        for Ct in time_sizes:
            ## square spatial side length
            Cxy = int(math.floor((C/Ct)**(1/2)))
            #A = Ct**(3/2) * C**(-1/2)
            #A = Ct/Cxy
            A = int(Ct)
            intersections[Smb]["aspect"].append(A) ## aspect ratio
            for ss in sub_shapes:
                ## single pixel column
                intersections[Smb]["ccounts"][ss].append(
                    calculate_chunk_intersections(
                        grid_shape=grid_shape,
                        chunk_shape=(Ct, Cxy, Cxy),
                        subset_shape=ss,
                        product=True,
                        )[-1]
                    )

    for ss in sub_shapes:
        cstr = "-".join(map(str, ss))
        fig_path = fig_dir.joinpath(f"chunk-geom_count_{cstr}.png")
        plot_colored_lines(
            domain_lines=[
                intersections[Smb]["aspect"]
                for Smb in chunk_sizes_mb
                ],
            range_lines=[
                intersections[Smb]["ccounts"][ss]
                for Smb in chunk_sizes_mb
                ],
            color_values=chunk_sizes_mb,
            plot_spec={
                "title":f"Chunks to get subset {ss}",
                #"xlabel":f"Aspect ratio Ct (Cx Cy)^(-1/2)",
                "xlabel":f"Timesteps per chunk",
                "ylabel":"Chunk Count",
                "cb_label":"Chunk Size (MB)",
                "cmap":"plasma",
                "yscale":"log"
                },
            fig_path=fig_path,
            )

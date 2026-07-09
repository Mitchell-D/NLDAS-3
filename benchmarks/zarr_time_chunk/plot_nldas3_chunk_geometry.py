import numpy as np
import math
import json
from pathlib import Path

from ChunkConfig import ChunkConfig,calculate_chunk_intersections
from plotting import plot_colored_lines

if __name__=="__main__":
    grid_shape = (8400, 6500, 11700)
    fig_dir = Path("figures/chunk-geom")
    dtype_size = 2
    chunk_sizes_mb = np.arange(1, 81, 5) / 2
    time_sizes = np.arange(1, 513, 8)
    sub_shapes = [
        (31, 1500, 1500),
        (31, 650, 650),
        (31, 64, 64),

        (365, 1500, 1500),
        (365, 650, 650),
        (365, 64, 64),

        (365*5, 1500, 1500),
        (365*5, 650, 650),
        (365*5, 64, 64),

        #(grid_shape[0], 1, 1),
        #(365*5, 128, 128),
        #(2922, 48, 48),
        #(1460, 16, 16),
        #(730, 128, 128),
        #(365, 256, 256),
        #(128, 512, 512),
        #(64, 1024, 1024),
        #(31, 1400, 2000),
        #(14, 2400, 6000),
        #(1, *grid_shape[1:]),
        ]

    npoints = chunk_sizes_mb * 1000**2 / dtype_size

    intersections = {}
    for i in range(npoints.size):
        Smb = chunk_sizes_mb[i]
        C = npoints[i] ## chunk size in points
        intersections[Smb] = {
            "aspect":[],
            "ccounts":{k:[] for k in sub_shapes},
            "ratio":{k:[] for k in sub_shapes},
            "waste":{k:[] for k in sub_shapes},
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
                ninter = calculate_chunk_intersections(
                    grid_shape=grid_shape,
                    chunk_shape=(Ct, Cxy, Cxy),
                    subset_shape=ss,
                    product=True,
                    )[-1]
                intersections[Smb]["ccounts"][ss].append(ninter)
                intersections[Smb]["ratio"][ss].append(
                    np.prod(ss) / (ninter * Cxy**2 * A)
                    )
                intersections[Smb]["waste"][ss].append(
                    ((ninter * Cxy**2 * A) - np.prod(ss)) \
                        * dtype_size / 1000**2
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
                "ylim":[10,10**6],
                "yscale":"log"
                },
            fig_path=fig_path,
            )

        fig_path = fig_dir.joinpath(f"chunk-geom_bytes_{cstr}.png")
        plot_colored_lines(
            domain_lines=[
                intersections[Smb]["aspect"]
                for Smb in chunk_sizes_mb
                ],
            range_lines=[
                np.array(intersections[Smb]["ccounts"][ss]) * Smb
                for Smb in chunk_sizes_mb
                ],
            color_values=chunk_sizes_mb,
            plot_spec={
                "title":f"Megabytes downloaded to get subset {ss}",
                #"xlabel":f"Aspect ratio Ct (Cx Cy)^(-1/2)",
                "xlabel":f"Timesteps per chunk",
                "ylabel":"Total Downloaded Size (MB)",
                "cb_label":"Chunk Size (MB)",
                "cmap":"plasma",
                "ylim":[10,10**6],
                "yscale":"log"
                },
            fig_path=fig_path,
            )

        fig_path = fig_dir.joinpath(f"chunk-geom_ratio_{cstr}.png")
        plot_colored_lines(
            domain_lines=[
                intersections[Smb]["aspect"]
                for Smb in chunk_sizes_mb
                ],
            range_lines=[
                np.array(intersections[Smb]["ratio"][ss])
                for Smb in chunk_sizes_mb
                ],
            color_values=chunk_sizes_mb,
            plot_spec={
                "title":f"Ratio Used vs Total Points for subset {ss}",
                #"xlabel":f"Aspect ratio Ct (Cx Cy)^(-1/2)",
                "xlabel":f"Timesteps per chunk",
                "ylabel":"Ratio of Downloaded Points Used",
                "cb_label":"Chunk Size (MB)",
                "cmap":"plasma",
                "ylim":[0,1],
                #"yscale":"log"
                },
            fig_path=fig_path,
            )

        fig_path = fig_dir.joinpath(f"chunk-geom_waste_{cstr}.png")
        plot_colored_lines(
            domain_lines=[
                intersections[Smb]["aspect"]
                for Smb in chunk_sizes_mb
                ],
            range_lines=[
                np.array(intersections[Smb]["waste"][ss])
                for Smb in chunk_sizes_mb
                ],
            color_values=chunk_sizes_mb,
            plot_spec={
                "title":f"Downloaded data wasted (MB) {ss}",
                #"xlabel":f"Aspect ratio Ct (Cx Cy)^(-1/2)",
                "xlabel":f"Timesteps per chunk",
                "ylabel":"MB downloaded but not used",
                "cb_label":"Chunk Size (MB)",
                "cmap":"plasma",
                #"ylim":[0,1],
                #"yscale":"log"
                },
            fig_path=fig_path,
            )

import numpy as np
from pathlib import Path

from ChunkConfig import ChunkConfig,calculate_chunk_intersections
from plotting import plot_point_cloud_3d

if __name__=="__main__":
    grid_shape = (8400, 6500, 11700)

    time_sizes = [1, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    lat_sizes = [
            1, 4, 10, 25, 50, 65, 100, 130, 260, 325, 650,
            ## good factors of 6500 coprime w/ 11700
            500,
            ]
    lon_sizes = [
            ## good common factors of (6500,11700)
            1, 4, 10, 25, 50, 65, 100, 130, 260, 325, 650,
            ## good factors of 11700 coprime w/ 6500
            180, 450,
            ]
    dtype_bytesize = 4
    chunk_size_bounds_mb = (0.1, 16) ## 100 KB to 16 MB
    area_aspect_bounds = (1/5, 5) ## lat/lon size
    access_patterns = [
        (8400, 1, 1),
        (1, 512, 512),
        (24, 512, 512),
        (1200, 64, 64),
        ]

    layouts = np.stack(np.meshgrid(
        time_sizes, lat_sizes, lon_sizes, indexing="ij"
        ), axis=0)

    chunk_sizes_mb = np.prod(layouts, axis=0) * dtype_bytesize / 1000**2
    chunk_area_asp = layouts[1] / layouts[2]
    m_size = (chunk_sizes_mb > chunk_size_bounds_mb[0]) \
            & (chunk_sizes_mb < chunk_size_bounds_mb[1])
    m_asp = (chunk_area_asp > area_aspect_bounds[0]) \
            & (chunk_area_asp < area_aspect_bounds[1])
    layouts = layouts[:,m_size & m_asp]
    chunk_sizes_mb = chunk_sizes_mb[:, m_size & m_asp]
    chunk_asp = layouts[0] * (layouts[1]*layouts[2]) ** -0.5
    print(chunk_sizes_mb)
    print(layouts)
    print(chunk_sizes_mb.shape)
    print(layouts.shape)

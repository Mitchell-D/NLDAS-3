# chunking notes

Assuming daily data w/ float32 variables over 23 years

lat | lon | time | size/chunk (MB) | N<sub>t</sub> | N<sub>xy</sub> | N<sub>xy</sub>/N<sub>t</sub>
--- | --- | --- | --- | --- | --- | ---

500 | 900 | 1 | 1.8 | 8400 | 169 | 0.020
325 | 650 | 1 | 0.845 | 8400 | 360 | 0.043
500 | 300 | 1 | 0.6 | 8400 | 507 | 0.060
260 | 260 | 1 | 0.2704 | 8400 | 1125 | 0.134
130 | 260 | 1 | 0.1352 | 8400 | 2250 | 0.268

500 | 900 | 8 | 14.4 | 1050 | 169 | 0.161
325 | 650 | 8 | 6.76 | 1050 | 360 | 0.343
500 | 300 | 8 | 4.8 | 1050 | 507 | 0.483
260 | 260 | 8 | 2.1632 | 1050 | 1125 | 1.071
130 | 260 | 8 | 1.0816 | 1050 | 2250 | 2.143

500 | 900 | 24 | 43.2 | 350 | 169 | 0.483
325 | 650 | 24 | 20.28 | 350 | 360 | 1.029
500 | 300 | 24 | 14.4 | 350 | 507 | 1.449
260 | 260 | 24 | 6.4896 | 350 | 1125 | 3.214
130 | 260 | 24 | 3.2448 | 350 | 2250 | 6.429

Testing with 4 original (500, 900) chunks over 4 years, that's
~10.52 GB per run, or 157.8 GB total

## methodology for explicit testing

Want to test the following reads from the s3 bucket:

1. several random single chunks
2. single pixel column
3. single time slice

I don't think it's worth selecting random small subsets since the
performance will be highly sensitive to chunk boundaries, and as
such results will be predictable from the above measurements and the
number of chunks intersected.

Use chunks (6-7, 9-10) (2500-3500, 7200-9000)

visualize with plot of px/sec on y axis and Nxy/Nt on x axis for
each configuration. Separate plots for each access method.

Keep in mind that the chunk-based retrievals are the most
untrustworthy since they include partial chunks that won't be
present in the final dataset (since this subset's size doesn't
necessarily have the chunk size as a factor) and will be affected
by the same latency overhead as all others. use the

**visualizations**

- scatterplot x: `time_start`, y: `dt_init`
  per test type, color by cache configuration

- scatterplot x: `time_start+dt_init`, y: `dt_load`
  per test type, color by cache configuration

- scatterplot per test type x:`N_xy/N_t` y: median points/sec w/ IQR

- nested bar plot grouped by chunk configuration. each group has
  bar for median points/sec with IQR of each test

- overall scatterplot and linear regression of number of points
  per request vs `dt_load` to estimate latency overhead

## simulate for full grid:

1. chunks per pixel column
2. chunks per time slice
3. mean chunks per watershed
4. spatial points loaded per watershed
5. number of chunks with/without valid land
   (or mean valid points per chunk)
6. points loaded/points included per watershed

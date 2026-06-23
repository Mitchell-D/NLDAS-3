"""

LZ77: dictionary-based method that shortens byte string motifs

shuffle: compress across the byte or bit positions, which is useful because
    many datasets bounded within a

bitround: drop the configured number of trailing bits from the float mantissa

intnorm: truncate the data between bounds, then rescale and store the
    data as uint16 with a provided number of unique values (resolution).

delta: encode data by storing the absolute value of the first element, then
    the differences between adjacent values.

pcodec: lossless float or int compression, based on
    https://arxiv.org/pdf/2502.06112
"""

extract_shape = (1024, 650, 900) ## 1.2 GB in 2-byte dtype

extract_feats = ["Tair", "PSurf", "SWdown", "Wind_E", "Rainf"]

## features with norm values that depend on the time resolution
conditional_norm_feats = ["Rainf"]

## default normalization bounds
norm_bounds = {
    "Tair":(180, 330),
    "PSurf":(30000,110000),
    "SWdown":(0, 1380),
    "Wind_E":(-25,25),
    "Rainf":{
        "hourly":(0,512),
        "daily":(0,1800),
        },
    }

## custom per-feature variants for setting int norm resolution
## by floating point accuracy for cleaner coordinates.
custom_norm_digits = {
    "Tair":2,
    "PSurf":-1,
    "Wind_E":2,
    "SWdown":0,
    "Rainf":{
        "hourly":2,
        "daily":1.2,
        },
    }

def get_custom_norm_res(feat, time_res):
    d = custom_norm_digits[feat]
    b = norm_bounds[feat]
    if feat in conditional_norm_feats:
        d = d[time_res]
        b = b[time_res]
    return int(10**d * (b[1]-b[0])) + 1

region_origins = {
    "midwest": (3500, 7200),
    "alaska": (5500, 900),
    "desertw": (2500, 5400),
    "yucatan": (1000, 7200),
    "southeast": (2500, 8100),
    "mountainw": (3500, 4500),
    }

## top-level arguments determining  store structure
encoding_args = {
    "chunks":(32,325,450), ## 7.2 MB in 2-byte dtype
    "shards":(128,650,900), ## 460 MB in 2-byte dtype
    "dtype":"f32",
    }

## options common to all native compressors
compressor_options = {
    ## bytes of data treated as a unit of compression
    "blocksize":0,  ## bytes
    ## compression intensity (higher -> larger cr, slower decode)
    "clevel":5, ## 0-9
    ## align & compress bits or bytes by bit sig across elements
    "shuffle":"noshuffle",  ## or shuffle, bitshuffle
    }

## native bytes-to-bytes codecs
compressors = {
    ## LZ77-style; very fast comp/decomp, worse cr
    "blosclz":compressor_options,
    ## LZ77-style, min entropy coding, fast reads
    "lz4":compressor_options,
    ## lz4 with more intense compression; better cr, slightly slower
    "lz4hc":compressor_options,
    ## block-oriented, focused on predictable speed, worse cr
    "snappy":compressor_options,
    ## LZ77 w/ huffman (DEFLATE); decent cr, lower comp/decomp speed.
    "zlib":compressor_options,
    ## dict-based compressor w/ entropy coding; high cr, good speed.
    "zstd":compressor_options,
    }

## non-native array-to-byte codecs (requires numcodecs)
serializers = {
    "pcodec":{
        "level":6, ## 0-12
        "mode_spec":"auto", ## auto or classic; whether to guess structure of data
        ## delta strategy {"auto", "none", "try_consecutive", "try_lookback"}
        "delta_spec":"try_consecutive",
        "paging_spec":"equal_pages_up_to", ## paging strategy
        "delta_encoding_order":1, ## 0-7 or None
        },
    "zfpy":{
        "mode":4,
        "tolerance":-1,
        "rate":-1,
        "precision":-1,
        },
    }

## array-to-array codecs
filters = {
    "bitround":{
        "keepbits":10, ## mantissa bits to keep; float32 has 23.
        },
    "intnorm":{
        "bounds":None, ## 2-tuple of flaots (min, max) must be defined
        "resolution":65535, ## number of ints to spread data across
        "mask_val":65535, ## value reserved for non-finite input data
        "store_dtype":"u2",
        "source_dtype":"f4",
        "offset":0, ## integer offset after converting to resolution
        },
    "delta":{
        "dtype":None, ## must defined at runtime
        "astype":None, ## must defined at runtime
        },
    }

pipeline_config = [
    ## default compression
    ("f4",[("zstd",{})]),
    ("f4",[("lz4hc",{})]),
    ("f2",[("zstd",{})]),
    ("f2",[("lz4hc",{})]),

    ## 4096 resolution integer norm
    ("f4",[("intnorm",{"resolution":4096}) ]),
    ("f4",[("intnorm",{"resolution":4096}), ("zstd", {}) ]),
    ("f4",[
        ("intnorm",{"resolution":4096}),
        ("zstd",{"shuffle":"shuffle"}),
        ]),
    ("f4",[
        ("intnorm",{"resolution":4096}),
        ("zstd",{"shuffle":"bitshuffle"})
        ]),
    ("f4",[
        ("intnorm",{"resolution":4096}),
        ("delta",{"dtype":"u2","astype":"u2"}),
        ("zstd",{"clevel":5,"shuffle":"bitshuffle"})
        ]),
    ("f4",[
        ("intnorm",{"resolution":4096,"store_dtype":"i4"}),
        ("zfpy",{}),
        ("zstd",{"clevel":5,"shuffle":"bitshuffle"})
        ]),
    ("f4",[
        ("intnorm",{"resolution":4096,"store_dtype":"u4"}),
        ("pcodec",{}),
        ("zstd",{"clevel":5,"shuffle":"bitshuffle"})
        ]),

    ## 2048 resolution integer norm
    ("f4",[("intnorm",{"resolution":2048})]),
    ("f4",[
        ("intnorm",{"resolution":2048}),
        ("zstd",{"shuffle":"shuffle"})
        ]),
    ("f4",[
        ("intnorm",{"resolution":2048}),
        ("zstd",{"shuffle":"bitshuffle"})
        ]),
    ("f4",[
        ("intnorm",{"resolution":2048}),
        ("delta",{"dtype":"u2","astype":"u2"}),
        ("zstd",{"clevel":5,"shuffle":"bitshuffle"})
        ]),
    ("f4",[
        ("intnorm",{"resolution":2048,"store_dtype":"i4"}),
        ("zfpy",{}),
        ("zstd",{"clevel":5,"shuffle":"bitshuffle"})
        ]),
    ("f4",[
        ("intnorm",{"resolution":2048,"store_dtype":"u4"}),
        ("pcodec",{}),
        ("zstd",{"clevel":5,"shuffle":"bitshuffle"})
        ]),

    ## custom resolution integer norm
    ("f4",[("intnorm",{"resolution":"custom"})]),
    ("f4",[
        ("intnorm",{"resolution":"custom"}),
        ("zstd",{"shuffle":"shuffle"})
        ]),
    ("f4",[
        ("intnorm",{"resolution":"custom"}),
        ("zstd",{"shuffle":"bitshuffle"})
        ]),
    ("f4",[
        ("intnorm",{"resolution":"custom"}),
        ("delta",{"dtype":"u2","astype":"u2"}),
        ("zstd",{"clevel":5,"shuffle":"bitshuffle"})
        ]),
    ("f4",[
        ("intnorm",{"resolution":"custom","store_dtype":"i4"}),
        ("zfpy",{}),
        ("zstd",{"clevel":5,"shuffle":"bitshuffle"})
        ]),
    ("f4",[
        ("intnorm",{"resolution":"custom","store_dtype":"u4"}),
        ("pcodec",{}),
        ("zstd",{"clevel":5,"shuffle":"bitshuffle"})
        ]),

    ## float truncation
    ("f4",[("bitround",{"keepbits":12})]),
    ("f4",[("bitround",{"keepbits":12}), ("zstd",{"shuffle":"bitshuffle"})]),
    ("f4",[("bitround",{"keepbits":10})]),
    ("f4",[("bitround",{"keepbits":10}), ("zstd",{"shuffle":"bitshuffle"})]),
    ("f4",[("bitround",{"keepbits":9})]),
    ("f4",[("bitround",{"keepbits":9}), ("zstd",{"shuffle":"bitshuffle"})]),
    ("f4",[("bitround",{"keepbits":8})]),
    ("f4",[("bitround",{"keepbits":8}), ("zstd",{"shuffle":"bitshuffle"})]),
    ("f4",[("bitround",{"keepbits":6})]),
    ("f4",[("bitround",{"keepbits":6}), ("zstd",{"shuffle":"bitshuffle"})]),
    ("f4",[
        ("bitround",{"keepbits":12}),
        ("zfpy",{}),
        ("zstd",{"shuffle":"bitshuffle"})
        ]),
    ("f4",[
        ("bitround",{"keepbits":10}),
        ("zfpy",{}),
        ("zstd",{"shuffle":"bitshuffle"})
        ]),
    ("f4",[
        ("bitround",{"keepbits":9}),
        ("zfpy",{}),
        ("zstd",{"shuffle":"bitshuffle"})
        ]),
    ("f4",[
        ("bitround",{"keepbits":8}),
        ("zfpy",{}),
        ("zstd",{"shuffle":"bitshuffle"})
        ]),
    ("f4",[
        ("bitround",{"keepbits":12}),
        ("pcodec",{}),
        ("zstd",{"shuffle":"bitshuffle"})
        ]),
    ("f4",[
        ("bitround",{"keepbits":10}),
        ("pcodec",{}),
        ("zstd",{"shuffle":"bitshuffle"})
        ]),
    ("f4",[
        ("bitround",{"keepbits":9}),
        ("pcodec",{}),
        ("zstd",{"shuffle":"bitshuffle"})
        ]),
    ("f4",[
        ("bitround",{"keepbits":8}),
        ("pcodec",{}),
        ("zstd",{"shuffle":"bitshuffle"})
        ]),
    ("f4",[("bitround",{"keepbits":3})]),
    ("f4",[("bitround",{"keepbits":3}), ("zstd",{"shuffle":"bitshuffle"})]),
    ]



## assemble unique string labels for all configured pipelines
all_labels = []
for dt,pl in pipeline_config:
    labels = [f"dtype:{dt}"]
    for el,args in pl:
        if len(args.keys()) > 0:
            k,v = zip(*sorted(args.items(), key=lambda s:s[0]))
        else:
            v = []
        labels.append(
            f"{el}"+["",f":{','.join(map(str,v))}"][bool(len(v))]
            )
    all_labels.append("_".join(labels))
pipelines = dict(zip(
    all_labels,
    [{"dtype":dt, "pipeline":pl} for dt,pl in pipeline_config],
    ))


## expose a variable with all codecs of any kind included
_f = {k:{**v, "_ctype":"filter"} for k,v in filters.items()}
_s = {k:{**v, "_ctype":"serializer"} for k,v in serializers.items()}
_c = {k:{**v, "_ctype":"compressor"} for k,v in compressors.items()}
default_codecs = {**_f, **_s, **_c}
assert len(default_codecs.keys()) == sum(len(c.keys()) for c in [_f, _s, _c])

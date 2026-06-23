from __future__ import annotations
import numpy as np
from zarr.abc.codec import ArrayArrayCodec
from zarr.core.buffer.core import NDBuffer
from zarr.core.array_spec import ArraySpec
from zarr.dtype import parse_dtype
from dataclasses import replace

class ClipScaleQuantizeCodec(ArrayArrayCodec):
    """
    Custom codec:
    - clips values
    - scales to integer range
    - handles NaNs
    - rounds
    - casts dtype
    """
    codec_id = "clip_scale_quantize_v1"
    codec_name = "clip_scale_quantize_v1"
    def __init__(self, bounds, resolution, source_dtype, offset=0, mask_val=0,
            store_dtype=np.uint16):
        self.min_value,self.max_value = map(float, bounds)
        self.scale_max = float(resolution) - 1
        self.encode_offset = offset
        self.source_dtype = np.dtype(source_dtype)
        self.store_dtype = np.dtype(store_dtype)
        #self.mask_val = np.array(mask_val).astype(self.dtype)
        self.mask_val = self.store_dtype.type(mask_val)
        assert self.max_value > self.min_value

        sdtype_max = np.iinfo(self.store_dtype).max
        sdtype_min = np.iinfo(self.store_dtype).min
        min_enc = -self.encode_offset
        max_enc = self.scale_max - self.encode_offset
        assert sdtype_min <= min_enc
        assert max_enc <= sdtype_max

    @property
    def resolved_dtype(self):
        return self.source_dtype

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        """
        Informs the Zarr pipeline that this codec changes the
        data type from float32 to int16 during encoding.
        """
        if chunk_spec.dtype.to_native_dtype() != np.dtype(self.source_dtype):
            raise ValueError(
                f"Codec expects {self.source_dtype} not {chunk_spec.dtype}"
                )
        return replace(
            chunk_spec,
            dtype=parse_dtype(self.store_dtype, zarr_format=3),
            fill_value=self.mask_val,
            )

    async def _encode_single(self, chunk_array:NDBuffer, spec:ArraySpec):
        if chunk_array is None:
            return None
        if not isinstance(chunk_array, np.ndarray):
            chunk_array = chunk_array.as_numpy_array()
        arr = np.array(chunk_array, dtype=np.float64)
        nan_mask = np.isnan(arr)
        arr = np.clip(arr, self.min_value, self.max_value)
        denom = (self.max_value - self.min_value)
        arr = (arr - self.min_value) / denom
        arr = arr * self.scale_max - self.encode_offset
        arr[nan_mask] = self.mask_val
        arr = np.rint(arr)
        return spec.prototype.nd_buffer.from_ndarray_like(
                arr.astype(self.store_dtype))

    async def _decode_single(self, chunk_array:np.ndarray, spec:ArraySpec):
        if chunk_array is None:
            return None
        if not isinstance(chunk_array, np.ndarray):
            chunk_array = chunk_array.as_numpy_array()
        arr = np.asarray(chunk_array, dtype=np.float64)
        arr[arr == self.mask_val] = np.nan
        arr = (arr+self.encode_offset) / self.scale_max
        arr = arr * (self.max_value - self.min_value)
        arr = arr + self.min_value
        return arr.astype(self.source_dtype)

    def to_dict(self):
        return {
            "name": self.codec_id,
            "configuration": {
                "bounds": [ self.min_value, self.max_value ],
                "resolution": self.scale_max + 1,
                "offset": self.encode_offset,
                "mask_val": int(self.mask_val),
                "source_dtype": self.source_dtype.str,
                "store_dtype": self.store_dtype.str,
                },
            }

    @classmethod
    def from_dict(cls, data):
        cfg = data["configuration"]

        return cls(
            bounds=cfg["bounds"],
            resolution=cfg["resolution"],
            offset=cfg["offset"],
            mask_val=cfg["mask_val"],
            source_dtype=np.dtype(cfg["source_dtype"]),
            store_dtype=np.dtype(cfg["store_dtype"]),
            )

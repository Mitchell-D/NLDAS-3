class ChunkConfig:
    def __init__(self, ntime, nlat, nlon):
        self.ntime = ntime
        self.nlat = nlat
        self.nlon = nlon

    def chunk_size_mb(self, dtype_size=4):
        return self.nlat*self.nlon*self.ntime*dtype_size/1000**2

    def chunk_layout(self, total_times, total_lats, total_lons):
        return (
            (total_times // self.ntime) + int((total_times % self.ntime) != 0),
            (total_lats // self.nlat) + int((total_lats % self.nlat) != 0),
            (total_lons // self.nlon) + int((total_lons % self.nlon) != 0),
            )

    def __str__(self):
        return "ChunkConfig(" + \
            f"ntime={self.ntime} nlat={self.nlat}, nlon={self.nlon})"

    def __repr__(self):
        return str(self)

    def as_tuple(self):
        return (self.ntime, self.nlat, self.nlon)

    def __eq__(self, other):
        if not isinstance(other, ChunkConfig):
            return NotImplemented
        return self.as_tuple() == other.as_tuple()

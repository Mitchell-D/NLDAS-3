import numpy as np
import itertools

class ChunkConfig:
    @staticmethod
    def _int_tuple_or_nested(arg):
        """
        convert iterable of ints or 1-tuple with an iterable of ints into
        a 1d numpy array of the int values
        """
        assert isinstance(arg, (list, tuple, np.ndarray)), type(arg)
        if isinstance(arg, np.ndarray):
            arg = arg.tolist()
        if len(arg) == 1 and isinstance(arg[0], (list, tuple)):
            arg = arg[0]
        elif not isinstance(arg[0], int):
            raise ValueError("argument must be an integer:", arg[0], arg)
        assert all(isinstance(v, int) and v>0 for v in arg), \
                f"Not all arguments are integers: {arg}"
        return np.array(arg)

    def __init__(self, *dim_sizes:tuple):
        self.cvec = ChunkConfig._int_tuple_or_nested(dim_sizes)

    @property
    def size(self):
        return int(np.prod(self.cvec))

    def chunk_size_mb(self, dtype_size=4):
        return np.prod(self.cvec)*dtype_size/1000**2

    def chunk_layout(self, grid_shape:tuple):
        gs = ChunkConfig._int_tuple_or_nested(grid_shape)
        assert gs.size == self.cvec.size, gs.size
        return gs // self.cvec + (gs % self.cvec != 0)

    def gen_chunks(self, grid_shape, return_tuple=False):
        """ lazily generate all chunks """
        gs = ChunkConfig._int_tuple_or_nested(grid_shape)
        assert gs.size == self.cvec.size, gs.size
        dims = [range(d) for d in self.chunk_layout(gs)]
        ## Generate the cartesian product of all dimensions
        for cixs in itertools.product(*dims):
            ## get slice along each axis for each chunk index
            chunk_slice = []
            for dix,cix in enumerate(cixs):
                pix_start = cix*self.cvec[dix]
                pix_end = min((cix+1)*self.cvec[dix], gs[dix])
                if return_tuple:
                    chunk_slice.append((pix_start, pix_end))
                else:
                    chunk_slice.append(slice(pix_start, pix_end))
            yield chunk_slice

    def get_chunks_and_residuals(self, grid_shape):
        """
        returns the number of full chunks and number of residual pixels in
        an additional partial chunk
        """
        return [
            (int(dsize//self.cvec[dix]), int(dsize%self.cvec[dix]))
            for dix,dsize in enumerate(self.cvec)
            ]

    def __str__(self):
        return f"ChunkConfig({self.cvec.tolist()})"

    def __repr__(self):
        return str(self)

    def __iter__(self):
        for v in self.cvec.tolist():
            yield v

    def as_tuple(self):
        return tuple(self)

    def __list__(self):
        return self.cvec.tolist()

    def __eq__(self, other):
        if not isinstance(other, ChunkConfig):
            return NotImplemented
        return tuple(self) == tuple(other)

    def intersections(self, grid_shape, subset_shape,
            offset=None, product=True):
        """ see documentation for calculate_chunk_intersections """
        gs = ChunkConfig._int_tuple_or_nested(grid_shape)
        ss = ChunkConfig._int_tuple_or_nested(subset_shape)
        if not offset is None:
            os = ChunkConfig._int_tuple_or_nested(offset)
        else:
            os = None
        assert gs.size == self.cvec.size, gs.size
        return calculate_chunk_intersections(
                grid_shape=gs,
                chunk_shape=self.cvec,
                subset_shape=ss,
                offset=os,
                product=product,
                )

def _intersections(g, c, s, o=None):
    """
    Calculate chunk intersections along a single dimension. If no offset is
    provided, calculates the minimum, maximum, and average number of
    intersections for any random subset of the given size.

    :@param g: grid size
    :@param c: chunk size
    :@param s: subset size
    :@param o: optional offset

    :@return: N chunks if offset is provided, otherwise (min, max, mean)
        number of chunks over all valid offsets
    """
    ## If offset is provided, explicitly calculate intersections
    if not o is None:
        ## one chunk (origin) plus the number of new chunk boundaries crossed
        ## size must be converted to the index of the last valid point by
        ## subtracting one; offset is assumed to point to the first valid.
        return 1 + (o+s-1)//c - o//c

    ## otherwise, calculate the minimum, maximum, and average number of
    ## intersections of all possible random subsets of size s
    n = (g-(s-1))//c ## full chunk cycles
    r = (g-(s-1))%c ## residual offsets
    if n != 0:
        xper = np.array([
            _intersections(g,c,s,vo)
            for vo in range(c)
            ])
        periodic_sum = np.sum(xper) * n
        periodic_count = xper.size * n
        periodic_min= np.amin(xper)
        periodic_max= np.amax(xper)
    else:
        xper = None
        periodic_sum = 0
        periodic_count = 0
        periodic_min = np.nan
        periodic_max= np.nan
    if r != 0:
        xres = np.array([
            _intersections(g,c,s,vo)
            for vo in range(n*c, n*c+r)
            ])
        residual_sum = np.sum(xres)
        residual_count = xres.size
        residual_min= np.amin(xres)
        residual_max= np.amax(xres)
    else:
        xres = None
        residual_sum = 0
        residual_count = 0
        residual_min = np.nan
        residual_max = np.nan

    #overall_mean = (periodic_sum+residual_sum)/(periodic_count+residual_count)
    return (
        int(np.nanmin([periodic_min, residual_min])),
        int(np.nanmax([periodic_max, residual_max])),
        float((periodic_sum+residual_sum)/(periodic_count+residual_count)),
        )

def calculate_chunk_intersections(
        grid_shape, chunk_shape, subset_shape, offset=None, product=False):
    """
    Calculate and return the minimum, maximum, and average number of chunks
    intersected by any n-dimensional subset of an arbitrary grid having a
    user-defined chunk layout.

    This method can be used to estimate the average number of chunks a user
    will need to load for any given access pattern.

    While the method supports chunks that aren't necessarily factors of their
    dimension, it doesn't consider the size differences between chunks, and
    only reports the total number that are intersected.

    :@param grid_shape: N-tuple of ints indicating the shape of the full domain
    :@param chunk_shape: N-tuple of ints indicating the shape of chunks, which
        don't necessarily need to be factors of the corresponding dimension.
    :@param subset_shape: N-tuple of ints indicating the shape of any subset.
    :@param offset: N-tuple of ints indicating the index offset

    :@return: tuple of (min_chunks, max_chunks, mean_chunks) if no offset is
        provided, or an integer number of chunks if a specific offset is given.
    """
    # Ensure dimensions match
    G = np.array(grid_shape)
    C = np.array(chunk_shape)
    S = np.array(subset_shape)

    assert C.size==G.size, "Number of dims must be uniform"
    assert S.size==G.size, "Number of dims must be uniform"

    ## explicitly calculate the number of intersections if an offset provided
    if not offset is None:
        O = np.array(offset)
        assert O.size==G.size, "Number of dims must be uniform"
        nints = [
            _intersections(G[d], C[d], S[d], O[d])
            for d in range(G.size)
            ]
        if product:
            return np.prod(nints)
        return nints

    ##
    mins,maxs,means = map(np.asarray, zip(*[
        _intersections(G[d], C[d], S[d])
        for d in range(G.size)
        ]))
    if product:
        return int(np.prod(mins)),int(np.prod(maxs)),float(np.prod(means))
    return mins,maxs,means

"""Explicit-layout readers for MFC native restart files; no equation-index guessing."""

from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.io import FortranFile


def record(path, dtype, header_dtype):
    with FortranFile(path, "r", header_dtype=np.dtype(header_dtype)) as stream:
        return stream.read_record(np.dtype(dtype))


class Reader:
    """Read one snapshot, optionally assembling a different simulation decomposition."""

    def __init__(self, config, grid, root):
        self.config, self.grid, self.root = config, grid, Path(root)
        self.dtype = np.dtype(config.get("dtype", "<f8"))
        self.particle_dtype = np.dtype(config.get("particle_dtype", "<f8"))
        self.integer_dtype = np.dtype(config.get("integer_dtype", "<i4"))
        self.header_dtype = config.get("record_header_dtype", "<u4")
        self.format = config.get("format", "shared")
        if self.format not in ("shared", "shards", "serial"):
            raise ValueError("io.format must be shared, shards, or serial")
        if self.dtype.kind != "f" or self.particle_dtype.kind != "f" or self.integer_dtype.kind != "i":
            raise ValueError("Expected floating-point field/particle dtypes and signed integer IDs")
        indices = config["indices"]
        self.indices = [indices["rho"], *indices["momentum"], indices["energy"]]
        if len(self.indices) != 5 or any(int(i) != i or i < 1 or i > config["nvars"] for i in self.indices) or len(set(self.indices)) != 5:
            raise ValueError("Provide five distinct one-based conservative equation indices within nvars")
        if not isinstance(config["nvars"], int) or config["nvars"] < 5:
            raise ValueError("nvars must be an integer >= 5")
        self.indices = [int(i) for i in self.indices]
        self.blocks = config.get("blocks", [{"offset": [0, 0, 0], "shape": list(grid.shape), "rank": 0}])
        coverage = np.zeros(grid.shape, dtype=np.uint8)
        for block in self.blocks:
            offset, shape = np.asarray(block["offset"]), np.asarray(block["shape"])
            if offset.shape != (3,) or shape.shape != (3,) or np.any(shape <= 0) or np.any(offset < 0) or np.any(offset + shape > grid.shape):
                raise ValueError("Invalid block offset/shape")
            if not np.all(offset == offset.astype(int)) or not np.all(shape == shape.astype(int)):
                raise ValueError("Block sizes and offsets must be integers")
            block["offset"], block["shape"] = offset.astype(int).tolist(), shape.astype(int).tolist()
            sl = tuple(slice(int(a), int(a + n)) for a, n in zip(offset, shape))
            if np.any(coverage[sl]):
                raise ValueError("Overlapping restart blocks")
            coverage[sl] = 1
        if not np.all(coverage):
            raise ValueError("Restart blocks do not cover the grid")
        if self.format == "shared" and len(self.blocks) != 1:
            raise ValueError("Shared files use one global block")
        ranks = [block.get("rank", 0) for block in self.blocks]
        if len(set(ranks)) != len(ranks) or any(not isinstance(rank, int) or rank < 0 for rank in ranks):
            raise ValueError("Block ranks must be distinct nonnegative integers")
        if len({self.path(config["flow"], 0, rank, 1) for rank in ranks}) != len(ranks):
            raise ValueError("Each block must have a distinct flow filename; include {rank} in the pattern")

    def path(self, pattern, step, rank=0, var=0):
        return self.root / pattern.format(step=step, rank=rank, var=var)

    def input_paths(self, step):
        paths = set()
        for block in self.blocks:
            rank = block.get("rank", 0)
            for var in self.indices if self.format == "serial" else [0]:
                paths.add(self.path(self.config["flow"], step, rank, var))
        if self.config.get("particle_shards", False):
            for block in self.blocks:
                paths.add(self.path(self.config["particles"], step, block.get("rank", 0)))
        else:
            paths.add(self.path(self.config["particles"], step))
        return sorted(paths)

    def fingerprint(self, step):
        return [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in self.input_paths(step)]

    def particles(self, step):
        if not self.config.get("particle_shards", False):
            path = self.path(self.config["particles"], step)
            if path.stat().st_size % (20 * self.particle_dtype.itemsize):
                raise ValueError(f"Partial IB state record: {path}")
            raw = np.fromfile(path, dtype=self.particle_dtype)
            if raw.size == 0 or raw.size % 20:
                raise ValueError(f"Invalid 20-field IB state file: {path}")
            states = raw.reshape(-1, 20).astype(float)
        else:
            rows = {}
            dtype = np.dtype([("id", self.integer_dtype), ("state", self.particle_dtype, (20,))])
            for block in self.blocks:
                path = self.path(self.config["particles"], step, block.get("rank", 0))
                if (path.stat().st_size - self.integer_dtype.itemsize) % dtype.itemsize:
                    raise ValueError(f"Partial IB state shard record: {path}")
                with open(path, "rb") as stream:
                    count = np.fromfile(stream, dtype=self.integer_dtype, count=1)
                    if count.size != 1 or count[0] < 0:
                        raise ValueError(f"Invalid particle count in {path}")
                    data = np.fromfile(stream, dtype=dtype)
                if len(data) != count[0]:
                    raise ValueError(f"Truncated particle shard: {path}")
                for row in data:
                    key = int(row["id"])
                    if key < 1 or (key in rows and not np.array_equal(rows[key], row["state"])):
                        raise ValueError(f"Invalid or inconsistent replicated particle ID {key}")
                    rows[key] = row["state"].copy()
            if sorted(rows) != list(range(1, len(rows) + 1)) or not rows:
                raise ValueError("Particle IDs must cover 1..N without gaps")
            states = np.array([rows[key] for key in sorted(rows)], dtype=float)
        if not np.all(np.isfinite(states)) or np.any(states[:, 19] <= 0):
            raise ValueError("Invalid particle state; check precision and missing records")
        return states

    def fields(self, step, report=None):
        """Yield rho, three momenta, rho E as float64 global cell arrays."""
        for var in self.indices:
            start = perf_counter()
            if report is not None:
                report(f"reading conservative variable {var}")
            result = np.empty(self.grid.shape)
            for block in self.blocks:
                shape = tuple(block["shape"])
                count = int(np.prod(shape))
                path = self.path(self.config["flow"], step, block.get("rank", 0), var)
                if self.format == "serial":
                    raw = record(path, self.dtype, self.header_dtype)
                    if raw.size != count:
                        raise ValueError(f"Incorrect Fortran record length: {path}")
                else:
                    if path.stat().st_size != count * self.config["nvars"] * self.dtype.itemsize:
                        raise ValueError(f"Incorrect restart size: {path}; check grid, nvars, precision, and downsampling")
                    # Bulk sequential I/O before the F-to-C layout conversion.
                    # A strided copy directly from a Lustre-backed mmap can
                    # otherwise trigger many small, scattered page reads.
                    raw = np.fromfile(path, dtype=self.dtype, count=count, offset=(var - 1) * count * self.dtype.itemsize)
                    if raw.size != count:
                        raise ValueError(f"Truncated restart variable {var}: {path}")
                sl = tuple(slice(a, a + n) for a, n in zip(block["offset"], shape))
                result[sl] = raw.reshape(shape, order="F")
                del raw
            if report is not None:
                report(f"conservative variable {var} read/assembled in {perf_counter() - start:.2f}s")
            yield result

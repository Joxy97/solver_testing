"""Compact, memory-mappable QUBOs. See docs/STORAGE.md for the file format.

Only coefficients and two short identity strings are persisted. Iteration adapters
keep existing solver APIs usable without rebuilding Python lists of interactions.
"""

from __future__ import annotations

import hashlib
import heapq
import math
import os
from pathlib import Path
import shutil
import struct
import tempfile

import numpy as np


MAGIC = b"QUBO0001"
HEADER = struct.Struct("<8sBB6xQQQd64s64s80x")
CSR, PACKED = 0, 1
BLOCK = 65_536
TOLERANCE = 1e-12
CONVENTION = "E(x) = offset + sum_i linear[i]*x_i + sum_{i<j} quadratic[i,j]*x_i*x_j"
RECORD = np.dtype([("i", "<u8"), ("j", "<u8"), ("v", "<f8")])


def pair_start(n: int, row: int) -> int:
    return row * (2 * n - row - 1) // 2


def _aligned(size: int) -> int:
    return (size + 7) // 8 * 8


def _text(value: str) -> bytes:
    encoded = value.encode("utf-8")
    if len(encoded) > 63 or b"\0" in encoded:
        raise ValueError("QUBO identity fields must contain at most 63 UTF-8 bytes")
    return encoded


def _copy_array(handle, values) -> None:
    for start in range(0, len(values), BLOCK):
        values[start : start + BLOCK].tofile(handle)


class LinearTerms:
    def __init__(self, values, count=None):
        self.values = values
        self.count = count

    def __len__(self):
        if self.count is None:
            self.count = sum(np.count_nonzero(block) for _, block in self.blocks())
        return int(self.count)

    def blocks(self, size=BLOCK):
        for start in range(0, len(self.values), size):
            yield start, self.values[start : start + size]

    def __iter__(self):
        for start, block in self.blocks():
            for index in np.flatnonzero(block):
                yield [start + int(index), float(block[index])]


class QuadraticTerms:
    def __init__(self, n, layout, values, count, indptr=None, indices=None):
        self.n, self.layout, self.values, self.count = n, layout, values, count
        self.indptr, self.indices = indptr, indices

    def __len__(self):
        return self.count

    def blocks(self, size=BLOCK):
        """Yield bounded (row indices, column indices, coefficients) arrays."""
        if self.layout == CSR:
            for start in range(0, self.count, size):
                end = min(start + size, self.count)
                positions = np.arange(start, end, dtype=np.uint64)
                rows = np.searchsorted(self.indptr, positions, side="right") - 1
                yield rows, self.indices[start:end], self.values[start:end]
        else:
            # Row-wise slices avoid allocating an O(n^2) array of triangle indices.
            for row in range(self.n - 1):
                begin, end = pair_start(self.n, row), pair_start(self.n, row + 1)
                for start in range(begin, end, size):
                    values = self.values[start : min(start + size, end)]
                    columns = np.flatnonzero(values)
                    if len(columns):
                        yield (
                            np.full(len(columns), row, dtype=np.int64),
                            columns + row + 1 + start - begin,
                            values[columns],
                        )

    def __iter__(self):
        for rows, columns, values in self.blocks():
            for row, column, value in zip(rows, columns, values):
                yield [int(row), int(column), float(value)]


class MappedQubo(dict):
    """A QUBO mapping whose coefficient payload stays backed by its file.

    Keep this object alive while using its array views. close() releases all views;
    callers must not retain arrays after closing their owner.
    """

    def __init__(self, path, *, temporary=None):
        self.path = Path(path)
        self._temporary = temporary
        self._maps = []
        with self.path.open("rb") as handle:
            raw = handle.read(HEADER.size)
        if len(raw) != HEADER.size:
            raise ValueError("Truncated QUBO header")
        magic, layout, width, n, count, linear_count, offset, kind, identity = HEADER.unpack(raw)
        if magic != MAGIC or layout not in (CSR, PACKED) or width not in (4, 8):
            raise ValueError("Unsupported QUBO format, layout, or index width")
        if n < 1 or n > np.iinfo(np.intp).max or count > n * (n - 1) // 2 or linear_count > n:
            raise ValueError("Invalid QUBO dimensions or term counts")
        if width == 4 and n - 1 > np.iinfo(np.uint32).max:
            raise ValueError("QUBO indices do not fit the declared type")
        if not math.isfinite(offset):
            raise ValueError("Non-finite QUBO offset")
        linear_end = HEADER.size + 8 * n
        if layout == CSR:
            index_start = linear_end + 8 * (n + 1)
            value_start = _aligned(index_start + width * count)
            expected = value_start + 8 * count
        else:
            value_start = linear_end
            expected = value_start + 8 * (n * (n - 1) // 2)
        if expected != self.path.stat().st_size:
            raise ValueError("QUBO file size does not match its header")
        try:
            self.problem_type = kind.split(b"\0", 1)[0].decode("utf-8") or "qubo"
            self.instance_id = identity.split(b"\0", 1)[0].decode("utf-8") or self.path.stem
            self.layout = layout
            linear = self._map("<f8", n, HEADER.size)
            if layout == CSR:
                indptr = self._map("<u8", n + 1, linear_end)
                indices = self._map(f"<u{width}", count, index_start)
                values = self._map("<f8", count, value_start)
                if indptr[0] != 0 or indptr[-1] != count:
                    raise ValueError("Invalid CSR row pointers")
                for start in range(0, n, BLOCK):
                    chunk = indptr[start : min(n, start + BLOCK) + 1]
                    if np.any(chunk[1:] < chunk[:-1]) or np.any(chunk > count):
                        raise ValueError("Invalid CSR row pointers")
                quadratic = QuadraticTerms(n, layout, values, count, indptr, indices)
            else:
                values = self._map("<f8", n * (n - 1) // 2, value_start)
                quadratic = QuadraticTerms(n, layout, values, count)
            super().__init__(num_variables=int(n), offset=offset, convention=CONVENTION,
                             linear=LinearTerms(linear, linear_count), quadratic=quadratic)
        except BaseException:
            self.close()
            raise

    def _map(self, dtype, count, offset):
        if count == 0:
            return np.empty(0, dtype=dtype)
        array = np.memmap(self.path, dtype=dtype, mode="r", offset=offset, shape=(count,))
        self._maps.append(array)
        return array

    def close(self):
        self.clear()
        for array in self._maps:
            array._mmap.close()
        self._maps.clear()
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()


def linear_values(qubo, *, copy=False, dtype=np.float64):
    terms = qubo["linear"]
    if isinstance(terms, LinearTerms):
        return np.array(terms.values, dtype=dtype, copy=True) if copy else np.asarray(terms.values, dtype=dtype)
    values = np.zeros(qubo["num_variables"], dtype=dtype)
    for index, value in terms:
        values[index] += value
    return values


def quadratic_blocks(qubo, size=BLOCK):
    terms = qubo["quadratic"]
    if isinstance(terms, QuadraticTerms):
        yield from terms.blocks(size)
    else:
        # Legacy JSON is already materialized; conversion scratch remains bounded.
        for start in range(0, len(terms), size):
            chunk = terms[start : start + size]
            yield (np.fromiter((x[0] for x in chunk), dtype=np.int64, count=len(chunk)),
                   np.fromiter((x[1] for x in chunk), dtype=np.int64, count=len(chunk)),
                   np.fromiter((x[2] for x in chunk), dtype=np.float64, count=len(chunk)))


def coefficient_blocks(qubo):
    linear = linear_values(qubo)
    for start in range(0, len(linear), BLOCK):
        block = linear[start : start + BLOCK]
        yield block[block != 0]
    terms = qubo["quadratic"]
    if isinstance(terms, QuadraticTerms):
        for start in range(0, len(terms.values), BLOCK):
            block = terms.values[start : start + BLOCK]
            yield block[block != 0]
    else:
        for _, _, values in quadratic_blocks(qubo):
            yield values


def quadratic_arrays(qubo):
    """Allocate only the numeric COO arrays required by a solver backend."""
    count = len(qubo["quadratic"])
    first, second = np.empty(count, dtype=np.int64), np.empty(count, dtype=np.int64)
    terms = qubo["quadratic"]
    direct = isinstance(terms, QuadraticTerms) and terms.layout == CSR
    values = terms.values if direct else np.empty(count, dtype=np.float64)
    cursor = 0
    for rows, columns, block in quadratic_blocks(qubo):
        end = cursor + len(block)
        first[cursor:end], second[cursor:end] = rows, columns
        if not direct:
            values[cursor:end] = block
        cursor = end
    if cursor != count:
        raise ValueError("QUBO term count does not match its payload")
    return first, second, values


def _header(n, count, linear_count, offset, layout, kind="", identity=""):
    width = 4 if n - 1 <= np.iinfo(np.uint32).max else 8
    return HEADER.pack(MAGIC, layout, width, n, count, linear_count, offset, _text(kind), _text(identity))


def write_sorted(path, n, linear, terms, offset=0.0):
    """Write canonical sorted terms with bounded buffers and O(n) row pointers.

    Spool CSR columns and values while counting rows, then choose the smaller
    payload. Dense output is filled sequentially; no full matrix is allocated.
    """
    path = Path(path)
    width = 4 if n - 1 <= np.iinfo(np.uint32).max else 8
    with tempfile.TemporaryDirectory(prefix="qubo-write-", dir=path.parent) as scratch:
        scratch = Path(scratch)
        pointers = np.lib.format.open_memmap(scratch / "ptr.npy", mode="w+", dtype="<u8", shape=(n + 1,))
        columns = np.empty(BLOCK, dtype=f"<u{width}")
        values = np.empty(BLOCK, dtype="<f8")
        used = count = row = 0
        previous = (-1, -1)
        try:
            with (scratch / "indices").open("wb") as ci, (scratch / "values").open("wb") as cv:
                for i, j, value in terms:
                    i, j, value = int(i), int(j), float(value)
                    if not 0 <= i < j < n or (i, j) <= previous or not math.isfinite(value):
                        raise ValueError("QUBO terms must be finite, unique, sorted upper-triangular pairs")
                    previous = (i, j)
                    if abs(value) <= TOLERANCE:
                        continue
                    while row <= i:
                        pointers[row] = count
                        row += 1
                    columns[used], values[used] = j, value
                    used += 1
                    count += 1
                    if used == BLOCK:
                        columns.tofile(ci)
                        values.tofile(cv)
                        used = 0
                columns[:used].tofile(ci)
                values[:used].tofile(cv)
            pointers[row:] = count
            pairs = n * (n - 1) // 2
            layout = PACKED if 8 * pairs <= 8 * (n + 1) + (width + 8) * count else CSR
            linear_count = sum(np.count_nonzero(linear[start : start + BLOCK]) for start in range(0, n, BLOCK))
            with path.open("wb") as handle:
                handle.write(_header(n, count, int(linear_count), offset, layout))
                _copy_array(handle, linear)
                if layout == CSR:
                    _copy_array(handle, pointers)
                    with (scratch / "indices").open("rb") as source:
                        shutil.copyfileobj(source, handle, BLOCK * width)
                    handle.write(b"\0" * (_aligned(handle.tell()) - handle.tell()))
                    with (scratch / "values").open("rb") as source:
                        shutil.copyfileobj(source, handle, BLOCK * 8)
                else:
                    with (scratch / "indices").open("rb") as ci, (scratch / "values").open("rb") as cv:
                        for i in range(n - 1):
                            remaining = int(pointers[i + 1] - pointers[i])
                            # Chunk columns independently of row length.
                            next_column = i + 1
                            while remaining:
                                take = min(BLOCK, remaining)
                                js = np.fromfile(ci, dtype=f"<u{width}", count=take)
                                vs = np.fromfile(cv, dtype="<f8", count=take)
                                for start in range(next_column, int(js[-1]) + 1, BLOCK):
                                    end = min(start + BLOCK, int(js[-1]) + 1)
                                    block = np.zeros(end - start, dtype="<f8")
                                    lo, hi = np.searchsorted(js, [start, end])
                                    block[js[lo:hi] - start] = vs[lo:hi]
                                    block.tofile(handle)
                                next_column = int(js[-1]) + 1
                                remaining -= take
                            for start in range(next_column, n, BLOCK):
                                np.zeros(min(BLOCK, n - start), dtype="<f8").tofile(handle)
        finally:
            pointers._mmap.close()


def from_sorted(n, linear, terms, offset=0.0):
    temporary = tempfile.TemporaryDirectory(prefix="qubo-sorted-")
    try:
        path = Path(temporary.name) / "data.qubo"
        write_sorted(path, n, linear, terms, offset)
        return MappedQubo(path, temporary=temporary)
    except BaseException:
        temporary.cleanup()
        raise


def save_qubo(qubo, destination, *, problem_type="qubo", instance_id=""):
    """Atomically save coefficients without materializing a serialization buffer."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        if isinstance(qubo, MappedQubo):
            if destination.resolve() == qubo.path.resolve():
                raise ValueError("Cannot replace a QUBO while it is memory-mapped; choose another destination")
            with qubo.path.open("rb") as source, temporary.open("wb") as handle:
                shutil.copyfileobj(source, handle, BLOCK * 8)
        else:
            # Accept legacy mappings without assuming sorted or unique terms.
            accumulator = TermAccumulator()
            try:
                linear = linear_values(qubo)
                for i, j, value in qubo["quadratic"]:
                    if i == j:
                        linear[i] += value
                    else:
                        accumulator.add(min(i, j), max(i, j), value)
                write_sorted(temporary, qubo["num_variables"], linear, accumulator.terms(), qubo.get("offset", 0.0))
            finally:
                accumulator.close()
        with temporary.open("r+b") as handle:
            fields = list(HEADER.unpack(handle.read(HEADER.size)))
            fields[-2:] = [_text(problem_type), _text(instance_id)]
            handle.seek(0)
            handle.write(HEADER.pack(*fields))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def content_digest(qubo):
    """Stable binary digest, streamed without JSON or a second coefficient copy."""
    digest = hashlib.sha256()
    digest.update(struct.pack("<Qd", qubo["num_variables"], qubo["offset"]))
    linear = linear_values(qubo)
    for start in range(0, len(linear), BLOCK):
        digest.update(np.asarray(linear[start : start + BLOCK], dtype="<f8").tobytes())
    for first, second, values in quadratic_blocks(qubo):
        records = np.empty(len(values), dtype=RECORD)
        records["i"], records["j"], records["v"] = first, second, values
        digest.update(records.tobytes())
    return digest.hexdigest()


class TermAccumulator:
    """External merge accumulator with bounded numeric buffers, no edge dictionary."""

    def __init__(self):
        self._temporary = tempfile.TemporaryDirectory(prefix="qubo-terms-")
        self.root = Path(self._temporary.name)
        self.buffer = np.empty(BLOCK, dtype=RECORD)
        self.used = 0
        self.runs = []
        self.counter = 0

    def add(self, i, j, value):
        self.buffer[self.used] = i, j, value
        self.used += 1
        if self.used == BLOCK:
            self._spill()

    def _new_path(self):
        self.counter += 1
        return self.root / f"run_{self.counter}"

    def _spill(self):
        if not self.used:
            return
        block = self.buffer[:self.used]
        block[:] = block[np.argsort(block[["i", "j"]], order=["i", "j"], kind="stable")]
        path = self._new_path()
        with path.open("wb") as handle:
            block.tofile(handle)
        self.runs.append(path)
        self.used = 0
        # Limit merge fan-in and open files without repeatedly merging all old runs.

    @staticmethod
    def _records(path):
        with path.open("rb") as handle:
            while True:
                block = np.fromfile(handle, dtype=RECORD, count=4096)
                if not len(block):
                    break
                for entry in block:
                    yield int(entry["i"]), int(entry["j"]), float(entry["v"])

    def terms(self):
        self._spill()
        runs = list(self.runs)
        while len(runs) > 32:
            merged = []
            for start in range(0, len(runs), 32):
                group = runs[start : start + 32]
                path = self._new_path()
                used = 0
                with path.open("wb") as handle:
                    for record in heapq.merge(*(self._records(p) for p in group), key=lambda item: item[:2]):
                        self.buffer[used] = record
                        used += 1
                        if used == BLOCK:
                            self.buffer.tofile(handle)
                            used = 0
                    self.buffer[:used].tofile(handle)
                for old in group:
                    old.unlink()
                merged.append(path)
            runs = merged
        previous, total = None, 0.0
        for i, j, value in heapq.merge(*(self._records(p) for p in runs), key=lambda item: item[:2]):
            if previous != (i, j):
                if previous is not None:
                    yield *previous, total
                previous, total = (i, j), value
            else:
                total += value
        if previous is not None:
            yield *previous, total

    def close(self):
        self._temporary.cleanup()

    def __del__(self):
        self.close()

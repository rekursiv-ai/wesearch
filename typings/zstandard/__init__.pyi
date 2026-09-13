# The ``zstandard`` wheel is the pre-3.14 fallback and is not installed in the
# 3.14-only monorepo, so the checkers need the surface wesearch touches.

from collections.abc import Buffer
from types import TracebackType
from typing import IO, Self

class ZstdError(Exception): ...

class ZstdCompressionWriter:
    def write(self, data: Buffer) -> int: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...

class ZstdCompressor:
    def __init__(self, level: int = 3) -> None: ...
    def compress(self, data: Buffer) -> bytes: ...
    def stream_writer(
        self,
        writer: IO[bytes],
        size: int = -1,
        write_size: int = ...,
        write_return_read: bool = True,
        closefd: bool = True,
    ) -> ZstdCompressionWriter: ...

class ZstdDecompressionReader:
    def read(self, size: int = -1) -> bytes: ...

class ZstdDecompressor:
    def __init__(self) -> None: ...
    def stream_reader(
        self,
        source: IO[bytes] | Buffer,
        read_size: int = ...,
        read_across_frames: bool = False,
        closefd: bool = True,
    ) -> ZstdDecompressionReader: ...

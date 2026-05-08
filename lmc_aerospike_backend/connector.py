# SPDX-License-Identifier: Apache-2.0
"""
Aerospike remote connector for LMCache.

Uses the official sync Aerospike Python client (aerospike>=19.0.0) with
asyncio.to_thread() wrappers. The sync client is chosen over the alpha
async client because it provides full batch API support (batch_read,
batch_write) critical for bulk KV cache loading performance.
"""

from typing import List, Optional
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import aerospike
from aerospike_helpers.batch import records as batch_helpers
from aerospike_helpers.operations import operations

from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.protocol import RemoteMetadata
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)

_METADATA_BIN = "metadata"
_KV_DATA_BIN = "kv_data"
_MAX_BATCH_SIZE = 64
_READ_PARALLELISM = 4


class AerospikeConnector(RemoteConnector):
    """
    LMCache RemoteConnector backed by Aerospike.

    Stores each KV cache chunk as a single Aerospike record with two bins:
      - ``metadata``: serialized RemoteMetadata (shapes, dtypes, format)
      - ``kv_data``:  raw tensor bytes

    All async methods delegate to the sync Aerospike client via
    asyncio.to_thread() to avoid blocking the event loop.
    """

    def __init__(
        self,
        host: str,
        port: int,
        namespace: str,
        set_name: str,
        local_cpu_backend: LocalCPUBackend,
        pool_size: int = 64,
        connect_timeout: int = 5000,
        username: str = "",
        password: str = "",
    ):
        super().__init__(local_cpu_backend.config, local_cpu_backend.metadata)

        self._namespace = namespace
        self._set = set_name
        self._local_cpu_backend = local_cpu_backend

        as_config = {
            "hosts": [(host, port)],
            "policies": {
                "max_conns_per_node": pool_size,
                "min_conns_per_node": pool_size,
                "timeout": connect_timeout,
                "write": {
                    "total_timeout": 10000,
                    "socket_timeout": 10000,
                    "max_retries": 2,
                },
                "batch_write": {
                    "total_timeout": 30000,
                    "socket_timeout": 10000,
                },
            },
        }

        self._client = aerospike.client(as_config)
        if username and password:
            self._client.connect(username, password)
        else:
            self._client.connect()

        self._read_pool = ThreadPoolExecutor(
            max_workers=_READ_PARALLELISM,
            thread_name_prefix="as-read",
        )

        logger.info(
            "Aerospike connector initialized: %s:%d ns=%s set=%s pool=%d "
            "read_parallelism=%d",
            host, port, namespace, set_name, pool_size, _READ_PARALLELISM,
        )

    def _as_key(self, key: CacheEngineKey) -> tuple:
        return (self._namespace, self._set, key.to_string())

    # ------------------------------------------------------------------
    # Required abstract methods (6)
    # ------------------------------------------------------------------

    def _exists_sync(self, key: CacheEngineKey) -> bool:
        try:
            _, meta = self._client.exists(self._as_key(key))
            return meta is not None
        except aerospike.exception.RecordNotFound:
            return False

    async def exists(self, key: CacheEngineKey) -> bool:
        return await asyncio.to_thread(self._exists_sync, key)

    def exists_sync(self, key: CacheEngineKey) -> bool:
        return self._exists_sync(key)

    def _get_sync(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        try:
            _, _, bins = self._client.get(self._as_key(key))
        except aerospike.exception.RecordNotFound:
            return None

        if bins is None:
            return None

        metadata_bytes = bins.get(_METADATA_BIN)
        kv_bytes = bins.get(_KV_DATA_BIN)

        if metadata_bytes is None or kv_bytes is None:
            logger.warning("Incomplete record for key %s", key.to_string())
            return None

        metadata = RemoteMetadata.deserialize(memoryview(metadata_bytes))

        memory_obj = self._local_cpu_backend.allocate(
            metadata.shapes, metadata.dtypes, metadata.fmt,
        )
        if memory_obj is None:
            logger.warning("Failed to allocate memory for key %s", key.to_string())
            return None

        view = memory_obj.byte_array
        if not isinstance(view, memoryview):
            view = memoryview(view)
        if view.format == "B":
            view[:len(kv_bytes)] = kv_bytes
        else:
            view.cast("B")[:len(kv_bytes)] = kv_bytes

        return memory_obj

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        return await asyncio.to_thread(self._get_sync, key)

    def _put_sync(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        import time as _time

        kv_buf = memory_obj.byte_array
        if isinstance(kv_buf, memoryview):
            kv_bytes = bytes(kv_buf)
        else:
            kv_bytes = bytes(kv_buf)

        metadata = RemoteMetadata(
            length=len(kv_bytes),
            shapes=self.meta_shapes,
            dtypes=self.meta_dtypes,
            fmt=self.meta_fmt,
        )
        metadata_bytes = bytes(metadata.serialize())

        t0 = _time.perf_counter()
        try:
            self._client.put(
                self._as_key(key),
                {_METADATA_BIN: metadata_bytes, _KV_DATA_BIN: kv_bytes},
            )
        except Exception:
            logger.exception(
                "Aerospike put EXCEPTION key=%s size=%d",
                key.to_string()[:60], len(kv_bytes),
            )
            raise
        elapsed = (_time.perf_counter() - t0) * 1000
        logger.info(
            "Aerospike put OK: key=%s size=%.2f MB, %.1f ms",
            key.to_string()[:60], len(kv_bytes) / (1024 * 1024), elapsed,
        )

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        await asyncio.to_thread(self._put_sync, key, memory_obj)

    def _list_sync(self) -> List[str]:
        keys: List[str] = []
        scan = self._client.scan(self._namespace, self._set)
        scan.foreach(lambda record: keys.append(record[0][2]))
        return keys

    async def list(self) -> List[str]:
        return await asyncio.to_thread(self._list_sync)

    async def close(self) -> None:
        try:
            self._read_pool.shutdown(wait=False)
            self._client.close()
            logger.info("Closed Aerospike connection")
        except Exception:
            logger.exception("Error closing Aerospike connection")

    # ------------------------------------------------------------------
    # Batched operations using native Aerospike batch APIs
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_list(lst, chunk_size):
        """Yield successive chunks of at most chunk_size from lst."""
        for i in range(0, len(lst), chunk_size):
            yield lst[i : i + chunk_size]

    # -- batched contains --

    def _batched_contains_sync(self, keys: List[CacheEngineKey]) -> int:
        """Check how many consecutive keys exist, starting from the first."""
        as_keys = [self._as_key(k) for k in keys]
        total_count = 0

        for chunk in self._chunk_list(as_keys, _MAX_BATCH_SIZE):
            batch_result = self._client.batch_read(chunk, bins=[])
            for batch_record in batch_result.batch_records:
                if batch_record.result != 0:
                    return total_count
                total_count += 1

        return total_count

    def support_batched_contains(self) -> bool:
        return True

    def batched_contains(self, keys: List[CacheEngineKey]) -> int:
        return self._batched_contains_sync(keys)

    def support_batched_async_contains(self) -> bool:
        return True

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        return await asyncio.to_thread(self._batched_contains_sync, keys)

    # -- batched get --

    def _read_chunk(self, as_keys: list) -> List[Optional[MemoryObj]]:
        """Fetch one chunk of keys (up to _MAX_BATCH_SIZE) via batch_read."""
        objs: List[Optional[MemoryObj]] = []
        batch_result = self._client.batch_read(as_keys)

        for batch_record in batch_result.batch_records:
            if batch_record.result != 0:
                objs.append(None)
                continue

            _, _, bins = batch_record.record
            metadata_bytes = bins.get(_METADATA_BIN)
            kv_bytes = bins.get(_KV_DATA_BIN)

            if metadata_bytes is None or kv_bytes is None:
                objs.append(None)
                continue

            metadata = RemoteMetadata.deserialize(
                memoryview(metadata_bytes)
            )
            memory_obj = self._local_cpu_backend.allocate(
                metadata.shapes, metadata.dtypes, metadata.fmt,
            )
            if memory_obj is None:
                objs.append(None)
                continue

            view = memory_obj.byte_array
            if not isinstance(view, memoryview):
                view = memoryview(view)
            if view.format == "B":
                view[:len(kv_bytes)] = kv_bytes
            else:
                view.cast("B")[:len(kv_bytes)] = kv_bytes

            objs.append(memory_obj)

        return objs

    def _batched_get_sync(
        self, keys: List[CacheEngineKey],
    ) -> List[Optional[MemoryObj]]:
        """Bulk fetch using parallel Aerospike batch_reads across multiple
        threads to saturate network bandwidth. Each batch_read is capped at
        _MAX_BATCH_SIZE records. Up to _READ_PARALLELISM batch_reads run
        concurrently via a thread pool."""
        import time as _time

        as_keys = [self._as_key(k) for k in keys]
        chunks = list(self._chunk_list(as_keys, _MAX_BATCH_SIZE))

        if len(chunks) <= 1:
            return self._read_chunk(as_keys) if as_keys else []

        t0 = _time.perf_counter()
        futures = [
            self._read_pool.submit(self._read_chunk, chunk)
            for chunk in chunks
        ]
        all_objs: List[Optional[MemoryObj]] = []
        for future in futures:
            all_objs.extend(future.result())
        elapsed = (_time.perf_counter() - t0) * 1000

        ok_count = sum(1 for o in all_objs if o is not None)
        logger.info(
            "Aerospike parallel batch_read: %d keys, %d chunks, "
            "%.1f ms, %d/%d OK",
            len(as_keys), len(chunks), elapsed,
            ok_count, len(as_keys),
        )

        return all_objs

    def support_batched_get(self) -> bool:
        return True

    async def batched_get(
        self, keys: List[CacheEngineKey],
    ) -> List[Optional[MemoryObj]]:
        return await asyncio.to_thread(self._batched_get_sync, keys)

    def support_batched_get_non_blocking(self) -> bool:
        return True

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
    ) -> List[MemoryObj]:
        """
        Fetch consecutive prefix of valid MemoryObjs. Release any trailing
        objects after the first failure to avoid memory leaks.
        """
        results = await asyncio.to_thread(self._batched_get_sync, keys)

        memory_objs: List[MemoryObj] = []
        found_failure = False
        for result in results:
            if found_failure:
                if isinstance(result, MemoryObj):
                    result.ref_count_down()
            elif isinstance(result, MemoryObj):
                memory_objs.append(result)
            else:
                found_failure = True

        return memory_objs

    # -- batched put --

    def _do_batch_write(self, write_records):
        """Execute a single batch_write and return (succeeded, failed_records).

        failed_records is a list of Write objects that got a retryable error
        (e.g. DEVICE_OVERLOAD=18).
        """
        import time as _time

        batch = batch_helpers.BatchRecords(write_records)
        t0 = _time.perf_counter()
        try:
            self._client.batch_write(batch)
        except Exception:
            logger.exception(
                "Aerospike batch_write EXCEPTION records=%d",
                len(write_records),
            )
            raise
        elapsed = (_time.perf_counter() - t0) * 1000

        failed_records = []
        for rec, orig in zip(batch.batch_records, write_records):
            if rec.result != 0:
                failed_records.append(orig)

        succeeded = len(write_records) - len(failed_records)
        if failed_records:
            codes: dict = {}
            for rec in batch.batch_records:
                if rec.result != 0:
                    codes[rec.result] = codes.get(rec.result, 0) + 1
            logger.warning(
                "Aerospike batch_write: %d/%d FAILED (%.1f ms). "
                "Error codes: %s",
                len(failed_records), len(write_records), elapsed, codes,
            )
        else:
            logger.info(
                "Aerospike batch_write OK: %d records, %.1f ms",
                len(write_records), elapsed,
            )
        return succeeded, failed_records

    def _batched_put_sync(
        self,
        keys: List[CacheEngineKey],
        memory_objs: List[MemoryObj],
    ) -> None:
        """Bulk store using Aerospike batch_write with retry for
        DEVICE_OVERLOAD (error 18) and other transient failures."""
        import time as _time

        pairs = list(zip(keys, memory_objs))

        for chunk in self._chunk_list(pairs, _MAX_BATCH_SIZE):
            write_records = []
            for key, memory_obj in chunk:
                kv_buf = memory_obj.byte_array
                kv_bytes = bytes(kv_buf) if isinstance(kv_buf, memoryview) \
                    else bytes(kv_buf)

                metadata = RemoteMetadata(
                    length=len(kv_bytes),
                    shapes=self.meta_shapes,
                    dtypes=self.meta_dtypes,
                    fmt=self.meta_fmt,
                )
                metadata_bytes = bytes(metadata.serialize())

                write_records.append(
                    batch_helpers.Write(
                        key=self._as_key(key),
                        ops=[
                            operations.write(_METADATA_BIN, metadata_bytes),
                            operations.write(_KV_DATA_BIN, kv_bytes),
                        ],
                    )
                )

            pending = write_records
            max_retries = 5
            for attempt in range(max_retries + 1):
                _, failed = self._do_batch_write(pending)
                if not failed:
                    break
                if attempt == max_retries:
                    logger.error(
                        "Aerospike batch_write: %d records still failed "
                        "after %d retries -- giving up",
                        len(failed), max_retries,
                    )
                    break
                backoff = min(0.2 * (2 ** attempt), 2.0)
                logger.info(
                    "Retrying %d failed records in %.1fs (attempt %d/%d)",
                    len(failed), backoff, attempt + 1, max_retries,
                )
                _time.sleep(backoff)
                pending = failed

    def support_batched_put(self) -> bool:
        return True

    async def batched_put(
        self,
        keys: List[CacheEngineKey],
        memory_objs: List[MemoryObj],
    ) -> None:
        await asyncio.to_thread(self._batched_put_sync, keys, memory_objs)

    # -- ping --

    def support_ping(self) -> bool:
        return True

    async def ping(self) -> int:
        try:
            info = await asyncio.to_thread(self._client.is_connected)
            return 0 if info else 1
        except Exception:
            return 1

    def __repr__(self) -> str:
        return (
            f"<AerospikeConnector ns={self._namespace} "
            f"set={self._set}>"
        )

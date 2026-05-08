# LMCache Aerospike Backend

Aerospike remote storage backend plugin for [LMCache](https://github.com/LMCache/LMCache) — enables distributed KV cache storage using Aerospike as the persistence layer.

## Overview

This package provides an LMCache `RemoteConnector` implementation backed by Aerospike, allowing vLLM inference servers to store and retrieve KV cache chunks from a high-performance Aerospike cluster. It supports batched operations for efficient bulk cache loading and includes retry logic for transient failures.

## Features

- **Full LMCache Integration**: Implements `RemoteConnector` and `ConnectorAdapter` interfaces
- **URL-Based Configuration**: Simple `aerospike://` URL scheme for easy setup
- **Batched Operations**: Native Aerospike batch APIs for high-throughput cache reads/writes
- **Async Support**: All operations are async-friendly using `asyncio.to_thread()`
- **Connection Pooling**: Configurable connection pool size and timeouts
- **Retry Logic**: Automatic retry with exponential backoff for transient failures
- **Docker Support**: Pre-built Dockerfile for vLLM integration

## Installation

### From Source

```bash
pip install -e .
```

### With Development Dependencies

```bash
pip install -e ".[dev]"
```

## Usage

### Basic Configuration

Configure LMCache to use Aerospike via environment variables or config:

```bash
export LMCACHE_REMOTE_URL="aerospike://localhost:3000/lmcache/kv_cache"
```

URL format:
```
aerospike://[user:pass@]host:port[/namespace[/set]]
```

Defaults:
- `namespace`: `lmcache`
- `set`: `kv_cache`

### Advanced Configuration

Additional options can be provided via LMCache `extra_config`:

```python
extra_config = {
    "aerospike_pool_size": 64,          # Connection pool size
    "aerospike_connect_timeout": 5000,  # Connection timeout (ms)
    "aerospike_username": "admin",       # Username (if auth enabled)
    "aerospike_password": "secret",    # Password (if auth enabled)
}
```

## Docker

A Dockerfile is provided for building a vLLM image with Aerospike support:

```bash
docker build -f Dockerfile.vllm-aerospike -t vllm-aerospike:latest .
```

## Architecture

### Data Model

Each KV cache chunk is stored as a single Aerospike record with two bins:

- `metadata`: Serialized `RemoteMetadata` (shapes, dtypes, format info)
- `kv_data`: Raw tensor bytes

Record key format: `(namespace, set, cache_engine_key_string)`

### Batched Operations

The connector implements optimized batched operations:

- **batch_read**: Parallel reads across multiple threads (up to 64 records per batch, 4 parallel workers)
- **batch_write**: Bulk writes with automatic retry for `DEVICE_OVERLOAD` errors
- **batched_contains**: Check existence of consecutive keys

## Project Structure

```
lmc-aerospike-backend/
├── lmc_aerospike_backend/
│   ├── __init__.py          # Package exports
│   ├── adapter.py           # LMCache ConnectorAdapter implementation
│   └── connector.py         # Aerospike RemoteConnector implementation
├── Dockerfile.vllm-aerospike # Docker image for vLLM integration
├── pyproject.toml           # Package configuration
└── README.md               # This file
```

## Requirements

- Python >= 3.10
- aerospike >= 19.0.0
- LMCache (for integration)

## Development

### Running Tests

```bash
pytest
```

## License

Apache-2.0

## Notes

- `features.conf` contains Aerospike license information and should not be committed to version control (already added to `.gitignore`)
- See `aerospike_configs.md` for detailed Aerospike configuration options

# SPDX-License-Identifier: Apache-2.0
"""
LMCache ConnectorAdapter for Aerospike.

Registers the ``aerospike://`` URL scheme and creates an
AerospikeConnector from the parsed URL and LMCache config.

URL format:
    aerospike://[user:pass@]host:port[/namespace[/set]]

Defaults:
    namespace = "lmcache"
    set       = "kv_cache"
"""

from lmcache.logging import init_logger
from lmcache.v1.storage_backend.connector import (
    ConnectorAdapter,
    ConnectorContext,
    parse_remote_url,
)
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector

logger = init_logger(__name__)

_DEFAULT_NAMESPACE = "lmcache"
_DEFAULT_SET = "kv_cache"
_DEFAULT_POOL_SIZE = 64
_DEFAULT_CONNECT_TIMEOUT = 5000


class AerospikeConnectorAdapter(ConnectorAdapter):
    """LMCache adapter that handles ``aerospike://`` URLs."""

    def __init__(self) -> None:
        super().__init__("aerospike://")

    def can_parse(self, url: str) -> bool:
        return url.startswith(self.schema)

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        from .connector import AerospikeConnector

        parsed = parse_remote_url(context.url)

        # Extract namespace/set from URL path: /namespace/set
        path_parts = [p for p in parsed.path.strip("/").split("/") if p]
        namespace = path_parts[0] if len(path_parts) >= 1 else _DEFAULT_NAMESPACE
        set_name = path_parts[1] if len(path_parts) >= 2 else _DEFAULT_SET

        config = context.config
        extra = (config.extra_config or {}) if config else {}

        pool_size = int(extra.get("aerospike_pool_size", _DEFAULT_POOL_SIZE))
        connect_timeout = int(
            extra.get("aerospike_connect_timeout", _DEFAULT_CONNECT_TIMEOUT)
        )
        username = str(extra.get("aerospike_username", parsed.username or ""))
        password = str(extra.get("aerospike_password", parsed.password or ""))

        logger.info(
            "Creating Aerospike connector: %s:%d ns=%s set=%s pool=%d",
            parsed.host, parsed.port, namespace, set_name, pool_size,
        )

        return AerospikeConnector(
            host=parsed.host,
            port=parsed.port,
            namespace=namespace,
            set_name=set_name,
            local_cpu_backend=context.local_cpu_backend,
            pool_size=pool_size,
            connect_timeout=connect_timeout,
            username=username,
            password=password,
        )

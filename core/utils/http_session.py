import aiohttp
from typing import Optional

_session: Optional[aiohttp.ClientSession] = None
_connector: Optional[aiohttp.TCPConnector] = None

def init_session() -> aiohttp.ClientSession:
    global _session
    global _connector
    if _session is None or _session.closed:
        if _connector is None or _connector.closed:
            _connector = aiohttp.TCPConnector(limit=256, ttl_dns_cache=300)
        _session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30.0),
            connector=_connector,
        )
    return _session

async def close_session() -> None:
    global _session
    global _connector
    try:
        if _session is not None and not _session.closed:
            await _session.close()
    finally:
        try:
            if _connector is not None and not _connector.closed:
                await _connector.close()
        finally:
            _session = None
            _connector = None

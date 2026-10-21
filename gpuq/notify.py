"""Phone notifications through ntfy (https://ntfy.sh). The topic works like a password: anyone who
knows it can read the messages, so `gpuq setup` generates a long random one."""
from __future__ import annotations

import urllib.request


def send(topic, title: str, message: str, server: str = "https://ntfy.sh", timeout: float = 30) -> bool:
    if not topic:
        return False
    req = urllib.request.Request(f"{server.rstrip('/')}/{topic}", data=message.encode("utf-8"), method="POST",
                                 headers={"Title": title.encode("utf-8").decode("latin-1", "replace")})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return 200 <= r.status < 300
    except Exception:
        return False

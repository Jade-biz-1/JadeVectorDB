"""
Shared slowapi rate-limiter instance.
Import `limiter` here rather than creating it in each router so there is
exactly one limiter across the whole application.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])

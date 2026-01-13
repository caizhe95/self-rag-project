import hashlib
import pickle
from typing import Any, Optional


class SimpleCache:
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._cache = {}

    def _generate_key(self, *args) -> str:
        key_str = pickle.dumps(args)
        return hashlib.md5(key_str).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        return self._cache.get(key)

    def set(self, key: str, value: Any):
        if len(self._cache) >= self.max_size:
            first_key = next(iter(self._cache))
            del self._cache[first_key]
        self._cache[key] = value

    def clear(self):
        self._cache.clear()


query_cache = SimpleCache(max_size=100)


def cached_retrieval(func):
    def wrapper(*args):
        cache_key = query_cache._generate_key(*args)
        result = query_cache.get(cache_key)
        if result is not None:
            print(f"缓存命中: {args[0][:50]}...")
            return result

        result = func(*args)
        query_cache.set(cache_key, result)
        return result

    return wrapper
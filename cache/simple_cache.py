import hashlib
import pickle
import time
from typing import Any, Optional, Tuple


class SimpleCache:
    """支持TTL的缓存"""

    def __init__(self, max_size: int = 100, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl  # 过期时间（秒）
        self._cache = {}  # {key: (value, timestamp)}

    def _generate_key(self, *args) -> str:
        key_str = pickle.dumps(args)
        return hashlib.md5(key_str).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                del self._cache[key]  # 过期删除
        return None

    def set(self, key: str, value: Any):
        if len(self._cache) >= self.max_size:
            # LRU淘汰：删除最早的数据
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]

        self._cache[key] = (value, time.time())

    def clear(self):
        self._cache.clear()


# 多级缓存实例
class MultiLevelCache:
    """三级缓存：Embedding > 检索结果 > 重排序结果"""

    def __init__(self, config):
        # Embedding缓存
        self.embedding_cache = SimpleCache(
            max_size=config.CACHE_MAX_SIZE * 2,
            ttl=7200  # 2小时
        )

        # 检索结果缓存
        self.retrieval_cache = SimpleCache(
            max_size=config.CACHE_MAX_SIZE,
            ttl=1800  # 30分钟
        )

        # 重排序缓存
        self.rerank_cache = SimpleCache(
            max_size=config.CACHE_MAX_SIZE // 2,
            ttl=600  # 10分钟
        )

    def get_embedding(self, text: str) -> Optional[Any]:
        key = self.embedding_cache._generate_key(text)
        return self.embedding_cache.get(key)

    def set_embedding(self, text: str, embedding: Any):
        key = self.embedding_cache._generate_key(text)
        self.embedding_cache.set(key, embedding)


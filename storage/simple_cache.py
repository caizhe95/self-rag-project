# core/simple_cache.py
import hashlib
import json
import time
from typing import Any, Optional


class SimpleCache:
    """多级缓存（面试级：TTL + LRU）"""

    def __init__(self, max_size: int = 100, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self._cache = {}  # {key: (value, timestamp)}

    def _generate_key(self, *args, **kwargs) -> str:
        """生成稳定键（使用JSON而非pickle）"""
        sorted_kwargs = tuple(sorted(kwargs.items()))
        key_data = (args, sorted_kwargs)
        key_str = json.dumps(key_data, sort_keys=True, ensure_ascii=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                del self._cache[key]  # 过期删除

        return None

    def set(self, key: str, value: Any, *args, **kwargs):
        """设置缓存"""
        # LRU淘汰
        if len(self._cache) >= self.max_size:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]

        self._cache[key] = (value, time.time())

    def clear(self):
        """清空缓存"""
        self._cache.clear()


class MultiLevelCache:
    """三级缓存：Embedding > 检索结果 > 重排序结果"""

    def __init__(self, config):
        # Embedding缓存（2小时）
        self.embedding_cache = SimpleCache(
            max_size=config.cache_max_size * 2,
            ttl=7200
        )

        # 检索结果缓存（30分钟）
        self.retrieval_cache = SimpleCache(
            max_size=config.cache_max_size,
            ttl=1800
        )

        # 重排序缓存（10分钟）
        self.rerank_cache = SimpleCache(
            max_size=config.cache_max_size // 2,
            ttl=600
        )

    def get_embedding(self, text: str) -> Optional[Any]:
        key = self.embedding_cache._generate_key(text)
        return self.embedding_cache.get(key)

    def set_embedding(self, text: str, embedding: Any):
        key = self.embedding_cache._generate_key(text)
        self.embedding_cache.set(key, embedding)
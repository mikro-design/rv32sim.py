import pytest

from memif_cache import LRUCacheMemIf, _parse_bool, _parse_int


def _make_cache(config=None):
    return LRUCacheMemIf(lambda *_: 0, lambda *_: None, config=config or {})


def test_parse_helpers():
    assert _parse_int("0x10") == 16
    assert _parse_int(True) == 1
    assert _parse_int(None, default=7) == 7
    assert _parse_bool(None, default=True) is True
    assert _parse_bool(0) is False
    assert _parse_bool("on") is True
    assert _parse_bool("0") is False
    assert _parse_bool("maybe") is True


def test_reset_validation():
    with pytest.raises(ValueError):
        _make_cache({"size_bytes": 0})
    with pytest.raises(ValueError):
        _make_cache({"line_bytes": 0})
    with pytest.raises(ValueError):
        _make_cache({"size_bytes": 96, "line_bytes": 32, "ways": 2})


def test_access_hits_misses_and_stats():
    cache = _make_cache(
        {
            "size_bytes": 64,
            "line_bytes": 16,
            "ways": 2,
            "hit_penalty": 1,
            "miss_penalty": 5,
        }
    )
    assert cache.access(0, 4) == 5
    assert cache.access(0, 4) == 1
    assert cache.access(16, 4) == 5
    assert cache.stats() == {"accesses": 3, "hits": 1, "misses": 2}


def test_access_size_zero_and_write_allocate():
    cache = _make_cache({"write_allocate": False})
    assert cache.access(0, 0) == 0
    assert cache.access(0, 4, is_write=True) == 0
    assert cache.access(0, 4) == 0
    assert cache.stats()["misses"] == 2


def test_eviction_with_single_way():
    cache = _make_cache({"size_bytes": 32, "line_bytes": 16, "ways": 1})
    cache.access(0, 4)
    cache.access(32, 4)
    cache.access(0, 4)
    assert cache.stats()["misses"] == 3

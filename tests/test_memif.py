import pytest

from memif import FlatMemIf, MemIfBase, _parse_bool, _parse_int, load_memif


def test_parse_helpers():
    assert _parse_int("0x10") == 16
    assert _parse_int(True) == 1
    assert _parse_int(None, default=3) == 3
    assert _parse_bool("yes") is True
    assert _parse_bool("0") is False
    assert _parse_bool("maybe") is True


def test_flat_memif_stats():
    memif = FlatMemIf(lambda *_: 0, lambda *_: None, config={"penalty": "2"})
    assert memif.access(0, 4) == 2
    assert memif.access(0, 4) == 2
    stats = memif.stats()
    assert stats["accesses"] == 2

    memif = FlatMemIf(lambda *_: 0, lambda *_: None, config={"count_misses": True})
    memif.access(0, 4)
    stats = memif.stats()
    assert stats["misses"] == 1


def test_load_memif_spec_errors():
    with pytest.raises(ValueError):
        load_memif("", lambda *_: 0, lambda *_: None)

    memif = load_memif("memif:FlatMemIf", lambda *_: 0, lambda *_: None)
    assert isinstance(memif, FlatMemIf)


def test_parse_bool_and_base_class():
    assert _parse_bool(None, default=True) is True
    assert _parse_bool(True) is True
    assert _parse_bool(0) is False
    assert _parse_bool(1.5) is True
    assert _parse_bool("off") is False

    base = MemIfBase(lambda *_: 0, lambda *_: None)
    assert base.access(0, 1) == 0
    assert base.stats() == {}

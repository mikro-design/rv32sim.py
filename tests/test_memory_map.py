import pytest

from memory_map import MemoryMap


def test_default_regions_created():
    mem = MemoryMap(0x1000)
    mem.ensure_regions()
    assert any(region.name == "flash" for region in mem.regions)
    assert any(region.name == "sram" for region in mem.regions)
    flash = next(region for region in mem.regions if region.name == "flash")
    sram = next(region for region in mem.regions if region.name == "sram")
    assert flash.start == 0x0
    assert flash.end == 0x00020000
    assert sram.start == 0x20000000


def test_add_region_rejects_overlap():
    mem = MemoryMap(0x1000)
    mem.add_region(0x0, 0x100, "r0")
    with pytest.raises(ValueError):
        mem.add_region(0x80, 0x180, "r1")


def test_read_write_across_regions():
    mem = MemoryMap(0x1000)
    mem.regions = []
    mem.initialized = True
    mem.add_region(0x0, 0x10, "r0")
    mem.add_region(0x10, 0x20, "r1")
    mem.write_memory(0x0e, b"\x01\x02\x03\x04")
    assert mem.read_bytes(0x0e, 4) == b"\x01\x02\x03\x04"


def test_expand_region_preserves_data():
    mem = MemoryMap(0x1000)
    mem.regions = []
    mem.initialized = True
    region = mem.add_region(0x100, 0x110, "r0")
    mem.write_memory(0x108, b"\xaa\xbb")
    mem.expand_region(region, 0x120)
    assert region.end == 0x120
    assert mem.read_bytes(0x108, 2) == b"\xaa\xbb"
    mem.expand_region(region, 0x120)
    assert region.end == 0x120


def test_get_stack_top_prefers_sram():
    mem = MemoryMap(0x1000)
    mem.regions = []
    mem.initialized = True
    mem.add_region(0x0, 0x100, "flash")
    mem.add_region(0x20000000, 0x20001000, "sram")
    assert mem.get_stack_top() == 0x20001000


def test_memory_map_errors():
    with pytest.raises(ValueError):
        MemoryMap(0x1000).add_region(0x100, 0x100, "bad")

    mem = MemoryMap(0x1000)
    mem.regions = []
    mem.initialized = True
    mem.add_region(0x0, 0x10, "r0")
    with pytest.raises(ValueError):
        mem.read_memory(0x20, 1)
    with pytest.raises(ValueError):
        mem.read_bytes(0x20, 1)
    with pytest.raises(ValueError):
        mem.write_memory(0x20, b"\x00")

    r0 = mem.add_region(0x20, 0x30, "r1")
    mem.add_region(0x40, 0x50, "r2")
    with pytest.raises(ValueError):
        mem.expand_region(r0, 0x45)

    mem.regions = []
    mem.initialized = True
    assert mem.get_stack_top() == 0

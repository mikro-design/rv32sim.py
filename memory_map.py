class MemoryRegion:
    def __init__(self, start, end, name="mem"):
        if end <= start:
            raise ValueError(f"Invalid memory region {name}: 0x{start:08x}-0x{end:08x}")
        self.start = start
        self.end = end
        self.name = name
        self.data = bytearray(end - start)


class MemoryMap:
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.regions = []
        self.initialized = False

    def init_default_regions(self):
        self.regions = []
        self.add_region(0x00000000, 0x00020000, "flash")
        sram_end = 0x20000000 + self.memory_size
        self.add_region(0x20000000, sram_end, "sram")
        self.initialized = True

    def ensure_regions(self):
        if not self.regions and not self.initialized:
            self.init_default_regions()

    def add_region(self, start, end, name="mem"):
        start &= 0xffffffff
        end &= 0xffffffff
        region = MemoryRegion(start, end, name)
        for existing in self.regions:
            if not (region.end <= existing.start or region.start >= existing.end):
                raise ValueError(
                    f"Overlapping memory regions: {existing.name} "
                    f"0x{existing.start:08x}-0x{existing.end:08x} and "
                    f"{name} 0x{start:08x}-0x{end:08x}"
                )
        self.regions.append(region)
        return region

    def find_region(self, addr):
        self.ensure_regions()
        for region in self.regions:
            if region.start <= addr < region.end:
                return region
        return None

    def _iter_spans(self, addr, size, op):
        remaining = size
        cur = addr
        while remaining:
            region = self.find_region(cur)
            if region is None:
                raise ValueError(f"Memory {op} out of range at 0x{cur:08x}")
            offset = cur - region.start
            chunk = min(remaining, region.end - cur)
            yield region, offset, chunk
            cur += chunk
            remaining -= chunk

    def read_memory(self, addr, size):
        data = bytearray()
        for region, offset, chunk in self._iter_spans(addr, size, "read"):
            data += region.data[offset:offset + chunk]
        return int.from_bytes(data, "little")

    def read_bytes(self, addr, size):
        data = bytearray()
        for region, offset, chunk in self._iter_spans(addr, size, "read"):
            data += region.data[offset:offset + chunk]
        return bytes(data)

    def write_memory(self, addr, data):
        idx = 0
        for region, offset, chunk in self._iter_spans(addr, len(data), "write"):
            region.data[offset:offset + chunk] = data[idx:idx + chunk]
            idx += chunk

    def expand_region(self, region, new_end):
        if new_end <= region.end:
            return
        for existing in self.regions:
            if existing is region:
                continue
            if not (new_end <= existing.start or region.start >= existing.end):
                raise ValueError(
                    f"Overlapping memory regions: {region.name} "
                    f"0x{region.start:08x}-0x{new_end:08x} and "
                    f"{existing.name} 0x{existing.start:08x}-0x{existing.end:08x}"
                )
        new_data = bytearray(new_end - region.start)
        new_data[: len(region.data)] = region.data
        region.data = new_data
        region.end = new_end

    def get_stack_top(self):
        self.ensure_regions()
        sram = None
        for region in self.regions:
            if region.name.lower() == "sram":
                sram = region
                break
        if sram:
            return sram.end
        if self.regions:
            return max(self.regions, key=lambda r: r.end).end
        return 0

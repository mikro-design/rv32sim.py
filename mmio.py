class MMIOManager:
    def __init__(self, sim):
        self.sim = sim

    def register_read_handler(self, addr, func, width=1):
        self.sim.read_handlers[addr] = {"func": func, "width": width}

    def register_write_handler(self, addr, func, width=1):
        self.sim.write_handlers[addr] = {"func": func, "width": width}

    def _find_stub_entry(self, addr):
        if addr in self.sim.hw_stubs:
            entry = self.sim.hw_stubs[addr]
            width = entry.get("width", 4) if isinstance(entry, dict) else 4
            return addr, entry, width
        base = addr & ~0x3
        if base in self.sim.hw_stubs:
            entry = self.sim.hw_stubs[base]
            width = entry.get("width", 4) if isinstance(entry, dict) else 4
            if base <= addr < base + width:
                return base, entry, width
        base = addr & ~0x1
        if base in self.sim.hw_stubs:
            entry = self.sim.hw_stubs[base]
            width = entry.get("width", 4) if isinstance(entry, dict) else 4
            if base <= addr < base + width:
                return base, entry, width
        return None, None, None

    def _stub_read_value(self, stub):
        def to_int(val):
            if isinstance(val, str):
                try:
                    return int(val, 0)
                except ValueError:
                    return 0
            try:
                return int(val)
            except (TypeError, ValueError):
                return 0
        if isinstance(stub, dict):
            seq = stub.get("sequence")
            if seq is not None:
                if not isinstance(seq, (list, tuple)) or not seq:
                    return to_int(stub.get("value", 0))
                idx = stub.get("_seq_idx", 0)
                repeat = bool(stub.get("repeat"))
                hold_last = stub.get("hold_last", True)
                if repeat:
                    val = seq[idx % len(seq)]
                    stub["_seq_idx"] = idx + 1
                    return to_int(val)
                if idx < len(seq):
                    val = seq[idx]
                    stub["_seq_idx"] = idx + 1
                    return to_int(val)
                val = seq[-1] if hold_last else stub.get("value", 0)
                return to_int(val)
            return to_int(stub.get("value", 0))
        return to_int(stub)

    def _find_handler_entry(self, addr, handlers):
        if addr in handlers:
            entry = handlers[addr]
            return addr, entry, entry.get("width", 1)
        for base, entry in handlers.items():
            width = entry.get("width", 1)
            if base <= addr < base + width:
                return base, entry, width
        return None, None, None

    def _call_read_handler(self, handler, addr, size):
        func = handler["func"]
        try:
            return func(addr, size)
        except TypeError:
            return func()

    def _call_write_handler(self, handler, addr, value, size):
        func = handler["func"]
        try:
            func(addr, value, size)
        except TypeError:
            func(value)

    def read_value(self, addr, size):
        addr &= 0xffffffff

        base, handler, width = self._find_handler_entry(addr, self.sim.read_handlers)
        if handler:
            if size <= width and addr + size <= base + width:
                full = self.sim._u32(self._call_read_handler(handler, addr, size))
                shift = (addr - base) * 8
                mask = (1 << (size * 8)) - 1
                value = (full >> shift) & mask
                self.sim._track_hw_access(base, "read", [(value >> (8 * i)) & 0xff for i in range(size)])
                return value
            if width == 1 and size > 1:
                value = 0
                bytes_read = []
                for i in range(size):
                    b = self.sim._u32(self._call_read_handler(handler, addr + i, 1)) & 0xff
                    value |= b << (8 * i)
                    bytes_read.append(b)
                self.sim._track_hw_access(base, "read", bytes_read)
                return value

        if self.sim._assert_active() and self.sim.is_hardware_region(addr):
            value, base = self.sim._assert_read(addr, size)
            if value is not None:
                base_addr = base if base is not None else addr
                self.sim._track_hw_access(base_addr, "read", [(value >> (8 * i)) & 0xff for i in range(size)])
                return value

        base, stub, width = self._find_stub_entry(addr)
        if stub is not None and size <= width and addr + size <= base + width:
            full = self._stub_read_value(stub)
            full = self.sim._u32(full)
            shift = (addr - base) * 8
            mask = (1 << (size * 8)) - 1
            value = (full >> shift) & mask
            self.sim._track_hw_access(base, "read", [(value >> (8 * i)) & 0xff for i in range(size)])
            return value

        if self.sim._assert_active() and self.sim.is_hardware_region(addr):
            value, base = self.sim._mmio_state_read(addr, size)
            if value is not None:
                base_addr = base if base is not None else addr
                self.sim._track_hw_access(base_addr, "read", [(value >> (8 * i)) & 0xff for i in range(size)])
                return value

        if not self.sim.is_hardware_region(addr):
            return self.sim._read_memory(addr, size)

        self.sim._track_hw_access(addr, "read", [0] * size)
        return 0

    def write_value(self, addr, size, value):
        addr &= 0xffffffff
        value = self.sim._u32(value)
        value &= (1 << (size * 8)) - 1
        should_halt = self.sim.tohost_addr is not None and addr == self.sim.tohost_addr
        reg = self.sim.svd_index.find_register(addr) if self.sim.svd_index else None
        base, handler, width = self._find_handler_entry(addr, self.sim.write_handlers)
        if handler:
            if size <= width and addr + size <= base + width:
                if size == width and addr == base:
                    self._call_write_handler(handler, addr, value, size)
                    bytes_written = [(value >> (8 * i)) & 0xff for i in range(size)]
                else:
                    bytes_written = []
                    for i in range(size):
                        b = (value >> (8 * i)) & 0xff
                        self._call_write_handler(handler, addr + i, b, 1)
                        bytes_written.append(b)
                self.sim._track_hw_access(base, "write", bytes_written)
                if should_halt:
                    self.sim.tohost_value = self.sim._read_tohost_value(value)
                    self.sim._halt("tohost", self.sim.tohost_value)
                return
            if width == 1 and size > 1:
                bytes_written = []
                for i in range(size):
                    b = (value >> (8 * i)) & 0xff
                    self._call_write_handler(handler, addr + i, b, 1)
                    bytes_written.append(b)
                self.sim._track_hw_access(base, "write", bytes_written)
                if should_halt:
                    self.sim.tohost_value = self.sim._read_tohost_value(value)
                    self.sim._halt("tohost", self.sim.tohost_value)
                return

        if self.sim._assert_active() and self.sim.is_hardware_region(addr) and not should_halt and not self.sim.assert_writes:
            base = self.sim._mmio_state_write(addr, size, value, reg=reg)
            base_addr = base if base is not None else addr
            self.sim._track_hw_access(base_addr, "write", [(value >> (8 * i)) & 0xff for i in range(size)])
            return

        base, stub, width = self._find_stub_entry(addr)
        if stub is not None and size <= width and addr + size <= base + width:
            cur = stub.get("value", 0) if isinstance(stub, dict) else stub
            cur = self.sim._u32(cur)
            bytes_written = []
            for i in range(size):
                b = (value >> (8 * i)) & 0xff
                shift = (addr - base + i) * 8
                cur = (cur & ~(0xff << shift)) | (b << shift)
                bytes_written.append(b)
            if isinstance(stub, dict):
                stub["value"] = cur
            else:
                self.sim.hw_stubs[base] = cur
            self.sim._track_hw_access(base, "write", bytes_written)
            if should_halt:
                self.sim.tohost_value = self.sim._read_tohost_value(value)
                self.sim._halt("tohost", self.sim.tohost_value)
            return

        if self.sim._assert_active() and self.sim.is_hardware_region(addr) and not should_halt and self.sim.assert_writes:
            base = self.sim._assert_write(addr, size, value)
            base_addr = base if base is not None else addr
            self.sim._track_hw_access(base_addr, "write", [(value >> (8 * i)) & 0xff for i in range(size)])
            self.sim._mmio_state_write(addr, size, value, reg=reg)
            return

        if not self.sim.is_hardware_region(addr):
            self.sim._write_memory(addr, value.to_bytes(size, "little"))
            if should_halt:
                self.sim.tohost_value = self.sim._read_tohost_value(value)
                self.sim._halt("tohost", self.sim.tohost_value)
            return

        self.sim._track_hw_access(addr, "write", [(value >> (8 * i)) & 0xff for i in range(size)])
        if should_halt:
            self.sim.tohost_value = self.sim._read_tohost_value(value)
            self.sim._halt("tohost", self.sim.tohost_value)

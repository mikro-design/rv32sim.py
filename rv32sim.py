# rv32sim.py
# Complete RV32I simulator with fully working GDB remote stub
# Fixed all known issues: vMustReplyEmpty, qXfer conflicting responses, malformed packet crashes

import struct
import socket
import threading
import binascii
import time
import sys
import os


class MemoryRegion:
    def __init__(self, start, end, name="mem"):
        if end <= start:
            raise ValueError(f"Invalid memory region {name}: 0x{start:08x}-0x{end:08x}")
        self.start = start
        self.end = end
        self.name = name
        self.data = bytearray(end - start)

class HaltException(Exception):
    def __init__(self, reason, code=None):
        message = f"{reason}: {code}" if code is not None else reason
        super().__init__(message)
        self.reason = reason
        self.code = code

class TrapException(Exception):
    def __init__(self, cause, tval=0):
        super().__init__(f"trap {cause}")
        self.cause = cause
        self.tval = tval

class RV32Sim:
    def __init__(self, memory_size=16 * 1024 * 1024, gdb_port=3333, detect_mode=False, assisted_mode=False):
        self.memory_size = memory_size
        self.regs = [0] * 32
        self.pc = 0
        self.instr_count = 0
        self.read_handlers = {}
        self.write_handlers = {}
        self.breakpoints = set()
        self._skip_breakpoint_once = False
        self.elf_loaded = False
        self.elf_path = None

        # Hardware register stubbing
        self.detect_mode = detect_mode
        self.assisted_mode = assisted_mode  # Interactive mode that reports each new hardware access
        self.hw_accesses = {}  # Track all hardware register accesses
        self.hw_stubs = {}  # Stubbed hardware register values
        self.func_stubs = {}  # Function entry stubs (addr -> return value)
        self.memory_regions = []  # List of MemoryRegion instances for valid memory regions
        self._memory_initialized = False
        self.tohost_addr = None
        self.tohost_size = 0
        self.tohost_value = None
        self.exit_code = None
        self.halt_reason = None
        self.priv = 3
        self.last_instr = None
        self.supports_s_mode = False
        self.readonly_zero_csrs = {0x7a0, 0x7a1, 0x7a2, 0x7a3, 0x7a5}
        self.mcycle = 0
        self.mcycle_suppress = False
        self.minstret = 0
        self.minstret_suppress = False
        self.pmpcfg0 = 0
        self.pmpaddr0 = 0
        self.pmp_granularity = 2

        # CSR registers (Control and Status Registers)
        self.csrs = {}
        # Common CSRs
        self.csrs[0x300] = 0x1800  # mstatus (MPP=M by default)
        self.csrs[0x301] = (1 << 30) | (1 << 12) | (1 << 8) | (1 << 2)  # misa: RV32IMC
        self.csrs[0x304] = 0  # mie (machine interrupt enable)
        self.csrs[0x305] = 0  # mtvec (machine trap vector)
        self.csrs[0x340] = 0  # mscratch
        self.csrs[0x341] = 0  # mepc (machine exception PC)
        self.csrs[0x342] = 0  # mcause
        self.csrs[0x343] = 0  # mtval
        self.csrs[0x344] = 0  # mip (machine interrupt pending)
        self.csrs[0xf11] = 0  # mvendorid
        self.csrs[0xf12] = 0  # marchid
        self.csrs[0xf13] = 0  # mimpid
        self.csrs[0xf14] = 0  # mhartid
        self.csrs[0x320] = 0  # mcountinhibit
        self.csrs[0x7c0] = 0  # Custom CSR used by code

        self.gdb_port = gdb_port
        self.gdb_thread = None
        self.gdb_client = None
        self.running_gdb = False

    def step(self):
        try:
            self.execute()
            return True
        except Exception as e:
            print(f"[SIM] Execution stopped: {e}")
            return False

    def add_breakpoint(self, addr):
        print(f"[SIM] Adding breakpoint at 0x{addr:08x}")
        self.breakpoints.add(addr & 0xffffffff)

    def remove_breakpoint(self, addr):
        self.breakpoints.discard(addr & 0xffffffff)

    def dump_regs(self):
        names = ["zero","ra","sp","gp","tp","t0","t1","t2","fp","s1",
                 "a0","a1","a2","a3","a4","a5","a6","a7",
                 "s2","s3","s4","s5","s6","s7","s8","s9","s10","s11",
                 "t3","t4","t5","t6"]
        for i in range(32):
            print(f"x{i:2} ({names[i]:>6}) = 0x{self.regs[i]:08x}")
        print(f"pc             = 0x{self.pc:08x}")

    def register_read_handler(self, addr, func, width=1):
        self.read_handlers[addr] = {"func": func, "width": width}

    def register_write_handler(self, addr, func, width=1):
        self.write_handlers[addr] = {"func": func, "width": width}

    def configure_tohost(self, addr, size=8):
        self.tohost_addr = addr & 0xffffffff
        self.tohost_size = size

    def _u32(self, val):
        return val & 0xffffffff

    def _s32(self, val):
        val &= 0xffffffff
        return val - 0x100000000 if val & 0x80000000 else val

    def _clz32(self, val):
        val &= 0xffffffff
        return 32 if val == 0 else 32 - val.bit_length()

    def _ctz32(self, val):
        val &= 0xffffffff
        if val == 0:
            return 32
        return (val & -val).bit_length() - 1

    def _cpop32(self, val):
        return (val & 0xffffffff).bit_count()

    def _mask_mstatus(self, value):
        value &= 0xffffffff
        if not self.supports_s_mode:
            value &= ~0x1800
            value |= 0x1800
        return value

    def _pmpaddr0_read(self):
        value = self.pmpaddr0 & 0xffffffff
        value &= ~0x1
        if (self.pmpcfg0 & 0x18) == 0:
            value &= ~(1 << (self.pmp_granularity - 1))
        return value & 0xffffffff

    def _csr_read(self, csr):
        if csr in self.readonly_zero_csrs:
            return 0
        if csr in (0xB00, 0xC00, 0xB01, 0xC01):
            return self.mcycle & 0xffffffff
        if csr in (0xB80, 0xC80, 0xB81, 0xC81):
            return (self.mcycle >> 32) & 0xffffffff
        if csr in (0xB02, 0xC02):
            return self.minstret & 0xffffffff
        if csr in (0xB82, 0xC82):
            return (self.minstret >> 32) & 0xffffffff
        if csr == 0x3A0:
            return self.pmpcfg0 & 0xffffffff
        if csr == 0x3B0:
            return self._pmpaddr0_read()
        return self.csrs.get(csr, 0) & 0xffffffff

    def _csr_write(self, csr, value):
        value &= 0xffffffff
        if csr in self.readonly_zero_csrs:
            return
        if csr in (0xB00, 0xC00, 0xB01, 0xC01):
            self.mcycle = (self.mcycle & 0xffffffff00000000) | value
            self.mcycle_suppress = True
            return
        if csr in (0xB80, 0xC80, 0xB81, 0xC81):
            self.mcycle = (self.mcycle & 0xffffffff) | (value << 32)
            self.mcycle_suppress = True
            return
        if csr in (0xB02, 0xC02):
            self.minstret = (self.minstret & 0xffffffff00000000) | value
            self.minstret_suppress = True
            return
        if csr in (0xB82, 0xC82):
            self.minstret = (self.minstret & 0xffffffff) | (value << 32)
            self.minstret_suppress = True
            return
        if csr == 0x3A0:
            self.pmpcfg0 = value
            return
        if csr == 0x3B0:
            self.pmpaddr0 = value & ~0x1
            return
        if csr == 0x300:
            self.csrs[0x300] = self._mask_mstatus(value)
            return
        if csr == 0x301:
            return
        self.csrs[csr] = value

    def _init_default_memory_regions(self):
        self.memory_regions = []
        self._add_memory_region(0x00000000, 0x00020000, "flash")
        sram_end = 0x20000000 + self.memory_size
        self._add_memory_region(0x20000000, sram_end, "sram")
        self._memory_initialized = True

    def _ensure_memory_regions(self):
        if not self.memory_regions and not self._memory_initialized:
            self._init_default_memory_regions()

    def _add_memory_region(self, start, end, name="mem"):
        start &= 0xffffffff
        end &= 0xffffffff
        region = MemoryRegion(start, end, name)
        for existing in self.memory_regions:
            if not (region.end <= existing.start or region.start >= existing.end):
                raise ValueError(
                    f"Overlapping memory regions: {existing.name} "
                    f"0x{existing.start:08x}-0x{existing.end:08x} and "
                    f"{name} 0x{start:08x}-0x{end:08x}"
                )
        self.memory_regions.append(region)
        return region

    def _find_region(self, addr):
        self._ensure_memory_regions()
        for region in self.memory_regions:
            if region.start <= addr < region.end:
                return region
        return None

    def _read_memory(self, addr, size):
        data = bytearray()
        remaining = size
        cur = addr
        while remaining:
            region = self._find_region(cur)
            if region is None:
                raise ValueError(f"Memory read out of range at 0x{cur:08x}")
            offset = cur - region.start
            chunk = min(remaining, region.end - cur)
            data += region.data[offset:offset + chunk]
            cur += chunk
            remaining -= chunk
        return int.from_bytes(data, "little")

    def _write_memory(self, addr, data):
        remaining = len(data)
        cur = addr
        idx = 0
        while remaining:
            region = self._find_region(cur)
            if region is None:
                raise ValueError(f"Memory write out of range at 0x{cur:08x}")
            offset = cur - region.start
            chunk = min(remaining, region.end - cur)
            region.data[offset:offset + chunk] = data[idx:idx + chunk]
            cur += chunk
            idx += chunk
            remaining -= chunk

    def get_stack_top(self):
        self._ensure_memory_regions()
        sram = None
        for region in self.memory_regions:
            if region.name.lower() == "sram":
                sram = region
                break
        if sram:
            return sram.end
        if self.memory_regions:
            return max(self.memory_regions, key=lambda r: r.end).end
        return 0

    def _find_stub_entry(self, addr):
        if addr in self.hw_stubs:
            entry = self.hw_stubs[addr]
            width = entry.get("width", 4) if isinstance(entry, dict) else 4
            return addr, entry, width
        base = addr & ~0x3
        if base in self.hw_stubs:
            entry = self.hw_stubs[base]
            width = entry.get("width", 4) if isinstance(entry, dict) else 4
            if base <= addr < base + width:
                return base, entry, width
        base = addr & ~0x1
        if base in self.hw_stubs:
            entry = self.hw_stubs[base]
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

    def _track_hw_access(self, addr, access_type, values=None):
        if not (self.detect_mode or self.assisted_mode):
            return
        is_new = addr not in self.hw_accesses
        if is_new:
            entry = {"type": access_type, "count": 0, "pcs": set()}
            if access_type in ("read", "read/write"):
                entry["values_read"] = []
            if access_type in ("write", "read/write"):
                entry["values_written"] = []
            self.hw_accesses[addr] = entry
        else:
            if self.hw_accesses[addr]["type"] != access_type:
                self.hw_accesses[addr]["type"] = "read/write"
        self.hw_accesses[addr]["count"] += 1
        self.hw_accesses[addr]["pcs"].add(self.pc)

        if values is not None:
            if isinstance(values, int):
                values = [values]
            if access_type == "read" and "values_read" in self.hw_accesses[addr]:
                self.hw_accesses[addr]["values_read"].extend([v & 0xff for v in values])
            if access_type == "write" and "values_written" in self.hw_accesses[addr]:
                self.hw_accesses[addr]["values_written"].extend([v & 0xff for v in values])

        if self.assisted_mode and is_new:
            if access_type == "read":
                print(f"\n[ASSIST] New hardware READ at 0x{addr:08x} from PC=0x{self.pc:08x}")
                print("[ASSIST] Returning 0 (default). Add stub to config if different value needed.")
                self.hw_accesses[addr]["hint"] = "Read access - returns 0 by default"
            elif access_type == "write":
                val = values if isinstance(values, int) else (values[0] if values else 0)
                print(
                    f"\n[ASSIST] New hardware WRITE at 0x{addr:08x} from PC=0x{self.pc:08x}, "
                    f"value=0x{val:02x}"
                )
                print("[ASSIST] Write-only register - may not need stub unless also read")
                self.hw_accesses[addr]["hint"] = f"Write access with value 0x{val:02x}"

    def is_hardware_region(self, addr):
        """Check if address is outside configured memory regions"""
        return self._find_region(addr) is None

    def load_stub_config(self, filename):
        """Load hardware register stubs from config file"""
        import json
        try:
            with open(filename, 'r') as f:
                config = json.load(f)

            # Load memory regions if specified
            regions = config.get('memory_regions')
            if regions is not None:
                self.memory_regions = []
                self._memory_initialized = True
                for region in regions:
                    start = int(region['start'], 16) if isinstance(region['start'], str) else region['start']
                    end = int(region['end'], 16) if isinstance(region['end'], str) else region['end']
                    name = region.get('name', 'unknown')
                    self._add_memory_region(start, end, name)
                    print(f"[SIM] Memory region: {name} 0x{start:08x}-0x{end:08x}")

            # Load hardware stubs
            for addr_str, value in config.get('hw_stubs', {}).items():
                addr = int(addr_str, 16) if isinstance(addr_str, str) else addr_str
                self.hw_stubs[addr] = value
            print(f"[SIM] Loaded {len(self.hw_stubs)} hardware stubs from {filename}")

            # Load function stubs (entry PC -> return value)
            for addr_str, stub in config.get('func_stubs', {}).items():
                addr = int(addr_str, 16) if isinstance(addr_str, str) else addr_str
                if isinstance(stub, dict):
                    ret = stub.get("value", 0)
                else:
                    ret = stub
                self.func_stubs[addr] = ret
            if self.func_stubs:
                print(f"[SIM] Loaded {len(self.func_stubs)} function stubs from {filename}")

            if 'tohost_addr' in config:
                addr = config['tohost_addr']
                addr = int(addr, 16) if isinstance(addr, str) else addr
                size = config.get('tohost_size', 8)
                self.configure_tohost(addr, size=size)

            if 'fromhost_addr' in config:
                addr = config['fromhost_addr']
                addr = int(addr, 16) if isinstance(addr, str) else addr
                self.hw_stubs.setdefault(addr, {"value": 0, "width": 8})
        except FileNotFoundError:
            print(f"[SIM] No stub config found: {filename}")
        except Exception as e:
            print(f"[SIM] Error loading stub config: {e}")

    def save_hw_accesses(self, filename):
        """Save detected hardware accesses to a template config file"""
        import json

        # Build the stub entries with hints and detected values
        stubs = {}
        for addr, info in sorted(self.hw_accesses.items()):
            stub = {
                "value": 0,
                "access_type": info['type'],
                "access_count": info['count'],
                "pc_locations": [f"0x{pc:08x}" for pc in sorted(info['pcs'])]
            }
            if 'hint' in info:
                stub['hint'] = info['hint']
            if 'values_written' in info and info['values_written']:
                stub['values_written'] = [f"0x{v:02x}" for v in info['values_written'][:10]]  # First 10
            if 'values_read' in info and info['values_read']:
                stub['values_read'] = [f"0x{v:02x}" for v in info['values_read'][:10]]  # First 10
            stubs[f"0x{addr:08x}"] = stub

        config = {"hw_stubs": stubs}
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"[SIM] Saved hardware access map to {filename}")
        print(f"[SIM] Detected {len(self.hw_accesses)} unique hardware register accesses")

    def load_elf(self, filename):
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"ELF file not found: {filename}")

        with open(filename, 'rb') as f:
            elf = f.read()

        if len(elf) < 52 or elf[:4] != b'\x7fELF':
            raise ValueError("Invalid ELF file")
        if elf[4] != 1 or elf[5] != 1:
            raise ValueError("Unsupported ELF class or endianness (expected ELF32 little-endian)")

        # ELF32 header: e_ident[16], e_type(H), e_machine(H), e_version(I), e_entry(I), e_phoff(I), e_shoff(I), e_flags(I), e_ehsize(H), e_phentsize(H), e_phnum(H), e_shentsize(H), e_shnum(H), e_shstrndx(H)
        e_type, e_machine, e_version, e_entry, e_phoff = struct.unpack('<16xHHIII', elf[:32])
        e_ehsize, e_phentsize, e_phnum = struct.unpack('<HHH', elf[40:46])
        if e_phentsize < 32:
            raise ValueError("Invalid program header entry size")
        if e_phoff + e_phentsize * e_phnum > len(elf):
            raise ValueError("Program headers extend beyond EOF")

        if e_machine != 243:  # EM_RISCV
            raise ValueError("Not a RISC-V ELF")

        self.pc = e_entry & 0xffffffff

        for i in range(e_phnum):
            off = e_phoff + i * e_phentsize
            if off + e_phentsize > len(elf):
                raise ValueError("Program header extends beyond EOF")
            ph = struct.unpack('<IIIIIIII', elf[off:off+32])
            p_type, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz = ph[0], ph[1], ph[2], ph[3], ph[4], ph[5]
            if p_type == 1:  # PT_LOAD
                if p_memsz < p_filesz:
                    raise ValueError("Invalid segment sizes")
                if p_offset + p_filesz > len(elf):
                    raise ValueError("Segment extends beyond EOF")
                end = p_vaddr + p_memsz
                data = elf[p_offset:p_offset + p_filesz]
                self._write_memory(p_vaddr, data)
                if p_memsz > p_filesz:
                    self._write_memory(p_vaddr + p_filesz, b'\x00' * (p_memsz - p_filesz))
                if p_paddr != p_vaddr:
                    self._write_memory(p_paddr, data)

        self.elf_loaded = True
        self.elf_path = os.path.abspath(filename)
        print(f"[SIM] Loaded ELF: {self.elf_path}")
        print(f"[SIM] Entry point: 0x{self.pc:08x}")

    def sign_extend(self, val, bits):
        return (val & ((1 << bits) - 1)) - (1 << bits) if (val & (1 << (bits - 1))) else val

    def _div_trunc(self, a, b):
        if b == 0:
            return None
        if a == -0x80000000 and b == -1:
            return -0x80000000
        sign = -1 if (a < 0) ^ (b < 0) else 1
        return sign * (abs(a) // abs(b))

    def _rem_trunc(self, a, b):
        if b == 0:
            return a
        if a == -0x80000000 and b == -1:
            return 0
        return a - self._div_trunc(a, b) * b

    def _read_value(self, addr, size):
        addr &= 0xffffffff

        base, handler, width = self._find_handler_entry(addr, self.read_handlers)
        if handler:
            if size <= width and addr + size <= base + width:
                full = self._u32(self._call_read_handler(handler, addr, size))
                shift = (addr - base) * 8
                mask = (1 << (size * 8)) - 1
                value = (full >> shift) & mask
                self._track_hw_access(base, "read", [(value >> (8 * i)) & 0xff for i in range(size)])
                return value
            if width == 1 and size > 1:
                value = 0
                bytes_read = []
                for i in range(size):
                    b = self._u32(self._call_read_handler(handler, addr + i, 1)) & 0xff
                    value |= b << (8 * i)
                    bytes_read.append(b)
                self._track_hw_access(base, "read", bytes_read)
                return value

        base, stub, width = self._find_stub_entry(addr)
        if stub is not None and size <= width and addr + size <= base + width:
            full = self._stub_read_value(stub)
            full = self._u32(full)
            shift = (addr - base) * 8
            mask = (1 << (size * 8)) - 1
            value = (full >> shift) & mask
            self._track_hw_access(base, "read", [(value >> (8 * i)) & 0xff for i in range(size)])
            return value

        if not self.is_hardware_region(addr):
            return self._read_memory(addr, size)

        self._track_hw_access(addr, "read", [0] * size)
        return 0

    def _write_value(self, addr, size, value):
        addr &= 0xffffffff
        value = self._u32(value)
        value &= (1 << (size * 8)) - 1
        should_halt = self.tohost_addr is not None and addr == self.tohost_addr
        base, handler, width = self._find_handler_entry(addr, self.write_handlers)
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
                self._track_hw_access(base, "write", bytes_written)
                if should_halt:
                    self.tohost_value = self._read_tohost_value(value)
                    self._halt("tohost", self.tohost_value)
                return
            if width == 1 and size > 1:
                bytes_written = []
                for i in range(size):
                    b = (value >> (8 * i)) & 0xff
                    self._call_write_handler(handler, addr + i, b, 1)
                    bytes_written.append(b)
                self._track_hw_access(base, "write", bytes_written)
                if should_halt:
                    self.tohost_value = self._read_tohost_value(value)
                    self._halt("tohost", self.tohost_value)
                return

        base, stub, width = self._find_stub_entry(addr)
        if stub is not None and size <= width and addr + size <= base + width:
            cur = stub.get("value", 0) if isinstance(stub, dict) else stub
            cur = self._u32(cur)
            bytes_written = []
            for i in range(size):
                b = (value >> (8 * i)) & 0xff
                shift = (addr - base + i) * 8
                cur = (cur & ~(0xff << shift)) | (b << shift)
                bytes_written.append(b)
            if isinstance(stub, dict):
                stub["value"] = cur
            else:
                self.hw_stubs[base] = cur
            self._track_hw_access(base, "write", bytes_written)
            if should_halt:
                self.tohost_value = self._read_tohost_value(value)
                self._halt("tohost", self.tohost_value)
            return

        if not self.is_hardware_region(addr):
            self._write_memory(addr, value.to_bytes(size, "little"))
            if should_halt:
                self.tohost_value = self._read_tohost_value(value)
                self._halt("tohost", self.tohost_value)
            return

        self._track_hw_access(addr, "write", [(value >> (8 * i)) & 0xff for i in range(size)])
        if should_halt:
            self.tohost_value = self._read_tohost_value(value)
            self._halt("tohost", self.tohost_value)

    def load_byte(self, addr):
        return self._read_value(addr, 1)

    def load_half(self, addr):
        return self._read_value(addr, 2)

    def load_word(self, addr):
        return self._read_value(addr, 4)

    def store_byte(self, addr, val):
        self._write_value(addr, 1, val)

    def store_half(self, addr, val):
        self._write_value(addr, 2, val)

    def store_word(self, addr, val):
        self._write_value(addr, 4, val)

    def fetch(self):
        # Check if this is a compressed instruction (lowest 2 bits != 11)
        if self.pc & 0x1:
            raise TrapException(0, self.pc & 0xffffffff)
        first_half = self.load_half(self.pc)
        if (first_half & 0x3) != 0x3:
            return first_half  # 16-bit compressed instruction
        else:
            second_half = self.load_half(self.pc + 2)
            return first_half | (second_half << 16)  # 32-bit instruction

    def expand_compressed(self, c_instr):
        """Expand 16-bit compressed instruction to 32-bit"""
        quadrant = c_instr & 0x3
        funct3 = (c_instr >> 13) & 0x7

        def jal_imm_from_c(imm):
            return (((imm >> 20) & 0x1) << 31) | (((imm >> 1) & 0x3ff) << 21) | \
                   (((imm >> 11) & 0x1) << 20) | (((imm >> 12) & 0xff) << 12)

        def branch_imm_from_c(imm):
            imm &= 0x1fff
            return (((imm >> 12) & 0x1) << 31) | (((imm >> 5) & 0x3f) << 25) | \
                   (((imm >> 1) & 0xf) << 8) | (((imm >> 11) & 0x1) << 7)

        if quadrant == 0:  # C0
            rd_p = ((c_instr >> 2) & 0x7) + 8
            rs1_p = ((c_instr >> 7) & 0x7) + 8

            if funct3 == 0x0:  # C.ADDI4SPN
                imm = (((c_instr >> 7) & 0xF) << 6) | (((c_instr >> 12) & 0x1) << 5) | \
                      (((c_instr >> 11) & 0x1) << 4) | (((c_instr >> 5) & 0x1) << 3) | \
                      (((c_instr >> 6) & 0x1) << 2)
                if imm == 0:
                    self._illegal_instruction(c_instr)
                return (imm << 20) | (2 << 15) | (0 << 12) | (rd_p << 7) | 0x13  # addi rd', x2, imm
            elif funct3 == 0x2:  # C.LW
                imm = (((c_instr >> 6) & 0x1) << 2) | (((c_instr >> 10) & 0x7) << 3) | (((c_instr >> 5) & 0x1) << 6)
                return (imm << 20) | (rs1_p << 15) | (0x2 << 12) | (rd_p << 7) | 0x03  # lw
            elif funct3 == 0x6:  # C.SW
                imm = (((c_instr >> 6) & 0x1) << 2) | (((c_instr >> 10) & 0x7) << 3) | (((c_instr >> 5) & 0x1) << 6)
                rs2_p = ((c_instr >> 2) & 0x7) + 8
                imm_11_5 = (imm >> 5) & 0x7f
                imm_4_0 = imm & 0x1f
                return (imm_11_5 << 25) | (rs2_p << 20) | (rs1_p << 15) | (0x2 << 12) | (imm_4_0 << 7) | 0x23  # sw
            else:
                self._illegal_instruction(c_instr)

        elif quadrant == 1:  # C1
            rd_rs1 = (c_instr >> 7) & 0x1f
            rd_p = ((c_instr >> 7) & 0x7) + 8
            rs2_p = ((c_instr >> 2) & 0x7) + 8

            if funct3 == 0x0:  # C.ADDI or C.NOP
                imm = (((c_instr >> 12) & 0x1) << 5) | ((c_instr >> 2) & 0x1f)
                imm = self.sign_extend(imm, 6)
                return (imm << 20) | (rd_rs1 << 15) | (0 << 12) | (rd_rs1 << 7) | 0x13  # addi
            elif funct3 == 0x1:  # C.JAL
                imm = (((c_instr >> 12) & 0x1) << 11) | (((c_instr >> 11) & 0x1) << 4) | \
                      (((c_instr >> 9) & 0x3) << 8) | (((c_instr >> 8) & 0x1) << 10) | \
                      (((c_instr >> 7) & 0x1) << 6) | (((c_instr >> 6) & 0x1) << 7) | \
                      (((c_instr >> 3) & 0x7) << 1) | (((c_instr >> 2) & 0x1) << 5)
                imm = self.sign_extend(imm, 12)
                return jal_imm_from_c(imm) | (1 << 7) | 0x6f  # jal x1, imm
            elif funct3 == 0x2:  # C.LI
                imm = (((c_instr >> 12) & 0x1) << 5) | ((c_instr >> 2) & 0x1f)
                imm = self.sign_extend(imm, 6)
                return (imm << 20) | (0 << 15) | (0 << 12) | (rd_rs1 << 7) | 0x13  # addi rd, x0, imm
            elif funct3 == 0x3:  # C.ADDI16SP / C.LUI
                if rd_rs1 == 2:  # C.ADDI16SP
                    imm = (((c_instr >> 12) & 0x1) << 9) | (((c_instr >> 6) & 0x1) << 4) | \
                          (((c_instr >> 5) & 0x1) << 6) | (((c_instr >> 3) & 0x3) << 7) | \
                          (((c_instr >> 2) & 0x1) << 5)
                    imm = self.sign_extend(imm, 10)
                    if imm == 0:
                        self._illegal_instruction(c_instr)
                    return (imm << 20) | (2 << 15) | (0 << 12) | (2 << 7) | 0x13  # addi x2, x2, imm
                if rd_rs1 == 0:
                    self._illegal_instruction(c_instr)
                imm = (((c_instr >> 12) & 0x1) << 5) | ((c_instr >> 2) & 0x1f)
                imm = self.sign_extend(imm, 6) << 12
                if imm == 0:
                    self._illegal_instruction(c_instr)
                return (imm & 0xfffff000) | (rd_rs1 << 7) | 0x37  # lui
            elif funct3 == 0x4:
                op = (c_instr >> 10) & 0x3
                bit12 = (c_instr >> 12) & 0x1
                if op == 0x0:  # C.SRLI
                    if bit12:
                        self._illegal_instruction(c_instr)
                    shamt = (c_instr >> 2) & 0x1f
                    return (shamt << 20) | (rd_p << 15) | (0x5 << 12) | (rd_p << 7) | 0x13
                elif op == 0x1:  # C.SRAI
                    if bit12:
                        self._illegal_instruction(c_instr)
                    shamt = (c_instr >> 2) & 0x1f
                    imm = (0x20 << 5) | shamt
                    return (imm << 20) | (rd_p << 15) | (0x5 << 12) | (rd_p << 7) | 0x13
                elif op == 0x2:  # C.ANDI
                    imm = (((c_instr >> 12) & 0x1) << 5) | ((c_instr >> 2) & 0x1f)
                    imm = self.sign_extend(imm, 6)
                    return (imm << 20) | (rd_p << 15) | (0x7 << 12) | (rd_p << 7) | 0x13
                elif op == 0x3:  # C.SUB/C.XOR/C.OR/C.AND
                    if bit12:
                        self._illegal_instruction(c_instr)
                    funct2 = (c_instr >> 5) & 0x3
                    if funct2 == 0x0:  # C.SUB
                        return (0x20 << 25) | (rs2_p << 20) | (rd_p << 15) | (0x0 << 12) | (rd_p << 7) | 0x33
                    elif funct2 == 0x1:  # C.XOR
                        return (0x00 << 25) | (rs2_p << 20) | (rd_p << 15) | (0x4 << 12) | (rd_p << 7) | 0x33
                    elif funct2 == 0x2:  # C.OR
                        return (0x00 << 25) | (rs2_p << 20) | (rd_p << 15) | (0x6 << 12) | (rd_p << 7) | 0x33
                    elif funct2 == 0x3:  # C.AND
                        return (0x00 << 25) | (rs2_p << 20) | (rd_p << 15) | (0x7 << 12) | (rd_p << 7) | 0x33
                    self._illegal_instruction(c_instr)
            elif funct3 == 0x5:  # C.J
                imm = (((c_instr >> 12) & 0x1) << 11) | (((c_instr >> 11) & 0x1) << 4) | \
                      (((c_instr >> 9) & 0x3) << 8) | (((c_instr >> 8) & 0x1) << 10) | \
                      (((c_instr >> 7) & 0x1) << 6) | (((c_instr >> 6) & 0x1) << 7) | \
                      (((c_instr >> 3) & 0x7) << 1) | (((c_instr >> 2) & 0x1) << 5)
                imm = self.sign_extend(imm, 12)
                return jal_imm_from_c(imm) | (0 << 7) | 0x6f  # jal x0, imm
            elif funct3 == 0x6:  # C.BEQZ
                imm = (((c_instr >> 12) & 0x1) << 8) | (((c_instr >> 10) & 0x3) << 3) | \
                      (((c_instr >> 5) & 0x3) << 6) | (((c_instr >> 3) & 0x3) << 1) | \
                      (((c_instr >> 2) & 0x1) << 5)
                imm = self.sign_extend(imm, 9)
                return branch_imm_from_c(imm) | (0 << 20) | (rd_p << 15) | (0x0 << 12) | 0x63
            elif funct3 == 0x7:  # C.BNEZ
                imm = (((c_instr >> 12) & 0x1) << 8) | (((c_instr >> 10) & 0x3) << 3) | \
                      (((c_instr >> 5) & 0x3) << 6) | (((c_instr >> 3) & 0x3) << 1) | \
                      (((c_instr >> 2) & 0x1) << 5)
                imm = self.sign_extend(imm, 9)
                return branch_imm_from_c(imm) | (0 << 20) | (rd_p << 15) | (0x1 << 12) | 0x63
            else:
                self._illegal_instruction(c_instr)

        elif quadrant == 2:  # C2
            rd_rs1 = (c_instr >> 7) & 0x1f
            rs2 = (c_instr >> 2) & 0x1f

            if funct3 == 0x0:  # C.SLLI
                if rd_rs1 == 0:
                    self._illegal_instruction(c_instr)
                if (c_instr >> 12) & 0x1:
                    self._illegal_instruction(c_instr)
                shamt = (c_instr >> 2) & 0x1f
                return (shamt << 20) | (rd_rs1 << 15) | (0x1 << 12) | (rd_rs1 << 7) | 0x13
            elif funct3 == 0x2:  # C.LWSP
                if rd_rs1 == 0:
                    self._illegal_instruction(c_instr)
                imm = (((c_instr >> 12) & 0x1) << 5) | (((c_instr >> 4) & 0x7) << 2) | \
                      (((c_instr >> 2) & 0x3) << 6)
                return (imm << 20) | (2 << 15) | (0x2 << 12) | (rd_rs1 << 7) | 0x03
            elif funct3 == 0x4:
                bit12 = (c_instr >> 12) & 0x1
                if bit12 == 0:
                    if rs2 == 0:
                        if rd_rs1 == 0:
                            self._illegal_instruction(c_instr)
                        return (0 << 20) | (rd_rs1 << 15) | (0 << 12) | (0 << 7) | 0x67  # jalr x0, 0(rs1)
                    return (0 << 25) | (rs2 << 20) | (0 << 15) | (0 << 12) | (rd_rs1 << 7) | 0x33  # add rd, x0, rs2
                else:
                    if rs2 == 0:
                        if rd_rs1 == 0:
                            return 0x00100073  # ebreak
                        return (0 << 20) | (rd_rs1 << 15) | (0 << 12) | (1 << 7) | 0x67  # jalr x1, 0(rs1)
                    return (0 << 25) | (rs2 << 20) | (rd_rs1 << 15) | (0 << 12) | (rd_rs1 << 7) | 0x33  # add rd, rd, rs2
            elif funct3 == 0x6:  # C.SWSP
                imm = (((c_instr >> 7) & 0x3) << 6) | (((c_instr >> 9) & 0xf) << 2)
                imm_11_5 = (imm >> 5) & 0x7f
                imm_4_0 = imm & 0x1f
                return (imm_11_5 << 25) | (rs2 << 20) | (2 << 15) | (0x2 << 12) | (imm_4_0 << 7) | 0x23
            else:
                self._illegal_instruction(c_instr)

        self._illegal_instruction(c_instr)

    def execute(self):
        if self.pc in self.func_stubs:
            ret = self.func_stubs[self.pc]
            self.last_instr = None
            self.instr_count += 1
            self.regs[10] = self._u32(ret)
            next_pc = self.regs[1] & 0xffffffff

            mcountinhibit = self.csrs.get(0x320, 0)
            if not self.mcycle_suppress and not (mcountinhibit & 0x1):
                self.mcycle = (self.mcycle + 1) & 0xffffffffffffffff
            if not self.minstret_suppress and not (mcountinhibit & 0x4):
                self.minstret = (self.minstret + 1) & 0xffffffffffffffff
            self.mcycle_suppress = False
            self.minstret_suppress = False

            for i in range(32):
                self.regs[i] &= 0xffffffff
            self.regs[0] = 0
            self.pc = next_pc
            return

        instr = self.fetch()
        self.last_instr = instr

        # Check if compressed instruction
        is_compressed = (instr & 0x3) != 0x3
        if is_compressed:
            instr = self.expand_compressed(instr)
            instr_size = 2
        else:
            instr_size = 4

        opcode = instr & 0x7f
        rd = (instr >> 7) & 0x1f
        funct3 = (instr >> 12) & 0x7
        rs1 = (instr >> 15) & 0x1f
        rs2 = (instr >> 20) & 0x1f
        funct7 = instr >> 25

        rs1_u = self.regs[rs1] & 0xffffffff
        rs2_u = self.regs[rs2] & 0xffffffff
        rs1_s = self._s32(rs1_u)
        rs2_s = self._s32(rs2_u)

        self.instr_count += 1
        next_pc = self.pc + instr_size

        if opcode == 0b0110011:  # R-type
            shamt = rs2_u & 0x1f
            if funct7 == 0x24:  # Zbs extension (bit manipulation - single-bit)
                if funct3 == 0x1:  # BCLR
                    self.regs[rd] = rs1_u & ~(1 << shamt)
                elif funct3 == 0x5:  # BEXT
                    self.regs[rd] = (rs1_u >> shamt) & 1
                else:
                    self._illegal_instruction()
            elif funct7 == 0x14:  # Zbs extension (more bit ops)
                if funct3 == 0x1:  # BSET
                    self.regs[rd] = rs1_u | (1 << shamt)
                else:
                    self._illegal_instruction()
            elif funct7 == 0x34:  # Zbs BINV
                if funct3 == 0x1:  # BINV
                    self.regs[rd] = rs1_u ^ (1 << shamt)
                else:
                    self._illegal_instruction()
            elif funct7 == 0x10:  # Zba extension (shift-and-add)
                if funct3 == 0x2:  # SH1ADD
                    self.regs[rd] = (rs2_u + ((rs1_u << 1) & 0xffffffff)) & 0xffffffff
                elif funct3 == 0x4:  # SH2ADD
                    self.regs[rd] = (rs2_u + ((rs1_u << 2) & 0xffffffff)) & 0xffffffff
                elif funct3 == 0x6:  # SH3ADD
                    self.regs[rd] = (rs2_u + ((rs1_u << 3) & 0xffffffff)) & 0xffffffff
                else:
                    self._illegal_instruction()
            elif funct7 == 0x05:  # Zbb extension (min/max)
                if funct3 == 0x4:  # MIN
                    self.regs[rd] = rs1_u if rs1_s < rs2_s else rs2_u
                elif funct3 == 0x5:  # MINU
                    self.regs[rd] = rs1_u if rs1_u < rs2_u else rs2_u
                elif funct3 == 0x6:  # MAX
                    self.regs[rd] = rs1_u if rs1_s > rs2_s else rs2_u
                elif funct3 == 0x7:  # MAXU
                    self.regs[rd] = rs1_u if rs1_u > rs2_u else rs2_u
                else:
                    self._illegal_instruction()
            elif funct7 == 0x04:  # Zbb extension (zext.h)
                if funct3 == 0x4 and rs2 == 0:  # ZEXT.H
                    self.regs[rd] = rs1_u & 0xffff
                else:
                    self._illegal_instruction()
            elif funct7 == 0x01:  # M extension (multiply/divide)
                if funct3 == 0x0:  # MUL
                    self.regs[rd] = (rs1_u * rs2_u) & 0xffffffff
                elif funct3 == 0x1:  # MULH (signed x signed, upper 32 bits)
                    prod = (rs1_s * rs2_s) & 0xffffffffffffffff
                    self.regs[rd] = (prod >> 32) & 0xffffffff
                elif funct3 == 0x2:  # MULHSU (signed x unsigned, upper 32 bits)
                    prod = (rs1_s * rs2_u) & 0xffffffffffffffff
                    self.regs[rd] = (prod >> 32) & 0xffffffff
                elif funct3 == 0x3:  # MULHU (unsigned x unsigned, upper 32 bits)
                    prod = (rs1_u * rs2_u) & 0xffffffffffffffff
                    self.regs[rd] = (prod >> 32) & 0xffffffff
                elif funct3 == 0x4:  # DIV
                    if rs2_u == 0:
                        self.regs[rd] = 0xffffffff
                    else:
                        self.regs[rd] = self._div_trunc(rs1_s, rs2_s) & 0xffffffff
                elif funct3 == 0x5:  # DIVU
                    self.regs[rd] = 0xffffffff if rs2_u == 0 else (rs1_u // rs2_u) & 0xffffffff
                elif funct3 == 0x6:  # REM
                    if rs2_u == 0:
                        self.regs[rd] = rs1_u
                    else:
                        self.regs[rd] = self._rem_trunc(rs1_s, rs2_s) & 0xffffffff
                elif funct3 == 0x7:  # REMU
                    self.regs[rd] = rs1_u if rs2_u == 0 else (rs1_u % rs2_u) & 0xffffffff
                else:
                    self._illegal_instruction()
            elif funct7 in (0x00, 0x20):
                if funct3 == 0x0:
                    if funct7 == 0x00:  # ADD
                        self.regs[rd] = rs1_u + rs2_u
                    elif funct7 == 0x20:  # SUB
                        self.regs[rd] = rs1_u - rs2_u
                elif funct3 == 0x1:
                    if funct7 != 0x00:
                        self._illegal_instruction()
                    self.regs[rd] = rs1_u << shamt
                elif funct3 == 0x2:
                    if funct7 != 0x00:
                        self._illegal_instruction()
                    self.regs[rd] = 1 if rs1_s < rs2_s else 0
                elif funct3 == 0x3:
                    if funct7 != 0x00:
                        self._illegal_instruction()
                    self.regs[rd] = 1 if rs1_u < rs2_u else 0
                elif funct3 == 0x4:
                    if funct7 != 0x00:
                        self._illegal_instruction()
                    self.regs[rd] = rs1_u ^ rs2_u
                elif funct3 == 0x5:
                    if funct7 == 0x00:  # SRL
                        self.regs[rd] = rs1_u >> shamt
                    elif funct7 == 0x20:  # SRA
                        self.regs[rd] = self._s32(rs1_u) >> shamt
                    else:
                        self._illegal_instruction()
                elif funct3 == 0x6:
                    if funct7 != 0x00:
                        self._illegal_instruction()
                    self.regs[rd] = rs1_u | rs2_u
                elif funct3 == 0x7:
                    if funct7 != 0x00:
                        self._illegal_instruction()
                    self.regs[rd] = rs1_u & rs2_u
                else:
                    self._illegal_instruction()
            else:
                self._illegal_instruction()

        elif opcode == 0b0010011:  # I-type
            imm = self.sign_extend(instr >> 20, 12)
            shamt = (instr >> 20) & 0x1f
            if funct3 == 0x0:  # ADDI
                self.regs[rd] = rs1_u + imm
            elif funct3 == 0x4:  # XORI
                self.regs[rd] = rs1_u ^ imm
            elif funct3 == 0x6:  # ORI
                self.regs[rd] = rs1_u | imm
            elif funct3 == 0x7:  # ANDI
                self.regs[rd] = rs1_u & imm
            elif funct3 == 0x1:
                if funct7 == 0x24:  # BCLRI
                    self.regs[rd] = rs1_u & ~(1 << shamt)
                elif funct7 == 0x14:  # BSETI
                    self.regs[rd] = rs1_u | (1 << shamt)
                elif funct7 == 0x34:  # BINVI
                    self.regs[rd] = rs1_u ^ (1 << shamt)
                elif funct7 == 0x30:  # Zbb immediate ops
                    if shamt == 0x00:  # CLZ
                        self.regs[rd] = self._clz32(rs1_u)
                    elif shamt == 0x01:  # CTZ
                        self.regs[rd] = self._ctz32(rs1_u)
                    elif shamt == 0x02:  # CPOP
                        self.regs[rd] = self._cpop32(rs1_u)
                    elif shamt == 0x04:  # SEXT.B
                        self.regs[rd] = self.sign_extend(rs1_u & 0xff, 8) & 0xffffffff
                    elif shamt == 0x05:  # SEXT.H
                        self.regs[rd] = self.sign_extend(rs1_u & 0xffff, 16) & 0xffffffff
                    else:
                        self._illegal_instruction()
                elif funct7 == 0x00:  # SLLI
                    self.regs[rd] = rs1_u << shamt
                else:
                    self._illegal_instruction()
            elif funct3 == 0x5:
                if funct7 == 0x24:  # BEXTI
                    self.regs[rd] = (rs1_u >> shamt) & 1
                elif funct7 == 0x00:  # SRLI
                    self.regs[rd] = rs1_u >> shamt
                elif funct7 == 0x20:  # SRAI
                    self.regs[rd] = self._s32(rs1_u) >> shamt
                else:
                    self._illegal_instruction()
            elif funct3 == 0x2:  # SLTI
                self.regs[rd] = 1 if rs1_s < imm else 0
            elif funct3 == 0x3:  # SLTIU
                self.regs[rd] = 1 if rs1_u < (imm & 0xffffffff) else 0
            else:
                self._illegal_instruction()

        elif opcode == 0b0000011:  # Loads
            imm = self.sign_extend(instr >> 20, 12)
            addr = (rs1_u + imm) & 0xffffffff
            if funct3 == 0x0:  # LB
                self.regs[rd] = self.sign_extend(self.load_byte(addr), 8)
            elif funct3 == 0x1:  # LH
                self.regs[rd] = self.sign_extend(self.load_half(addr), 16)
            elif funct3 == 0x2:  # LW
                self.regs[rd] = self.load_word(addr)
            elif funct3 == 0x4:  # LBU
                self.regs[rd] = self.load_byte(addr)
            elif funct3 == 0x5:  # LHU
                self.regs[rd] = self.load_half(addr)
            else:
                self._illegal_instruction()

        elif opcode == 0b0100011:  # Stores
            imm = self.sign_extend((funct7 << 5) | rd, 12)
            addr = (rs1_u + imm) & 0xffffffff
            if funct3 == 0x0:  # SB
                self.store_byte(addr, rs2_u)
            elif funct3 == 0x1:  # SH
                self.store_half(addr, rs2_u)
            elif funct3 == 0x2:  # SW
                self.store_word(addr, rs2_u)
            else:
                self._illegal_instruction()

        elif opcode == 0b0001111:  # FENCE/FENCE.I
            if funct3 in (0x0, 0x1):
                pass
            else:
                self._illegal_instruction()

        elif opcode == 0b1100011:  # Branches
            imm = self.sign_extend(((instr >> 31) << 12) | ((instr >> 7 & 1) << 11) | ((funct7 & 0x3f) << 5) | ((instr >> 8 & 0xf) << 1), 13)
            take = False
            if funct3 == 0x0:  # BEQ
                take = rs1_u == rs2_u
            elif funct3 == 0x1:  # BNE
                take = rs1_u != rs2_u
            elif funct3 == 0x4:  # BLT
                take = rs1_s < rs2_s
            elif funct3 == 0x5:  # BGE
                take = rs1_s >= rs2_s
            elif funct3 == 0x6:  # BLTU
                take = rs1_u < rs2_u
            elif funct3 == 0x7:  # BGEU
                take = rs1_u >= rs2_u
            else:
                self._illegal_instruction()
            if take:
                next_pc = self.pc + imm

        elif opcode == 0b1101111:  # JAL
            imm = self.sign_extend(((instr >> 31) << 20) | ((instr >> 12 & 0xff) << 12) | ((instr >> 20 & 1) << 11) | ((instr >> 21 & 0x3ff) << 1), 21)
            if rd:
                self.regs[rd] = self.pc + instr_size
            next_pc = self.pc + imm

        elif opcode == 0b1100111:  # JALR
            imm = self.sign_extend(instr >> 20, 12)
            if rd:
                self.regs[rd] = self.pc + instr_size
            next_pc = (rs1_u + imm) & ~1

        elif opcode == 0b0110111:  # LUI
            self.regs[rd] = instr & 0xfffff000

        elif opcode == 0b0010111:  # AUIPC
            self.regs[rd] = (self.pc + (instr & 0xfffff000)) & 0xffffffff

        elif opcode == 0b1110011:  # SYSTEM instructions (ecall/ebreak/CSR)
            if funct3 == 0x0:  # ecall/ebreak
                if instr == 0x00000073:  # ecall
                    a7 = self.regs[17] & 0xffffffff
                    if a7 == 93:  # exit
                        code = self.regs[10] & 0xffffffff
                        self.exit_code = code
                        self._halt("exit", code)
                    elif a7 == 64:  # write
                        fd, buf, cnt = self.regs[10], self.regs[11], self.regs[12]
                        if fd in (1, 2):
                            data = bytes(self.load_byte((buf + i) & 0xffffffff) for i in range(cnt))
                            sys.stdout.buffer.write(data)
                            sys.stdout.flush()
                            self.regs[10] = cnt
                        else:
                            self.regs[10] = 0xffffffff
                    else:
                        if self.priv == 3:
                            cause = 11
                        elif self.priv == 1:
                            cause = 9
                        else:
                            cause = 8
                        raise TrapException(cause, 0)
                elif instr == 0x00100073:  # ebreak
                    raise TrapException(3, 0)
                elif instr == 0x30200073:  # mret
                    mstatus = self._csr_read(0x300)
                    mpp = (mstatus >> 11) & 0x3
                    self.priv = mpp
                    self._csr_write(0x300, mstatus & ~0x1800)
                    next_pc = self.csrs.get(0x341, 0) & 0xffffffff
                elif instr == 0x10200073:  # sret
                    mstatus = self._csr_read(0x300)
                    spp = (mstatus >> 8) & 0x1
                    self.priv = 1 if spp else 0
                    self._csr_write(0x300, mstatus & ~0x100)
                    next_pc = self.csrs.get(0x141, 0) & 0xffffffff
                elif instr == 0x00200073:  # uret
                    self.priv = 0
                    next_pc = self.csrs.get(0x041, 0) & 0xffffffff
                elif instr == 0x10500073:  # wfi
                    pass
                else:
                    self._illegal_instruction()
            else:
                csr = (instr >> 20) & 0xfff
                zimm = rs1  # For immediate versions, rs1 field holds immediate value
                t = self._csr_read(csr)
                if funct3 == 0x1:  # CSRRW
                    self._csr_write(csr, rs1_u)
                    if rd != 0:
                        self.regs[rd] = t
                elif funct3 == 0x2:  # CSRRS
                    if rs1 != 0:
                        self._csr_write(csr, t | rs1_u)
                    if rd != 0:
                        self.regs[rd] = t
                elif funct3 == 0x3:  # CSRRC
                    if rs1 != 0:
                        self._csr_write(csr, t & ~rs1_u)
                    if rd != 0:
                        self.regs[rd] = t
                elif funct3 == 0x5:  # CSRRWI
                    self._csr_write(csr, zimm)
                    if rd != 0:
                        self.regs[rd] = t
                elif funct3 == 0x6:  # CSRRSI
                    if zimm != 0:
                        self._csr_write(csr, t | zimm)
                    if rd != 0:
                        self.regs[rd] = t
                elif funct3 == 0x7:  # CSRRCI
                    if zimm != 0:
                        self._csr_write(csr, t & ~zimm)
                    if rd != 0:
                        self.regs[rd] = t
                else:
                    self._illegal_instruction()

        else:
            self._illegal_instruction()

        mcountinhibit = self.csrs.get(0x320, 0)
        if not self.mcycle_suppress and not (mcountinhibit & 0x1):
            self.mcycle = (self.mcycle + 1) & 0xffffffffffffffff
        if not self.minstret_suppress and not (mcountinhibit & 0x4):
            self.minstret = (self.minstret + 1) & 0xffffffffffffffff
        self.mcycle_suppress = False
        self.minstret_suppress = False

        for i in range(32):
            self.regs[i] &= 0xffffffff
        self.regs[0] = 0
        self.pc = next_pc & 0xffffffff

    def _halt(self, reason, code=None):
        self.halt_reason = reason
        raise HaltException(reason, code)

    def _raise_trap(self, cause, tval=0):
        mstatus = self._csr_read(0x300)
        mstatus = (mstatus & ~0x1800) | ((self.priv & 0x3) << 11)
        self._csr_write(0x300, mstatus)
        self.csrs[0x341] = self.pc & 0xffffffff
        self.csrs[0x342] = cause & 0xffffffff
        self.csrs[0x343] = tval & 0xffffffff
        self.priv = 3
        return self.csrs.get(0x305, 0) & 0xffffffff

    def _illegal_instruction(self, instr=None):
        if instr is None:
            instr = self.last_instr if self.last_instr is not None else 0
        raise TrapException(2, instr & 0xffffffff)

    def _read_tohost_value(self, fallback):
        if self.tohost_addr is None or self.tohost_size <= 0:
            return fallback & 0xffffffff
        try:
            return self._read_memory(self.tohost_addr, self.tohost_size) & 0xffffffffffffffff
        except Exception:
            return fallback & 0xffffffff

    def start_gdb_server(self):
        self.running_gdb = True
        self.gdb_thread = threading.Thread(target=self._gdb_server_loop, daemon=True)
        self.gdb_thread.start()
        print(f"[SIM] GDB server listening on localhost:{self.gdb_port}")
        print("[SIM] Connect with:")
        print("    riscv32-unknown-elf-gdb your_program.elf")
        print(f"    (gdb) target remote localhost:{self.gdb_port}")

    def _gdb_server_loop(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('127.0.0.1', self.gdb_port))
        server.listen(1)

        while self.running_gdb:
            try:
                server.settimeout(1.0)
                client, _ = server.accept()
                print("[SIM] GDB connected")
                self.gdb_client = client
                try:
                    self._handle_gdb_session(client)
                finally:
                    self.gdb_client = None
                    try:
                        client.close()
                    except Exception:
                        pass
                print("[SIM] GDB disconnected")
            except socket.timeout:
                continue
            except Exception as e:
                if self.running_gdb:
                    print(f"[SIM] GDB server error: {e}")

    def _escape_rsp(self, data):
        escaped = []
        for c in data:
            if c in "$#}*":
                escaped.append("}")
                escaped.append(chr(ord(c) ^ 0x20))
            else:
                escaped.append(c)
        return "".join(escaped)

    def _unescape_rsp(self, data):
        out = []
        i = 0
        while i < len(data):
            if data[i] == "}":
                i += 1
                if i >= len(data):
                    break
                out.append(chr(ord(data[i]) ^ 0x20))
            else:
                out.append(data[i])
            i += 1
        return "".join(out)

    def _send_packet(self, data=""):
        if not self.gdb_client:
            return
        escaped = self._escape_rsp(data)
        checksum = sum(ord(c) for c in escaped) % 256
        packet = f"${escaped}#{checksum:02x}".encode('ascii')
        self.gdb_client.send(packet)

    def _memory_map_xml(self):
        self._ensure_memory_regions()
        lines = [
            '<?xml version="1.0"?>',
            '<!DOCTYPE memory-map PUBLIC "+//IDN gnu.org//DTD GDB Memory Map V1.0//EN" '
            '"http://sourceware.org/gdb/gdb-memory-map.dtd">',
            '<memory-map>'
        ]
        for region in self.memory_regions:
            start = region.start & 0xffffffff
            length = (region.end - region.start) & 0xffffffff
            lines.append(
                f'  <memory type="ram" start="0x{start:08x}" length="0x{length:x}"/>'
            )
        lines.append('</memory-map>')
        return "\n".join(lines)

    def _recv_packet(self):
        if not self.gdb_client:
            return None
        state = 0
        data = ""
        checksum = 0
        while True:
            try:
                b = self.gdb_client.recv(1)
                if not b:
                    return None
                c = b.decode('latin-1')
                if state == 0:
                    if c in "+-":
                        continue
                    if c == '$':
                        state = 1
                        data = ""
                        checksum = 0
                elif state == 1:
                    if c == '#':
                        state = 2
                    else:
                        data += c
                        checksum = (checksum + ord(c)) % 256
                elif state == 2:
                    chk_hi = int(c, 16) << 4
                    state = 3
                elif state == 3:
                    if (chk_hi + int(c, 16)) % 256 == checksum:
                        self.gdb_client.send(b'+')
                        return self._unescape_rsp(data)
                    else:
                        self.gdb_client.send(b'-')
                        state = 0
            except:
                return None

    def _stop_reply_from_exception(self, exc):
        msg = str(exc)
        if "Program exited with code" in msg:
            try:
                code = int(msg.split("code", 1)[1].strip().split()[0], 0) & 0xff
            except Exception:
                code = 0
            return f"W{code:02x}"
        return "S05"

    def _handle_gdb_session(self, conn):
        # Don't send unsolicited packet - wait for GDB to query with '?'

        while self.running_gdb:
            pkt = self._recv_packet()
            if pkt is None:
                break

            reply = ""

            if pkt == "?":
                reply = "S05"

            elif pkt in ("D", "k"):
                reply = "OK"
                self._send_packet(reply)
                break

            elif pkt.startswith("Hg") or pkt.startswith("Hc"):
                reply = "OK"

            elif "qSupported" in pkt:
                reply = "PacketSize=1000;qXfer:features:read+;qXfer:memory-map:read+"

            elif pkt == "vMustReplyEmpty" or pkt.startswith("v"):
                reply = ""

            elif pkt.startswith("qXfer:features:read:target.xml:"):
                target_xml = '''<?xml version="1.0"?>
<!DOCTYPE target SYSTEM "gdb-target.dtd">
<target>
  <architecture>riscv:rv32</architecture>
  <feature name="org.gnu.gdb.riscv.cpu">
    <reg name="zero" bitsize="32" regnum="0"/>
    <reg name="ra" bitsize="32" regnum="1"/>
    <reg name="sp" bitsize="32" regnum="2"/>
    <reg name="gp" bitsize="32" regnum="3"/>
    <reg name="tp" bitsize="32" regnum="4"/>
    <reg name="t0" bitsize="32" regnum="5"/>
    <reg name="t1" bitsize="32" regnum="6"/>
    <reg name="t2" bitsize="32" regnum="7"/>
    <reg name="s0" bitsize="32" regnum="8"/>
    <reg name="s1" bitsize="32" regnum="9"/>
    <reg name="a0" bitsize="32" regnum="10"/>
    <reg name="a1" bitsize="32" regnum="11"/>
    <reg name="a2" bitsize="32" regnum="12"/>
    <reg name="a3" bitsize="32" regnum="13"/>
    <reg name="a4" bitsize="32" regnum="14"/>
    <reg name="a5" bitsize="32" regnum="15"/>
    <reg name="a6" bitsize="32" regnum="16"/>
    <reg name="a7" bitsize="32" regnum="17"/>
    <reg name="s2" bitsize="32" regnum="18"/>
    <reg name="s3" bitsize="32" regnum="19"/>
    <reg name="s4" bitsize="32" regnum="20"/>
    <reg name="s5" bitsize="32" regnum="21"/>
    <reg name="s6" bitsize="32" regnum="22"/>
    <reg name="s7" bitsize="32" regnum="23"/>
    <reg name="s8" bitsize="32" regnum="24"/>
    <reg name="s9" bitsize="32" regnum="25"/>
    <reg name="s10" bitsize="32" regnum="26"/>
    <reg name="s11" bitsize="32" regnum="27"/>
    <reg name="t3" bitsize="32" regnum="28"/>
    <reg name="t4" bitsize="32" regnum="29"/>
    <reg name="t5" bitsize="32" regnum="30"/>
    <reg name="t6" bitsize="32" regnum="31"/>
    <reg name="pc" bitsize="32" type="code_ptr" regnum="32"/>
  </feature>
</target>'''
                try:
                    parts = pkt.split(":")[-1].split(',')
                    offset = int(parts[0], 16)
                    length = int(parts[1], 16)
                    if offset >= len(target_xml):
                        reply = "l"
                    else:
                        chunk = target_xml[offset:offset + length]
                        if offset + length >= len(target_xml):
                            reply = "l" + chunk
                        else:
                            reply = "m" + chunk
                except Exception as e:
                    print(f"[SIM] qXfer parse error: {e}")
                    reply = "E01"

            elif pkt.startswith("qXfer:memory-map:read::"):
                memory_map = self._memory_map_xml()
                try:
                    parts = pkt.split(":")[-1].split(',')
                    offset = int(parts[0], 16)
                    length = int(parts[1], 16)
                    if offset >= len(memory_map):
                        reply = "l"
                    else:
                        chunk = memory_map[offset:offset + length]
                        if offset + length >= len(memory_map):
                            reply = "l" + chunk
                        else:
                            reply = "m" + chunk
                except Exception as e:
                    print(f"[SIM] qXfer memory-map parse error: {e}")
                    reply = "E01"

            elif pkt == "g":
                # Send all registers in little-endian byte order
                reg_bytes = b''.join(struct.pack('<I', r) for r in self.regs) + struct.pack('<I', self.pc)
                reply = binascii.hexlify(reg_bytes).decode()

            elif pkt.startswith("p"):
                try:
                    idx = int(pkt[1:], 16)
                    if idx < 32:
                        val = self.regs[idx]
                        reply = binascii.hexlify(struct.pack('<I', val & 0xffffffff)).decode()
                    elif idx == 32:
                        val = self.pc
                        reply = binascii.hexlify(struct.pack('<I', val & 0xffffffff)).decode()
                    else:
                        reply = "E01"
                except:
                    reply = ""

            elif pkt.startswith("P"):
                try:
                    reg_str, val_hex = pkt[1:].split("=", 1)
                    reg = int(reg_str, 16)
                    val_bytes = bytes.fromhex(val_hex)
                    value = int.from_bytes(val_bytes, "little")
                    if reg == 0:
                        reply = "OK"
                    elif reg < 32:
                        self.regs[reg] = value & 0xffffffff
                        reply = "OK"
                    elif reg == 32:
                        self.pc = value & 0xffffffff
                        reply = "OK"
                    else:
                        reply = "E01"
                except Exception as e:
                    print(f"[SIM] Register write error: {e}")
                    reply = "E01"

            elif pkt.startswith("G"):
                try:
                    data = bytes.fromhex(pkt[1:])
                    reg_count = 33
                    needed = reg_count * 4
                    if len(data) < needed:
                        reply = "E01"
                    else:
                        for i in range(32):
                            self.regs[i] = int.from_bytes(data[i*4:(i+1)*4], "little") & 0xffffffff
                        self.regs[0] = 0
                        self.pc = int.from_bytes(data[32*4:33*4], "little") & 0xffffffff
                        reply = "OK"
                except Exception as e:
                    print(f"[SIM] Register block write error: {e}")
                    reply = "E01"

            elif pkt.startswith("m"):
                try:
                    parts = pkt[1:].split(',')
                    addr = int(parts[0], 16)
                    length = int(parts[1], 16)
                    data = bytes(self.load_byte(addr + i) for i in range(length))
                    reply = binascii.hexlify(data).decode()
                except:
                    reply = "E01"

            elif pkt.startswith("M"):
                try:
                    header, data_hex = pkt[1:].split(":", 1)
                    addr_str, length_str = header.split(",", 1)
                    addr = int(addr_str, 16)
                    length = int(length_str, 16)
                    data = bytes.fromhex(data_hex)
                    if len(data) != length:
                        data = data[:length]
                    self._write_memory(addr, data)
                    reply = "OK"
                except Exception as e:
                    print(f"[SIM] Memory write error: {e}")
                    reply = "E01"

            elif pkt.startswith("X"):
                try:
                    header, data_raw = pkt[1:].split(":", 1)
                    addr_str, length_str = header.split(",", 1)
                    addr = int(addr_str, 16)
                    length = int(length_str, 16)
                    data = data_raw.encode("latin-1")
                    if len(data) != length:
                        data = data[:length]
                    self._write_memory(addr, data)
                    reply = "OK"
                except Exception as e:
                    print(f"[SIM] Binary memory write error: {e}")
                    reply = "E01"

            elif pkt.startswith("qRcmd,"):
                # GDB monitor command - decode hex-encoded command
                try:
                    cmd = bytes.fromhex(pkt[6:]).decode('ascii').strip()
                    if cmd == "reset_counter":
                        self.instr_count = 0
                        response = "Instruction counter reset\n"
                        reply = binascii.hexlify(response.encode()).decode()
                    elif cmd == "show_stats":
                        response = f"Instructions executed: {self.instr_count}\n"
                        reply = binascii.hexlify(response.encode()).decode()
                    elif cmd.startswith("run_steps"):
                        parts = cmd.split()
                        steps = int(parts[1], 0) if len(parts) > 1 else 0
                        executed = 0
                        error = None
                        while executed < steps:
                            try:
                                self.execute()
                            except Exception as e:
                                error = e
                                break
                            executed += 1
                        if error:
                            response = f"Stopped after {executed} steps: {error}\nPC=0x{self.pc:08x}\n"
                        else:
                            response = f"Ran {executed} steps\nPC=0x{self.pc:08x}\n"
                        reply = binascii.hexlify(response.encode()).decode()
                    elif cmd.startswith("run_until_pc"):
                        parts = cmd.split()
                        target = int(parts[1], 0) if len(parts) > 1 else self.pc
                        max_steps = int(parts[2], 0) if len(parts) > 2 else 1000000
                        executed = 0
                        error = None
                        while executed < max_steps and self.pc != target:
                            try:
                                self.execute()
                            except Exception as e:
                                error = e
                                break
                            executed += 1
                        if error:
                            response = (f"Stopped after {executed} steps: {error}\n"
                                        f"PC=0x{self.pc:08x}\n")
                        elif self.pc == target:
                            response = (f"Hit PC=0x{target:08x} after {executed} steps\n"
                                        f"PC=0x{self.pc:08x}\n")
                        else:
                            response = (f"Max steps reached ({executed}) without hitting PC=0x{target:08x}\n"
                                        f"PC=0x{self.pc:08x}\n")
                        reply = binascii.hexlify(response.encode()).decode()
                    elif cmd.startswith("load_elf"):
                        parts = cmd.split(maxsplit=1)
                        if len(parts) < 2:
                            response = "Usage: load_elf <path>\n"
                        else:
                            elf_path = parts[1].strip()
                            try:
                                self.load_elf(elf_path)
                                response = f"Loaded ELF {elf_path}\nPC=0x{self.pc:08x}\n"
                            except Exception as e:
                                response = f"Failed to load ELF: {e}\n"
                        reply = binascii.hexlify(response.encode()).decode()
                    else:
                        reply = ""
                except:
                    reply = ""

            elif pkt == "c":
                stop_reply = None
                while self.gdb_client and self.running_gdb:
                    if self.pc in self.breakpoints:
                        if self._skip_breakpoint_once:
                            self._skip_breakpoint_once = False
                        else:
                            print(f"[SIM] Hit breakpoint at PC=0x{self.pc:08x}")
                            self._skip_breakpoint_once = True
                            stop_reply = "S05"
                            break
                    try:
                        self.execute()
                    except Exception as e:
                        print(f"[SIM] Exception at PC={self.pc:08x}: {e}")
                        stop_reply = self._stop_reply_from_exception(e)
                        break
                if stop_reply is None:
                    stop_reply = "S05"
                self._send_packet(stop_reply)
                continue

            elif pkt.startswith("s"):
                try:
                    self.execute()
                    reply = "S05"
                except Exception as e:
                    reply = self._stop_reply_from_exception(e)

            elif pkt.startswith("Z") or pkt.startswith("z"):
                # Z0 = software breakpoint, z0 = remove software breakpoint
                # Format: Z0,addr,kind or z0,addr,kind
                try:
                    # Find first comma and parse from there
                    comma_idx = pkt.index(',')
                    parts = pkt[comma_idx+1:].split(',')
                    addr = int(parts[0], 16)
                    if pkt[0] == 'Z':
                        self.add_breakpoint(addr)
                    else:
                        self.remove_breakpoint(addr)
                    reply = "OK"
                except Exception as e:
                    print(f"[SIM] Breakpoint parse error: {e}")
                    reply = ""

            # ALWAYS send the packet
            self._send_packet(reply)


if __name__ == "__main__":
    elf_file = None
    gdb_port = 3333
    detect_mode = False
    assisted_mode = False
    stub_config = None

    args = sys.argv[1:]
    while args:
        arg = args.pop(0)
        if arg.startswith("--port="):
            gdb_port = int(arg.split("=")[1])
        elif arg.startswith("--stub="):
            stub_config = arg.split("=")[1]
        elif arg == "--detect":
            detect_mode = True
        elif arg == "--assisted":
            assisted_mode = True
            detect_mode = True  # Assisted mode implies detect mode
        elif arg in ("-h", "--help"):
            print("Usage: python rv32sim.py [program.elf] [OPTIONS]")
            print("Options:")
            print("  --port=PORT      GDB server port (default: 3333)")
            print("  --stub=FILE      Load hardware register stubs from JSON config")
            print("  --detect         Enable hardware access detection mode")
            print("  --assisted       Interactive mode - reports each new hardware access with hints")
            sys.exit(0)
        elif arg.startswith("-"):
            print(f"Unknown option: {arg}")
            sys.exit(1)
        else:
            if elf_file:
                print("Only one ELF file allowed")
                sys.exit(1)
            elf_file = arg

    sim = RV32Sim(gdb_port=gdb_port, detect_mode=detect_mode, assisted_mode=assisted_mode)

    def uart_write(val):
        if 32 <= val < 127 or val in '\n\r\t'.encode():
            print(chr(val), end="", flush=True)
    sim.register_write_handler(0x10000000, uart_write)

    # Load stub configuration if provided
    if stub_config:
        sim.load_stub_config(stub_config)

    if elf_file:
        try:
            sim.load_elf(elf_file)
            sim.regs[2] = (sim.get_stack_top() - 16) & 0xffffffff
        except Exception as e:
            print(f"[ERROR] {e}")
            sys.exit(1)

    sim.start_gdb_server()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[SIM] Shutting down...")
        if detect_mode and sim.hw_accesses:
            # Save detected hardware accesses to a config template
            config_name = f"{elf_file.replace('.elf', '')}_hw_stubs.json" if elf_file else "hw_stubs.json"
            sim.save_hw_accesses(config_name)

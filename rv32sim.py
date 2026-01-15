# rv32sim.py
# Complete RV32I simulator with fully working GDB remote stub
# Fixed all known issues: vMustReplyEmpty, qXfer conflicting responses, malformed packet crashes

import struct
import time
import sys
import os

from assertion_manager import AssertionManager
from cpu_core import CPUCore
from gdb_server import GDBServer
from memory_map import MemoryMap
from mmio import MMIOManager
from rtt_server import RTTServer
from memif import FlatMemIf, load_memif


_CSR_COUNTER_SPECS = {
    0xB00: ("mcycle", 0, "mcycle_suppress"),
    0xC00: ("mcycle", 0, "mcycle_suppress"),
    0xB01: ("mcycle", 0, "mcycle_suppress"),
    0xC01: ("mcycle", 0, "mcycle_suppress"),
    0xB80: ("mcycle", 32, "mcycle_suppress"),
    0xC80: ("mcycle", 32, "mcycle_suppress"),
    0xB81: ("mcycle", 32, "mcycle_suppress"),
    0xC81: ("mcycle", 32, "mcycle_suppress"),
    0xB02: ("minstret", 0, "minstret_suppress"),
    0xC02: ("minstret", 0, "minstret_suppress"),
    0xB82: ("minstret", 32, "minstret_suppress"),
    0xC82: ("minstret", 32, "minstret_suppress"),
}


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
        self.assert_mode = False
        self.assert_assist_mode = False
        self.assert_verbose = False
        self.assert_show_asm = False
        self.assert_writes = False
        self.assertions = {}
        self.assert_dirty = False
        self.assert_save_path = None
        self.svd_index = None
        self._svd_field_cache = {}
        self._input_ready = False
        self.mmio_state = {}
        self.mmio = MMIOManager(self)
        self.assert_manager = AssertionManager(self)
        self.memory_map = MemoryMap(self.memory_size)
        self.cpu = CPUCore(self, TrapException)
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
        self.gdb_server = GDBServer(self)
        self.rtt_server = RTTServer(self)
        self._init_perf_model()

    @property
    def memory_regions(self):
        return self.memory_map.regions

    @memory_regions.setter
    def memory_regions(self, regions):
        self.memory_map.regions = regions

    @property
    def _memory_initialized(self):
        return self.memory_map.initialized

    @_memory_initialized.setter
    def _memory_initialized(self, value):
        self.memory_map.initialized = bool(value)

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
        self.mmio.register_read_handler(addr, func, width=width)

    def register_write_handler(self, addr, func, width=1):
        self.mmio.register_write_handler(addr, func, width=width)

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

    def _is_power_of_two(self, value):
        return value > 0 and (value & (value - 1)) == 0

    def _init_perf_model(self):
        self.timing = {
            "base_cycles": 1,
            "load_penalty": 0,
            "store_penalty": 0,
            "branch_penalty": 1,
            "mispredict_penalty": 1,
        }
        self.bp_cfg = {
            "enabled": True,
            "entries": 8,
            "store_forward": False,
            "alias_word": True,
        }
        self._configure_branch_predictor(self.bp_cfg)
        self.memif_code = FlatMemIf(self._read_memory, self._write_memory, name="code")
        self.memif_data = FlatMemIf(self._read_memory, self._write_memory, name="data")

    def _configure_branch_predictor(self, cfg):
        self.bp_cfg = cfg
        self.bp_enabled = bool(cfg.get("enabled", True))
        self.bp_entries_count = int(cfg.get("entries", 8))
        self.bp_store_forward = bool(cfg.get("store_forward", False))
        self.bp_alias_word = bool(cfg.get("alias_word", True))
        if self.bp_entries_count <= 0:
            self.bp_entries_count = 0
            self.bp_entries = []
        else:
            self._reset_branch_predictor()

    def _reset_branch_predictor(self):
        self.bp_entries = []
        for _ in range(self.bp_entries_count):
            self.bp_entries.append({"valid": False, "pc": 0, "target": 0, "age": 0})
        self.bp_age_counter = 0

    def _bp_lookup(self, pc):
        if not self.bp_enabled:
            return None
        for idx, entry in enumerate(self.bp_entries):
            if entry["valid"] and entry["pc"] == pc:
                return idx
        return None

    def _bp_should_store(self, pc, target):
        if not self.bp_enabled:
            return False
        if not self.bp_store_forward and target > pc:
            return False
        if self.bp_alias_word:
            word_pc = pc >> 2
            for entry in self.bp_entries:
                if entry["valid"] and (entry["pc"] >> 2) == word_pc:
                    return False
        return True

    def _bp_insert(self, pc, target):
        if not self.bp_enabled:
            return
        idx = None
        for i, entry in enumerate(self.bp_entries):
            if not entry["valid"]:
                idx = i
                break
        if idx is None:
            idx = min(range(len(self.bp_entries)), key=lambda i: self.bp_entries[i]["age"])
        self.bp_age_counter += 1
        self.bp_entries[idx] = {
            "valid": True,
            "pc": pc & 0xffffffff,
            "target": target & 0xffffffff,
            "age": self.bp_age_counter,
        }

    def _branch_penalty(self, pc, target, take):
        if not take:
            idx = self._bp_lookup(pc)
            if idx is not None:
                self.bp_entries[idx]["valid"] = False
                return self.timing["mispredict_penalty"]
            return 0

        if not self.bp_enabled:
            return self.timing["branch_penalty"]

        idx = self._bp_lookup(pc)
        if idx is not None:
            entry = self.bp_entries[idx]
            if entry["target"] == (target & 0xffffffff):
                return 0
            entry["valid"] = False
            return self.timing["mispredict_penalty"]

        if self._bp_should_store(pc, target):
            self._bp_insert(pc, target)
        return self.timing["branch_penalty"]

    def _load_memif_from_cfg(self, cfg, name):
        if not cfg:
            return FlatMemIf(self._read_memory, self._write_memory, name=name)
        spec = cfg
        mem_cfg = {}
        if isinstance(cfg, dict):
            spec = cfg.get("module", "")
            mem_cfg = cfg.get("config", {})
        if not spec:
            return FlatMemIf(self._read_memory, self._write_memory, name=name)
        try:
            return load_memif(spec, self._read_memory, self._write_memory, config=mem_cfg, name=name)
        except Exception as exc:
            print(f"[SIM] Failed to load memif {name}: {exc}")
            return FlatMemIf(self._read_memory, self._write_memory, name=name)

    def _configure_memifs(self, cfg):
        if not cfg:
            return
        if isinstance(cfg, dict):
            code_cfg = cfg.get("code")
            data_cfg = cfg.get("data")
        else:
            code_cfg = cfg
            data_cfg = cfg
        self.memif_code = self._load_memif_from_cfg(code_cfg, "code")
        self.memif_data = self._load_memif_from_cfg(data_cfg, "data")

    def _reset_perf_state(self):
        if self.memif_code and hasattr(self.memif_code, "reset"):
            self.memif_code.reset()
        if self.memif_data and hasattr(self.memif_data, "reset"):
            self.memif_data.reset()
        if self.bp_enabled:
            self._reset_branch_predictor()

    def _memif_access(self, memif, addr, size, is_write=False):
        if not memif:
            return 0
        addr &= 0xffffffff
        if self.is_hardware_region(addr):
            return 0
        try:
            return int(memif.access(addr, size, is_write))
        except Exception as exc:
            name = getattr(memif, "name", "memif")
            print(f"[SIM] MemIF {name} access error: {exc}")
            return 0

    def _format_memif_stats(self, label, memif):
        if not memif or not hasattr(memif, "stats"):
            return ""
        stats = memif.stats() or {}
        if not stats:
            return ""
        accesses = int(stats.get("accesses", 0))
        text = f"MemIF {label} accesses: {accesses}\n"
        if "hits" in stats or "misses" in stats:
            hits = int(stats.get("hits", 0))
            misses = int(stats.get("misses", 0))
            total = hits + misses
            hit_rate = (hits / total) if total else 0.0
            text += (
                f"MemIF {label} hits: {hits}\n"
                f"MemIF {label} misses: {misses}\n"
                f"MemIF {label} hit rate: {hit_rate:.4f}\n"
            )
        return text

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

    def _read_counter_csr(self, csr):
        spec = _CSR_COUNTER_SPECS.get(csr)
        if spec is None:
            return None
        name, shift, _suppress = spec
        return (getattr(self, name) >> shift) & 0xffffffff

    def _write_counter_csr(self, csr, value):
        spec = _CSR_COUNTER_SPECS.get(csr)
        if spec is None:
            return False
        name, shift, suppress = spec
        cur = getattr(self, name)
        if shift == 0:
            cur = (cur & 0xffffffff00000000) | value
        else:
            cur = (cur & 0xffffffff) | (value << 32)
        setattr(self, name, cur)
        setattr(self, suppress, True)
        return True

    def _csr_read(self, csr):
        if csr in self.readonly_zero_csrs:
            return 0
        counter = self._read_counter_csr(csr)
        if counter is not None:
            return counter
        if csr == 0x3A0:
            return self.pmpcfg0 & 0xffffffff
        if csr == 0x3B0:
            return self._pmpaddr0_read()
        return self.csrs.get(csr, 0) & 0xffffffff

    def _csr_write(self, csr, value):
        value &= 0xffffffff
        if csr in self.readonly_zero_csrs:
            return
        if self._write_counter_csr(csr, value):
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
        self.memory_map.init_default_regions()

    def _ensure_memory_regions(self):
        self.memory_map.ensure_regions()

    def _add_memory_region(self, start, end, name="mem"):
        return self.memory_map.add_region(start, end, name)

    def _find_region(self, addr):
        return self.memory_map.find_region(addr)

    def _read_memory(self, addr, size):
        return self.memory_map.read_memory(addr, size)

    def _read_memory_bytes(self, addr, size):
        return self.memory_map.read_bytes(addr, size)

    def _write_memory(self, addr, data):
        self.memory_map.write_memory(addr, data)

    def _expand_region(self, region, new_end):
        self.memory_map.expand_region(region, new_end)

    def get_stack_top(self):
        return self.memory_map.get_stack_top()

    def _find_stub_entry(self, addr):
        return self.mmio._find_stub_entry(addr)

    def _stub_read_value(self, stub):
        return self.mmio._stub_read_value(stub)

    def _find_handler_entry(self, addr, handlers):
        return self.mmio._find_handler_entry(addr, handlers)

    def _call_read_handler(self, handler, addr, size):
        return self.mmio._call_read_handler(handler, addr, size)

    def _call_write_handler(self, handler, addr, value, size):
        self.mmio._call_write_handler(handler, addr, value, size)

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

            def parse_int(value, default):
                if value is None:
                    return default
                if isinstance(value, bool):
                    return int(value)
                if isinstance(value, str):
                    return int(value, 0)
                return int(value)

            def parse_bool(value, default=False):
                if value is None:
                    return default
                if isinstance(value, bool):
                    return value
                if isinstance(value, (int, float)):
                    return value != 0
                if isinstance(value, str):
                    text = value.strip().lower()
                    if text in ("1", "true", "yes", "on"):
                        return True
                    if text in ("0", "false", "no", "off"):
                        return False
                return bool(value)

            timing_cfg = config.get("timing")
            if timing_cfg is not None:
                updated = dict(self.timing)
                for key in updated:
                    if key in timing_cfg:
                        updated[key] = parse_int(timing_cfg.get(key), updated[key])
                self.timing = updated

            bp_cfg = config.get("branch_predictor")
            if bp_cfg is not None:
                merged = dict(self.bp_cfg)
                if "enabled" in bp_cfg:
                    merged["enabled"] = parse_bool(bp_cfg.get("enabled"), merged.get("enabled", True))
                if "entries" in bp_cfg:
                    merged["entries"] = parse_int(bp_cfg.get("entries"), merged.get("entries", 0))
                if "store_forward" in bp_cfg:
                    merged["store_forward"] = parse_bool(
                        bp_cfg.get("store_forward"), merged.get("store_forward", False)
                    )
                if "alias_word" in bp_cfg:
                    merged["alias_word"] = parse_bool(
                        bp_cfg.get("alias_word"), merged.get("alias_word", True)
                    )
                self._configure_branch_predictor(merged)

            memif_cfg = config.get("memif")
            if memif_cfg is not None:
                self._configure_memifs(memif_cfg)

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

            self._reset_perf_state()
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

    def load_svd(self, filename):
        self.assert_manager.load_svd(filename)

    def load_assert_config(self, filename):
        self.assert_manager.load_assert_config(filename)

    def save_assert_config(self, filename):
        self.assert_manager.save_assert_config(filename)

    def _assert_active(self):
        return self.assert_manager._assert_active()

    def _assert_read(self, addr, size):
        return self.assert_manager._assert_read(addr, size)

    def _assert_write(self, addr, size, value):
        return self.assert_manager._assert_write(addr, size, value)

    def _mmio_state_read(self, addr, size):
        return self.assert_manager._mmio_state_read(addr, size)

    def _mmio_state_write(self, addr, size, value, reg=None):
        return self.assert_manager._mmio_state_write(addr, size, value, reg=reg)

    def _parse_elf_header(self, elf):
        if len(elf) < 52 or elf[:4] != b'\x7fELF':
            raise ValueError("Invalid ELF file")
        if elf[4] != 1 or elf[5] != 1:
            raise ValueError("Unsupported ELF class or endianness (expected ELF32 little-endian)")

        _, e_machine, _, e_entry, e_phoff = struct.unpack('<16xHHIII', elf[:32])
        _, e_phentsize, e_phnum = struct.unpack('<HHH', elf[40:46])
        if e_phentsize < 32:
            raise ValueError("Invalid program header entry size")
        if e_phoff + e_phentsize * e_phnum > len(elf):
            raise ValueError("Program headers extend beyond EOF")
        if e_machine != 243:  # EM_RISCV
            raise ValueError("Not a RISC-V ELF")
        return e_entry, e_phoff, e_phentsize, e_phnum

    def _iter_elf_program_headers(self, elf, e_phoff, e_phentsize, e_phnum):
        for i in range(e_phnum):
            off = e_phoff + i * e_phentsize
            if off + e_phentsize > len(elf):
                raise ValueError("Program header extends beyond EOF")
            yield struct.unpack('<IIIIIIII', elf[off:off + 32])

    def _load_elf_segment(self, elf, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz):
        if p_memsz < p_filesz:
            raise ValueError("Invalid segment sizes")
        if p_offset + p_filesz > len(elf):
            raise ValueError("Segment extends beyond EOF")
        data = elf[p_offset:p_offset + p_filesz]
        self._write_memory(p_vaddr, data)
        if p_memsz > p_filesz:
            self._write_memory(p_vaddr + p_filesz, b'\x00' * (p_memsz - p_filesz))
        if p_paddr != p_vaddr:
            self._write_memory(p_paddr, data)

    def load_elf(self, filename):
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"ELF file not found: {filename}")

        with open(filename, 'rb') as f:
            elf = f.read()

        e_entry, e_phoff, e_phentsize, e_phnum = self._parse_elf_header(elf)
        self.pc = e_entry & 0xffffffff

        for ph in self._iter_elf_program_headers(elf, e_phoff, e_phentsize, e_phnum):
            p_type, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz = ph[:6]
            if p_type == 1:  # PT_LOAD
                self._load_elf_segment(elf, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz)

        self.elf_loaded = True
        self.elf_path = os.path.abspath(filename)
        print(f"[SIM] Loaded ELF: {self.elf_path}")
        print(f"[SIM] Entry point: 0x{self.pc:08x}")

    def sign_extend(self, val, bits):
        return self.cpu.sign_extend(val, bits)

    def _div_trunc(self, a, b):
        return self.cpu._div_trunc(a, b)

    def _rem_trunc(self, a, b):
        return self.cpu._rem_trunc(a, b)

    def _read_value(self, addr, size):
        return self.mmio.read_value(addr, size)

    def _write_value(self, addr, size, value):
        self.mmio.write_value(addr, size, value)

    def load_byte(self, addr):
        return self.cpu.load_byte(addr)

    def load_half(self, addr):
        return self.cpu.load_half(addr)

    def load_word(self, addr):
        return self.cpu.load_word(addr)

    def store_byte(self, addr, val):
        self.cpu.store_byte(addr, val)

    def store_half(self, addr, val):
        self.cpu.store_half(addr, val)

    def store_word(self, addr, val):
        self.cpu.store_word(addr, val)

    def _load_memif(self, addr, size):
        return self.cpu._load_memif(addr, size)

    def _store_memif(self, addr, size, value):
        return self.cpu._store_memif(addr, size, value)

    def fetch(self):
        return self.cpu.fetch()

    def expand_compressed(self, c_instr):
        return self.cpu.expand_compressed(c_instr)

    def execute(self):
        return self.cpu.execute()

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
        self.gdb_server.start(self.gdb_port)

    def stop_gdb_server(self):
        self.gdb_server.stop()

    def start_rtt_server(self, port=4444):
        self.rtt_server.start_data(port)

    def stop_rtt_server(self):
        self.rtt_server.stop_data()

    def start_rtt_cmd_server(self, port=4444):
        self.rtt_server.start_cmd(port)

    def stop_rtt_cmd_server(self):
        self.rtt_server.stop_cmd()


if __name__ == "__main__":
    elf_file = None
    gdb_port = 3333
    detect_mode = False
    assisted_mode = False
    stub_config = None
    assert_mode = False
    assert_assist_mode = False
    assert_config = None
    assert_output = None
    assert_verbose = False
    assert_show_asm = False
    assert_writes = False
    svd_file = None
    rtt_port = None
    rtt_cmd_port = None
    mem_regions = []

    args = sys.argv[1:]
    def _next_value(flag):
        if not args:
            print(f"Missing value for {flag}")
            sys.exit(1)
        return args.pop(0)

    def _parse_mem_region(spec):
        parts = spec.split(":")
        if len(parts) < 2:
            raise ValueError("mem region must be START:SIZE[:NAME]")
        start = int(parts[0], 0)
        size = int(parts[1], 0)
        if size <= 0:
            raise ValueError("mem region size must be > 0")
        name = ":".join(parts[2:]) if len(parts) > 2 else "mem"
        return start, start + size, name

    while args:
        arg = args.pop(0)
        if arg.startswith("--port="):
            gdb_port = int(arg.split("=")[1])
        elif arg == "--port":
            gdb_port = int(_next_value("--port"))
        elif arg.startswith("--stub="):
            stub_config = arg.split("=")[1]
        elif arg == "--stub":
            stub_config = _next_value("--stub")
        elif arg.startswith("--svd="):
            svd_file = arg.split("=", 1)[1]
        elif arg == "--svd":
            svd_file = _next_value("--svd")
        elif arg.startswith("--rtt-port="):
            rtt_port = int(arg.split("=", 1)[1], 0)
        elif arg == "--rtt-port":
            rtt_port = int(_next_value("--rtt-port"), 0)
        elif arg.startswith("--rtt-cmd-port="):
            rtt_cmd_port = int(arg.split("=", 1)[1], 0)
        elif arg == "--rtt-cmd-port":
            rtt_cmd_port = int(_next_value("--rtt-cmd-port"), 0)
        elif arg == "--rtt":
            rtt_port = 4444
        elif arg == "--rtt-openocd":
            rtt_cmd_port = 4444
            rtt_port = 4001
        elif arg.startswith("--mem-region="):
            mem_regions.append(arg.split("=", 1)[1])
        elif arg == "--mem-region":
            mem_regions.append(_next_value("--mem-region"))
        elif arg.startswith("--assert="):
            assert_config = arg.split("=", 1)[1]
            assert_mode = True
        elif arg == "--assert":
            assert_config = _next_value("--assert")
            assert_mode = True
        elif arg.startswith("--assert-assist"):
            assert_assist_mode = True
            assert_mode = True
            if arg.startswith("--assert-assist="):
                assert_output = arg.split("=", 1)[1]
        elif arg.startswith("--assert-verbose"):
            assert_mode = True
            if arg.startswith("--assert-verbose="):
                value = arg.split("=", 1)[1]
                assert_verbose = value.strip().lower() not in ("0", "false", "no", "off")
            else:
                assert_verbose = True
        elif arg == "--assert-asm":
            assert_mode = True
            assert_show_asm = True
        elif arg == "--assert-writes":
            assert_mode = True
            assert_writes = True
        elif arg.startswith("--assert-out="):
            assert_output = arg.split("=", 1)[1]
        elif arg == "--assert-out":
            assert_output = _next_value("--assert-out")
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
            print("  --svd=FILE       Load CMSIS-SVD file for register hints")
            print("  --assert=FILE    Load assertion JSON (strict MMIO assertions)")
            print("  --assert-assist  Interactive assertion assistant (prompts on MMIO)")
            print("  --assert-verbose Always show full field detail/enums in prompts")
            print("  --assert-asm     Show disassembly around MMIO access")
            print("  --assert-writes  Prompt/assert on MMIO writes (default: record only)")
            print("  --assert-out=FILE  Output assertion JSON (with --assert-assist)")
            print("  --rtt           Enable RTT server on port 4444")
            print("  --rtt-port=PORT Enable RTT server on PORT")
            print("  --rtt-cmd-port=PORT  Enable RTT command server on PORT")
            print("  --rtt-openocd   Enable RTT command on 4444 and data on 4001")
            print("  --mem-region=START:SIZE[:NAME]  Add a RAM region to the memory map")
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

    if assert_mode and not assert_assist_mode and not assert_config:
        print("[ERROR] --assert requires a FILE")
        sys.exit(1)

    sim = RV32Sim(gdb_port=gdb_port, detect_mode=detect_mode, assisted_mode=assisted_mode)

    def uart_write(val):
        if 32 <= val < 127 or val in '\n\r\t'.encode():
            print(chr(val), end="", flush=True)
    sim.register_write_handler(0x10000000, uart_write)

    if svd_file:
        sim.load_svd(svd_file)

    if assert_mode:
        sim.assert_mode = True
    if assert_verbose:
        sim.assert_verbose = True
        sim.assert_show_asm = True
    if assert_show_asm:
        sim.assert_show_asm = True
    if assert_writes:
        sim.assert_writes = True
    if assert_assist_mode:
        sim.assert_assist_mode = True
        sim.assert_mode = True
        if assert_output is None:
            assert_output = assert_config
        if assert_output is None:
            assert_output = f"{elf_file.replace('.elf', '')}_assertions.json" if elf_file else "assertions.json"
        sim.assert_save_path = assert_output

    if assert_config:
        sim.load_assert_config(assert_config)

    # Load stub configuration if provided
    if stub_config:
        sim.load_stub_config(stub_config)

    if mem_regions:
        if not sim.memory_regions and not sim._memory_initialized:
            sim._init_default_memory_regions()
        try:
            for spec in mem_regions:
                start, end, name = _parse_mem_region(spec)
                sim._add_memory_region(start, end, name)
                print(f"[SIM] Memory region: {name} 0x{start:08x}-0x{end:08x}")
        except Exception as e:
            print(f"[ERROR] {e}")
            sys.exit(1)

    if elf_file:
        try:
            sim.load_elf(elf_file)
            sim.regs[2] = (sim.get_stack_top() - 16) & 0xffffffff
        except Exception as e:
            print(f"[ERROR] {e}")
            sys.exit(1)

    sim.start_gdb_server()
    if rtt_port is not None:
        sim.start_rtt_server(rtt_port)
    if rtt_cmd_port is not None:
        sim.start_rtt_cmd_server(rtt_cmd_port)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[SIM] Shutting down...")
        sim.stop_gdb_server()
        sim.stop_rtt_server()
        sim.stop_rtt_cmd_server()
        if sim.assert_assist_mode and sim.assertions and sim.assert_dirty:
            out_path = sim.assert_save_path or (
                f"{elf_file.replace('.elf', '')}_assertions.json" if elf_file else "assertions.json"
            )
            sim.save_assert_config(out_path)
        if detect_mode and sim.hw_accesses:
            # Save detected hardware accesses to a config template
            config_name = f"{elf_file.replace('.elf', '')}_hw_stubs.json" if elf_file else "hw_stubs.json"
            sim.save_hw_accesses(config_name)

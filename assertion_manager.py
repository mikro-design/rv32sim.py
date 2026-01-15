from svd_parser import SvdIndex


class AssertionManager:
    def __init__(self, sim):
        object.__setattr__(self, "sim", sim)

    def __getattr__(self, name):
        return getattr(self.sim, name)

    def __setattr__(self, name, value):
        if name == "sim":
            object.__setattr__(self, name, value)
        else:
            setattr(self.sim, name, value)

    def _enable_input(self):
        if self._input_ready:
            return
        try:
            import readline  # noqa: F401
        except Exception:
            pass
        self._input_ready = True

    def _format_hex(self, value, width_bytes=None):
        if width_bytes is None:
            return f"0x{value:x}"
        digits = max(1, int(width_bytes) * 2)
        return f"0x{value:0{digits}x}"

    def _parse_assert_int(self, value, default=None):
        if value is None:
            return default
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value.strip(), 0)
            except ValueError:
                return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _parse_assert_bool(self, value, default=False):
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

    def load_svd(self, filename):
        try:
            self.svd_index = SvdIndex(filename)
            self._svd_field_cache = {}
            print(f"[SIM] Loaded SVD: {filename}")
        except FileNotFoundError:
            print(f"[SIM] SVD not found: {filename}")
        except Exception as e:
            print(f"[SIM] Error loading SVD: {e}")

    def load_assert_config(self, filename):
        import json
        try:
            with open(filename, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            print(f"[SIM] No assertion config found: {filename}")
            return
        except Exception as e:
            print(f"[SIM] Error loading assertion config: {e}")
            return

        entries = config.get("assertions")
        if entries is None and "hw_stubs" in config:
            entries = config.get("hw_stubs")
        if not isinstance(entries, dict):
            print(f"[SIM] Assertion config has no assertions: {filename}")
            return

        loaded = 0
        for addr_str, raw in entries.items():
            addr = self._parse_assert_int(addr_str)
            if addr is None:
                continue
            entry = self._normalize_assert_entry(raw)
            if not entry:
                continue
            self.assertions[addr] = entry
            loaded += 1
        print(f"[SIM] Loaded {loaded} assertion entries from {filename}")

    def save_assert_config(self, filename):
        import json
        out = {"assertions": {}}
        for addr, entry in sorted(self.assertions.items()):
            width = entry.get("width", 4)
            out_entry = {}
            if width:
                out_entry["width"] = width
            for key in ("register", "peripheral", "comment"):
                if key in entry:
                    out_entry[key] = entry[key]
            read_entry = entry.get("read")
            if read_entry:
                out_entry["read"] = self._serialize_assert_read(read_entry, width)
            write_entry = entry.get("write")
            if write_entry:
                out_entry["write"] = self._serialize_assert_write(write_entry, width)
            out["assertions"][self._format_hex(addr, 4)] = out_entry
        with open(filename, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"[SIM] Saved assertions to {filename}")

    def _normalize_assert_entry(self, raw):
        entry = {}
        if isinstance(raw, dict):
            width = self._parse_assert_int(raw.get("width"))
            if width is not None:
                entry["width"] = width
            read = raw.get("read")
            write = raw.get("write")
            if read is None and ("value" in raw or "sequence" in raw):
                read = {
                    "value": raw.get("value"),
                    "sequence": raw.get("sequence"),
                    "repeat": raw.get("repeat"),
                    "hold_last": raw.get("hold_last"),
                }
            entry_read = self._normalize_assert_read(read)
            if entry_read:
                entry["read"] = entry_read
            entry_write = self._normalize_assert_write(write)
            if entry_write:
                entry["write"] = entry_write
            for key in ("register", "peripheral", "comment"):
                if key in raw:
                    entry[key] = str(raw[key])
        else:
            value = self._parse_assert_int(raw)
            if value is None:
                return None
            entry["read"] = {"value": value}
        return entry

    def _normalize_assert_read(self, raw):
        if raw is None:
            return None
        if isinstance(raw, dict):
            if self._parse_assert_bool(raw.get("ignore"), False):
                return {"ignore": True}
            entry = {}
            if "sequence" in raw:
                seq = raw.get("sequence")
                if isinstance(seq, (list, tuple)):
                    entry["sequence"] = [
                        v for v in (self._parse_assert_int(x) for x in seq) if v is not None
                    ]
            if "repeat" in raw:
                entry["repeat"] = self._parse_assert_bool(raw.get("repeat"))
            if "hold_last" in raw:
                entry["hold_last"] = self._parse_assert_bool(raw.get("hold_last"), True)
            if "value" in raw:
                value = self._parse_assert_int(raw.get("value"))
                if value is not None:
                    entry["value"] = value
            return entry if entry else None
        value = self._parse_assert_int(raw)
        return {"value": value} if value is not None else None

    def _normalize_assert_write(self, raw):
        if raw is None:
            return None
        if isinstance(raw, dict):
            if self._parse_assert_bool(raw.get("ignore"), False):
                return {"ignore": True}
            entry = {}
            value = self._parse_assert_int(raw.get("value"))
            if value is not None:
                entry["value"] = value
            mask = self._parse_assert_int(raw.get("mask"))
            if mask is not None:
                entry["mask"] = mask
            return entry if entry else None
        value = self._parse_assert_int(raw)
        return {"value": value} if value is not None else None

    def _serialize_assert_read(self, read_entry, width):
        if read_entry.get("ignore"):
            return {"ignore": True}
        out = {}
        if "sequence" in read_entry:
            out["sequence"] = [self._format_hex(v, width) for v in read_entry.get("sequence", [])]
        if "repeat" in read_entry:
            out["repeat"] = bool(read_entry.get("repeat"))
        if "hold_last" in read_entry:
            out["hold_last"] = bool(read_entry.get("hold_last"))
        if "value" in read_entry:
            out["value"] = self._format_hex(read_entry.get("value", 0), width)
        return out

    def _serialize_assert_write(self, write_entry, width):
        if write_entry.get("ignore"):
            return {"ignore": True}
        out = {}
        if "value" in write_entry:
            out["value"] = self._format_hex(write_entry.get("value", 0), width)
        if "mask" in write_entry:
            out["mask"] = self._format_hex(write_entry.get("mask", 0), width)
        return out

    def _find_assert_entry(self, addr):
        if addr in self.assertions:
            entry = self.assertions[addr]
            width = entry.get("width", 4) if isinstance(entry, dict) else 4
            return addr, entry, width
        base = addr & ~0x3
        if base in self.assertions:
            entry = self.assertions[base]
            width = entry.get("width", 4) if isinstance(entry, dict) else 4
            if base <= addr < base + width:
                return base, entry, width
        base = addr & ~0x1
        if base in self.assertions:
            entry = self.assertions[base]
            width = entry.get("width", 4) if isinstance(entry, dict) else 4
            if base <= addr < base + width:
                return base, entry, width
        return None, None, None

    def _find_mmio_state_entry(self, addr):
        if addr in self.mmio_state:
            entry = self.mmio_state[addr]
            width = entry.get("width", 4)
            return addr, entry, width
        base = addr & ~0x3
        if base in self.mmio_state:
            entry = self.mmio_state[base]
            width = entry.get("width", 4)
            if base <= addr < base + width:
                return base, entry, width
        base = addr & ~0x1
        if base in self.mmio_state:
            entry = self.mmio_state[base]
            width = entry.get("width", 2)
            if base <= addr < base + width:
                return base, entry, width
        return None, None, None

    def _mmio_state_read(self, addr, size):
        base, entry, width = self._find_mmio_state_entry(addr)
        if entry is None or size > width or addr + size > base + width:
            return None, None
        full = self._u32(entry.get("value", 0))
        shift = (addr - base) * 8
        mask = (1 << (size * 8)) - 1
        value = (full >> shift) & mask
        return value, base

    def _mmio_state_write(self, addr, size, value, reg=None):
        if reg is not None:
            base = reg.address
            width = reg.width_bytes
        else:
            base = addr & ~0x3
            width = max(4, size)
        entry = self.mmio_state.get(base)
        if entry is None:
            entry = {"width": width, "value": 0}
            self.mmio_state[base] = entry
        entry_width = entry.get("width", width)
        if entry_width < width:
            entry_width = width
            entry["width"] = entry_width
        cur = self._u32(entry.get("value", 0))
        for i in range(size):
            b = (value >> (8 * i)) & 0xff
            shift = (addr - base + i) * 8
            cur = (cur & ~(0xff << shift)) | (b << shift)
        entry["value"] = cur & ((1 << (entry_width * 8)) - 1)
        return base

    def _assert_active(self):
        return self.assert_mode or self.assert_assist_mode

    def _field_map(self, reg):
        cached = self._svd_field_cache.get(reg.address)
        if cached is not None:
            return cached
        field_map = {field.name.lower(): field for field in reg.fields}
        self._svd_field_cache[reg.address] = field_map
        return field_map

    def _parse_field_value(self, field, text):
        if text is None:
            return None
        value_text = text.strip()
        for name, enum in field.enums.items():
            if name.lower() == value_text.lower():
                return enum.value
        return self._parse_assert_int(value_text)

    def _format_register_hint(self, reg):
        lines = []
        if reg is None:
            return lines
        access = f", {reg.access}" if reg.access else ""
        lines.append(
            f"[ASSERT] Register: {reg.path} (addr 0x{reg.address & 0xffffffff:08x}, "
            f"{reg.size_bits} bits{access})"
        )
        desc = self._clean_description(reg.description)
        if desc:
            lines.append(f"[ASSERT] Info: {desc}")
        if reg.reset_value is not None:
            lines.append(f"[ASSERT] Reset: {self._format_hex(reg.reset_value, reg.width_bytes)}")
        if reg.fields:
            field_items = [f"{f.name}{f.format_range()}" for f in reg.fields[:8]]
            if len(reg.fields) > 8:
                field_items.append("...")
            lines.append(f"[ASSERT] Fields: {', '.join(field_items)}")
        return lines

    def _enum_items(self, field):
        if not field.enums:
            return ""
        enums = sorted(field.enums.values(), key=lambda e: e.value)
        if self.assert_verbose:
            return ", ".join(f"{e.name}={self._format_hex(e.value)}" for e in enums)
        limit = 6
        parts = [f"{e.name}={self._format_hex(e.value)}" for e in enums[:limit]]
        if len(enums) > limit:
            parts.append("...")
        return ", ".join(parts)

    def _field_value_label(self, field, field_val):
        enum_match = None
        for enum in field.enums.values():
            if enum.value == field_val:
                enum_match = enum.name
                break
        value_label = self._format_hex(field_val)
        if enum_match:
            value_label = f"{value_label} ({enum_match})"
        return value_label

    def _clean_description(self, text):
        if not text:
            return ""
        if not isinstance(text, str):
            text = str(text)
        desc = text.strip()
        if not desc:
            return ""
        lowered = desc.lower()
        if lowered in ("description", "n/a", "na", "none", "reserved"):
            return ""
        if "defaultdict(" in lowered or "lambda" in lowered:
            return ""
        desc = desc.splitlines()[0].strip()
        if not self.assert_verbose and len(desc) > 120:
            desc = desc[:117] + "..."
        return desc

    def _print_field_details(self, reg, value=None, active_mask=None, full=None):
        if reg is None or not reg.fields:
            return
        if full is None:
            full = self.assert_verbose
        if not full:
            fields = []
            title = None
            if active_mask is not None:
                fields = [field for field in reg.fields if field.mask & active_mask]
                if fields:
                    title = "[ASSERT] Field details (active only; use ? for full list):"
            if not fields and value is not None:
                fields = [field for field in reg.fields if value & field.mask]
                if fields:
                    title = "[ASSERT] Field details (nonzero only; use ? for full list):"
            if not fields:
                print("[ASSERT] Field details: (use ? for full list)")
                return
            max_name = max(len(field.name) for field in fields)
            print(title)
            for field in fields:
                bits = field.format_range()
                line = f"[ASSERT] * {field.name:<{max_name}} {bits:<7}"
                if value is not None:
                    field_val = (value & field.mask) >> field.lsb
                    line += f" = {self._field_value_label(field, field_val)}"
                desc = self._clean_description(field.description)
                if desc:
                    line += f" - {desc}"
                print(line)
            return

        max_name = max(len(field.name) for field in reg.fields)
        print("[ASSERT] Field details (active marked with *):")
        for field in reg.fields:
            bits = field.format_range()
            is_active = bool(active_mask is not None and (field.mask & active_mask))
            marker = "*" if is_active else " "
            line = f"[ASSERT] {marker} {field.name:<{max_name}} {bits:<7}"
            if value is not None:
                field_val = (value & field.mask) >> field.lsb
                line += f" = {self._field_value_label(field, field_val)}"
            desc = self._clean_description(field.description)
            if desc:
                line += f" - {desc}"
            enums = self._enum_items(field)
            if enums:
                line += f" enums: {enums}"
            print(line)

    def _format_field_assignments(self, reg, value, mask):
        if reg is None or not reg.fields or not mask:
            return ""
        width_mask = (1 << (reg.width_bytes * 8)) - 1
        if mask == width_mask and len(reg.fields) > 4:
            return ""
        fields = [field for field in reg.fields if field.mask & mask]
        if not fields or len(fields) > 6:
            return ""
        parts = []
        for field in fields:
            field_val = (value & field.mask) >> field.lsb
            enum_match = None
            for enum in field.enums.values():
                if enum.value == field_val:
                    enum_match = enum.name
                    break
            if enum_match:
                parts.append(f"{field.name}={enum_match}")
            else:
                parts.append(f"{field.name}={self._format_hex(field_val)}")
        return " " + ", ".join(parts)

    def _peek_instruction(self, addr):
        try:
            first_half = self._read_memory(addr, 2)
        except Exception:
            return None, None
        if (first_half & 0x3) != 0x3:
            try:
                instr = self._u32(self.expand_compressed(first_half))
            except Exception:
                return None, None
            return instr, 2
        try:
            second_half = self._read_memory(addr + 2, 2)
        except Exception:
            return None, None
        instr = first_half | (second_half << 16)
        return instr, 4

    def _format_imm(self, imm):
        if imm < 0:
            return f"-0x{(-imm):x}"
        return f"0x{imm:x}"

    def _format_reg(self, idx):
        return f"x{idx}"

    def _disasm_instruction(self, pc, instr):
        opcode = instr & 0x7f
        rd = (instr >> 7) & 0x1f
        funct3 = (instr >> 12) & 0x7
        rs1 = (instr >> 15) & 0x1f
        rs2 = (instr >> 20) & 0x1f
        funct7 = (instr >> 25) & 0x7f

        if opcode == 0b0110111:  # LUI
            imm = instr & 0xfffff000
            return f"lui {self._format_reg(rd)}, 0x{imm:08x}"
        if opcode == 0b0010111:  # AUIPC
            imm = instr & 0xfffff000
            return f"auipc {self._format_reg(rd)}, 0x{imm:08x}"
        if opcode == 0b1101111:  # JAL
            imm = ((instr >> 31) & 0x1) << 20
            imm |= ((instr >> 21) & 0x3ff) << 1
            imm |= ((instr >> 20) & 0x1) << 11
            imm |= ((instr >> 12) & 0xff) << 12
            imm = self.sign_extend(imm, 21)
            target = (pc + imm) & 0xffffffff
            return f"jal {self._format_reg(rd)}, 0x{target:08x}"
        if opcode == 0b1100111 and funct3 == 0x0:  # JALR
            imm = self.sign_extend(instr >> 20, 12)
            return f"jalr {self._format_reg(rd)}, {self._format_imm(imm)}({self._format_reg(rs1)})"
        if opcode == 0b1100011:  # BRANCH
            imm = ((instr >> 31) & 0x1) << 12
            imm |= ((instr >> 25) & 0x3f) << 5
            imm |= ((instr >> 8) & 0xf) << 1
            imm |= ((instr >> 7) & 0x1) << 11
            imm = self.sign_extend(imm, 13)
            target = (pc + imm) & 0xffffffff
            mnem = {
                0x0: "beq",
                0x1: "bne",
                0x4: "blt",
                0x5: "bge",
                0x6: "bltu",
                0x7: "bgeu",
            }.get(funct3, "b??")
            return f"{mnem} {self._format_reg(rs1)}, {self._format_reg(rs2)}, 0x{target:08x}"
        if opcode == 0b0000011:  # LOAD
            imm = self.sign_extend(instr >> 20, 12)
            mnem = {
                0x0: "lb",
                0x1: "lh",
                0x2: "lw",
                0x4: "lbu",
                0x5: "lhu",
            }.get(funct3, "l??")
            return f"{mnem} {self._format_reg(rd)}, {self._format_imm(imm)}({self._format_reg(rs1)})"
        if opcode == 0b0100011:  # STORE
            imm = ((instr >> 25) << 5) | ((instr >> 7) & 0x1f)
            imm = self.sign_extend(imm, 12)
            mnem = {
                0x0: "sb",
                0x1: "sh",
                0x2: "sw",
            }.get(funct3, "s??")
            return f"{mnem} {self._format_reg(rs2)}, {self._format_imm(imm)}({self._format_reg(rs1)})"
        if opcode == 0b0010011:  # OP-IMM
            imm = self.sign_extend(instr >> 20, 12)
            if funct3 == 0x0:
                return f"addi {self._format_reg(rd)}, {self._format_reg(rs1)}, {self._format_imm(imm)}"
            if funct3 == 0x2:
                return f"slti {self._format_reg(rd)}, {self._format_reg(rs1)}, {self._format_imm(imm)}"
            if funct3 == 0x3:
                return f"sltiu {self._format_reg(rd)}, {self._format_reg(rs1)}, {self._format_imm(imm)}"
            if funct3 == 0x4:
                return f"xori {self._format_reg(rd)}, {self._format_reg(rs1)}, {self._format_imm(imm)}"
            if funct3 == 0x6:
                return f"ori {self._format_reg(rd)}, {self._format_reg(rs1)}, {self._format_imm(imm)}"
            if funct3 == 0x7:
                return f"andi {self._format_reg(rd)}, {self._format_reg(rs1)}, {self._format_imm(imm)}"
            if funct3 == 0x1:
                shamt = (instr >> 20) & 0x1f
                return f"slli {self._format_reg(rd)}, {self._format_reg(rs1)}, {shamt}"
            if funct3 == 0x5:
                shamt = (instr >> 20) & 0x1f
                if funct7 == 0x20:
                    return f"srai {self._format_reg(rd)}, {self._format_reg(rs1)}, {shamt}"
                return f"srli {self._format_reg(rd)}, {self._format_reg(rs1)}, {shamt}"
        if opcode == 0b0110011:  # OP
            if funct7 == 0x01:
                mnem = {
                    0x0: "mul",
                    0x1: "mulh",
                    0x2: "mulhsu",
                    0x3: "mulhu",
                    0x4: "div",
                    0x5: "divu",
                    0x6: "rem",
                    0x7: "remu",
                }.get(funct3, "mul?")
                return f"{mnem} {self._format_reg(rd)}, {self._format_reg(rs1)}, {self._format_reg(rs2)}"
            if funct3 == 0x0:
                return f"{'sub' if funct7 == 0x20 else 'add'} {self._format_reg(rd)}, {self._format_reg(rs1)}, {self._format_reg(rs2)}"
            if funct3 == 0x1:
                return f"sll {self._format_reg(rd)}, {self._format_reg(rs1)}, {self._format_reg(rs2)}"
            if funct3 == 0x2:
                return f"slt {self._format_reg(rd)}, {self._format_reg(rs1)}, {self._format_reg(rs2)}"
            if funct3 == 0x3:
                return f"sltu {self._format_reg(rd)}, {self._format_reg(rs1)}, {self._format_reg(rs2)}"
            if funct3 == 0x4:
                return f"xor {self._format_reg(rd)}, {self._format_reg(rs1)}, {self._format_reg(rs2)}"
            if funct3 == 0x5:
                return f"{'sra' if funct7 == 0x20 else 'srl'} {self._format_reg(rd)}, {self._format_reg(rs1)}, {self._format_reg(rs2)}"
            if funct3 == 0x6:
                return f"or {self._format_reg(rd)}, {self._format_reg(rs1)}, {self._format_reg(rs2)}"
            if funct3 == 0x7:
                return f"and {self._format_reg(rd)}, {self._format_reg(rs1)}, {self._format_reg(rs2)}"
        if opcode == 0b1110011:
            if instr == 0x00000073:
                return "ecall"
            if instr == 0x00100073:
                return "ebreak"
            csr = (instr >> 20) & 0xfff
            mnem = {
                0x1: "csrrw",
                0x2: "csrrs",
                0x3: "csrrc",
                0x5: "csrrwi",
                0x6: "csrrsi",
                0x7: "csrrci",
            }.get(funct3, "csr?")
            if funct3 in (0x5, 0x6, 0x7):
                zimm = rs1
                return f"{mnem} {self._format_reg(rd)}, 0x{csr:03x}, {zimm}"
            return f"{mnem} {self._format_reg(rd)}, 0x{csr:03x}, {self._format_reg(rs1)}"
        return f"unknown 0x{instr:08x}"

    def _format_asm_window(self, pc, count=8):
        lines = []
        addr = pc & 0xffffffff
        for i in range(count):
            try:
                half = self._read_memory(addr, 2)
            except Exception:
                break
            if (half & 0x3) != 0x3:
                raw = half & 0xffff
                try:
                    instr = self._u32(self.expand_compressed(raw))
                    asm = self._disasm_instruction(addr, instr)
                except Exception:
                    asm = f"c.? 0x{raw:04x}"
                size = 2
                raw_text = f"{raw:04x}"
            else:
                try:
                    raw = self._read_memory(addr, 4)
                except Exception:
                    break
                instr = raw & 0xffffffff
                asm = self._disasm_instruction(addr, instr)
                size = 4
                raw_text = f"{raw:08x}"
            prefix = ">>" if i == 0 else "  "
            lines.append(f"[ASSERT] {prefix} 0x{addr:08x}: {raw_text:<8} {asm}")
            addr = (addr + size) & 0xffffffff
        return lines

    def _disasm_at(self, pc):
        instr, _ = self._peek_instruction(pc)
        if instr is None:
            return None
        try:
            return self._disasm_instruction(pc, instr)
        except Exception:
            return None

    def _pick_nonzero_value(self, mask, width_mask):
        if mask:
            bit = mask & -mask
            if bit:
                return bit & width_mask
        return 1 & width_mask

    def _print_decision_table(self, reg, decision_groups, width_bytes):
        if not decision_groups:
            return
        for group in decision_groups:
            header = group.get("header")
            if header:
                print(f"[ASSERT] Decision ({header}):")
            else:
                print("[ASSERT] Decision:")
            rows = []
            for row in group.get("rows", []):
                value_text = self._format_hex(row["value"], width_bytes)
                assign = self._format_field_assignments(reg, row["value"], row.get("mask"))
                left = value_text + assign
                asm = self._disasm_at(row["pc"])
                right = f"0x{row['pc']:08x}"
                if asm:
                    right += f": {asm}"
                note = row.get("note")
                path = row.get("path")
                rows.append((left, right, note, path))
            if not rows:
                continue
            left_width = max(len(left) for left, _, _, _ in rows)
            for left, right, note, path in rows:
                line = f"[ASSERT]   {left:<{left_width}} -> {right}"
                meta = []
                if path:
                    meta.append(path)
                if note:
                    meta.append(note)
                if meta:
                    line += f" ({', '.join(meta)})"
                print(line)

    def _mask_to_fields(self, reg, mask):
        if reg is None or not reg.fields:
            return []
        hits = []
        for field in reg.fields:
            if field.mask & mask:
                hits.append(field.name)
        return hits

    def _decision_hints_for_read(self, reg, size):
        hints = []
        active_mask = None
        full_value_branch = False
        decision_groups = []
        branch_used = False
        loop_hints = []
        loop_on_value = False
        loop_compare = None
        instr = self.last_instr
        if instr is None:
            return hints, active_mask, decision_groups
        if (instr & 0x3) != 0x3:
            try:
                instr = self._u32(self.expand_compressed(instr))
                instr_size = 2
            except Exception:
                return hints, active_mask
        else:
            instr_size = 4
        opcode = instr & 0x7f
        if opcode != 0b0000011:  # not a load
            return hints, active_mask, decision_groups
        rd = (instr >> 7) & 0x1f
        pc = (self.pc + instr_size) & 0xffffffff
        width = max(size, reg.width_bytes if reg else size, 1)
        width_mask = (1 << (width * 8)) - 1
        states = {
            rd: {
                "origin_mask": width_mask,
                "shift": 0,
                "width": width,
                "valid": True,
                "touched": False,
            }
        }

        def map_mask_to_original(state, mask_current):
            shift = state["shift"]
            if shift >= 0:
                mapped = (mask_current << shift) & width_mask
            else:
                mapped = (mask_current >> (-shift)) & width_mask
            return mapped

        def record_active(mask):
            nonlocal active_mask
            if active_mask is None:
                active_mask = mask & width_mask
            else:
                active_mask |= mask & width_mask

        hints.append(f"Loaded into x{rd}")
        consts = {}
        for _ in range(8):
            nxt, sz = self._peek_instruction(pc)
            if nxt is None or sz is None:
                break
            cur_pc = pc
            opcode = nxt & 0x7f
            rd2 = (nxt >> 7) & 0x1f
            funct3 = (nxt >> 12) & 0x7
            rs1 = (nxt >> 15) & 0x1f
            rs2 = (nxt >> 20) & 0x1f
            state = states.get(rs1)
            if opcode == 0b0010011 and funct3 == 0x7 and state is not None:
                imm = self.sign_extend(nxt >> 20, 12) & width_mask
                mapped = map_mask_to_original(state, imm)
                state = dict(state)
                state["origin_mask"] &= mapped
                state["touched"] = True
                states[rd2] = state
                record_active(mapped)
                fields = self._mask_to_fields(reg, mapped)
                fields_text = f" fields: {', '.join(fields)}" if fields else ""
                hints.append(f"Mask 0x{mapped:0{width*2}x} via ANDI{fields_text}")
            elif opcode == 0b0010011 and funct3 == 0x1 and state is not None:
                shamt = (nxt >> 20) & 0x1f
                state = dict(state)
                state["shift"] -= shamt
                state["touched"] = True
                if shamt >= width:
                    state["origin_mask"] = 0
                else:
                    state["origin_mask"] &= (1 << (width - shamt)) - 1
                states[rd2] = state
                hints.append(f"Shift left by {shamt}")
            elif opcode == 0b0010011 and funct3 == 0x5 and state is not None:
                shamt = (nxt >> 20) & 0x1f
                state = dict(state)
                state["shift"] += shamt
                state["touched"] = True
                if shamt >= width:
                    state["origin_mask"] = 0
                else:
                    state["origin_mask"] &= (~((1 << shamt) - 1)) & width_mask
                states[rd2] = state
                hints.append(f"Shift right by {shamt}")
            elif opcode == 0b0010011 and funct3 in (0x2, 0x3) and state is not None:
                imm = self.sign_extend(nxt >> 20, 12)
                hints.append(f"Compare against {imm} via SLTI")
                if state["valid"]:
                    record_active(state["origin_mask"])
            elif opcode == 0b1100011 and (rs1 in states or rs2 in states):
                tracked_reg = rs1 if rs1 in states else rs2
                state = states.get(tracked_reg)
                other = rs2 if tracked_reg == rs1 else rs1
                imm = ((nxt >> 31) & 0x1) << 12
                imm |= ((nxt >> 25) & 0x3f) << 5
                imm |= ((nxt >> 8) & 0xf) << 1
                imm |= ((nxt >> 7) & 0x1) << 11
                imm = self.sign_extend(imm, 13)
                branch_target = (cur_pc + imm) & 0xffffffff
                fallthrough = (cur_pc + sz) & 0xffffffff
                cond = {
                    0x0: "BEQ",
                    0x1: "BNE",
                    0x4: "BLT",
                    0x5: "BGE",
                    0x6: "BLTU",
                    0x7: "BGEU",
                }.get(funct3, "BR")
                branch_used = True
                if branch_target < cur_pc:
                    loop_hints.append(
                        f"Loop ahead: {cond} x{rs1}, x{rs2} branches back to 0x{branch_target:08x}"
                    )
                    loop_on_value = True
                if other == 0:
                    if cond == "BEQ":
                        detail = f"zero -> 0x{branch_target:08x}, nonzero -> 0x{fallthrough:08x}"
                    elif cond == "BNE":
                        detail = f"nonzero -> 0x{branch_target:08x}, zero -> 0x{fallthrough:08x}"
                    elif cond in ("BLT", "BLTU"):
                        detail = f"negative -> 0x{branch_target:08x}, nonnegative -> 0x{fallthrough:08x}"
                    elif cond in ("BGE", "BGEU"):
                        detail = f"nonnegative -> 0x{branch_target:08x}, negative -> 0x{fallthrough:08x}"
                    else:
                        detail = f"true -> 0x{branch_target:08x}, false -> 0x{fallthrough:08x}"
                    hints.append(f"{cond} x{tracked_reg}, x0: {detail}")
                    if cond in ("BEQ", "BNE") and state is not None and state.get("valid"):
                        decision_mask = state["origin_mask"] & width_mask
                        always_zero = bool(decision_mask == 0)
                        if always_zero:
                            decision_mask = None
                        nonzero_val = self._pick_nonzero_value(decision_mask, width_mask)
                        zero_val = 0
                        if cond == "BEQ":
                            rows = [
                                {
                                    "value": zero_val,
                                    "pc": branch_target,
                                    "note": "zero",
                                    "path": "taken",
                                    "mask": decision_mask,
                                },
                                {
                                    "value": nonzero_val,
                                    "pc": fallthrough,
                                    "note": "nonzero",
                                    "path": "fallthrough",
                                    "mask": decision_mask,
                                },
                            ]
                        else:
                            rows = [
                                {
                                    "value": nonzero_val,
                                    "pc": branch_target,
                                    "note": "nonzero",
                                    "path": "taken",
                                    "mask": decision_mask,
                                },
                                {
                                    "value": zero_val,
                                    "pc": fallthrough,
                                    "note": "zero",
                                    "path": "fallthrough",
                                    "mask": decision_mask,
                                },
                            ]
                        if always_zero:
                            rows = [rows[0] if cond == "BEQ" else rows[1]]
                            rows[0]["note"] = "always zero after mask"
                        header = f"{cond} x{tracked_reg}, x0 @ 0x{cur_pc:08x}"
                        decision_groups.append({"header": header, "rows": rows})
                else:
                    detail = f"true -> 0x{branch_target:08x}, false -> 0x{fallthrough:08x}"
                    hints.append(f"{cond} x{tracked_reg}, x{other}: {detail}")
                    const_val = consts.get(other)
                    if const_val is not None and cond in ("BEQ", "BNE"):
                        compare_val = const_val & width_mask
                        decision_mask = state.get("origin_mask") if state else None
                        if decision_mask is not None:
                            compare_val &= decision_mask
                        diff = self._pick_nonzero_value(decision_mask, width_mask)
                        alt_val = (compare_val ^ diff) & width_mask
                        hints.append(
                            f"Compare against {self._format_hex(compare_val, width)} via x{other}"
                        )
                        if cond == "BEQ":
                            rows = [
                                {
                                    "value": compare_val,
                                    "pc": branch_target,
                                    "note": f"== {self._format_hex(compare_val, width)}",
                                    "path": "taken",
                                    "mask": decision_mask,
                                },
                                {
                                    "value": alt_val,
                                    "pc": fallthrough,
                                    "note": f"!= {self._format_hex(compare_val, width)}",
                                    "path": "fallthrough",
                                    "mask": decision_mask,
                                },
                            ]
                        else:
                            rows = [
                                {
                                    "value": alt_val,
                                    "pc": branch_target,
                                    "note": f"!= {self._format_hex(compare_val, width)}",
                                    "path": "taken",
                                    "mask": decision_mask,
                                },
                                {
                                    "value": compare_val,
                                    "pc": fallthrough,
                                    "note": f"== {self._format_hex(compare_val, width)}",
                                    "path": "fallthrough",
                                    "mask": decision_mask,
                                },
                            ]
                        header = (
                            f"{cond} x{tracked_reg}, x{other} @ 0x{cur_pc:08x} "
                            f"(x{other}={self._format_hex(const_val & width_mask, width)})"
                        )
                        decision_groups.append({"header": header, "rows": rows})
                        if branch_target < cur_pc:
                            loop_compare = {
                                "cond": cond,
                                "value": compare_val,
                                "mask": decision_mask,
                            }
                if state and state["valid"]:
                    if state["origin_mask"] == width_mask and not state.get("touched"):
                        full_value_branch = True
                    else:
                        record_active(state["origin_mask"])
            elif opcode == 0b0110011 and (rs1 in states or rs2 in states):
                state = states.get(rs1) or states.get(rs2)
                if state is not None:
                    state = dict(state)
                    state["valid"] = False
                    states[rd2] = state
                    hints.append("Value mixed with register operand")
            elif opcode == 0b1101111:
                imm = ((nxt >> 31) & 0x1) << 20
                imm |= ((nxt >> 21) & 0x3ff) << 1
                imm |= ((nxt >> 20) & 0x1) << 11
                imm |= ((nxt >> 12) & 0xff) << 12
                imm = self.sign_extend(imm, 21)
                target = (cur_pc + imm) & 0xffffffff
                if rd2 == 0 and target < cur_pc:
                    loop_hints.append(f"Loop ahead: JAL back to 0x{target:08x}")
            pc = (pc + sz) & 0xffffffff

            # Track immediate constants for compare detection.
            if opcode == 0b0110111:  # LUI
                if rd2 != 0:
                    consts[rd2] = nxt & 0xfffff000
            elif opcode == 0b0010111:  # AUIPC
                if rd2 != 0:
                    consts[rd2] = (cur_pc + (nxt & 0xfffff000)) & width_mask
            elif opcode == 0b0010011:
                imm = self.sign_extend(nxt >> 20, 12)
                if funct3 == 0x0:  # ADDI
                    if rs1 == 0:
                        if rd2 != 0:
                            consts[rd2] = imm & width_mask
                    elif rs1 in consts:
                        if rd2 != 0:
                            consts[rd2] = (consts[rs1] + imm) & width_mask
                    else:
                        consts.pop(rd2, None)
                elif funct3 == 0x6:  # ORI
                    if rs1 == 0:
                        if rd2 != 0:
                            consts[rd2] = imm & width_mask
                    elif rs1 in consts:
                        if rd2 != 0:
                            consts[rd2] = (consts[rs1] | (imm & width_mask)) & width_mask
                    else:
                        consts.pop(rd2, None)
                elif funct3 == 0x4:  # XORI
                    if rs1 == 0:
                        if rd2 != 0:
                            consts[rd2] = imm & width_mask
                    elif rs1 in consts:
                        if rd2 != 0:
                            consts[rd2] = (consts[rs1] ^ (imm & width_mask)) & width_mask
                    else:
                        consts.pop(rd2, None)
                elif funct3 == 0x7:  # ANDI
                    if rs1 in consts:
                        if rd2 != 0:
                            consts[rd2] = (consts[rs1] & (imm & width_mask)) & width_mask
                    else:
                        consts.pop(rd2, None)
                elif funct3 in (0x1, 0x5):  # shifts
                    shamt = (nxt >> 20) & 0x1f
                    if rs1 in consts:
                        if rd2 != 0:
                            if funct3 == 0x1:
                                consts[rd2] = (consts[rs1] << shamt) & width_mask
                            else:
                                if (nxt >> 25) & 0x7f == 0x20:
                                    consts[rd2] = (self._s32(consts[rs1]) >> shamt) & width_mask
                                else:
                                    consts[rd2] = (consts[rs1] >> shamt) & width_mask
                    else:
                        consts.pop(rd2, None)
                else:
                    consts.pop(rd2, None)
            elif opcode == 0b0000011:  # LOAD
                consts.pop(rd2, None)
            elif opcode == 0b0110011:  # OP (reg)
                consts.pop(rd2, None)
            elif opcode in (0b1101111, 0b1100111):  # JAL/JALR
                if rd2 != 0:
                    consts[rd2] = (cur_pc + sz) & width_mask
        if active_mask is not None:
            fields = self._mask_to_fields(reg, active_mask)
            fields_text = f" fields: {', '.join(fields)}" if fields else ""
            hints.append(f"Active bits mask 0x{active_mask:0{width*2}x}{fields_text}")
        elif full_value_branch:
            hints.append("Active bits: any (full register used)")
        if not branch_used:
            hints.append("No branch uses this value in the next 8 instructions")
        if loop_compare:
            compare_val = loop_compare["value"] & width_mask
            decision_mask = loop_compare.get("mask")
            diff = self._pick_nonzero_value(decision_mask, width_mask)
            alt_val = (compare_val ^ diff) & width_mask
            if loop_compare["cond"] == "BNE":
                hints.append(
                    f"Wait loop compares against {self._format_hex(compare_val, width)}; "
                    f"loop exits when value == {self._format_hex(compare_val, width)}"
                )
                hints.append(
                    f"Try seq={self._format_hex(alt_val, width)},{self._format_hex(compare_val, width)}"
                )
            else:
                hints.append(
                    f"Wait loop compares against {self._format_hex(compare_val, width)}; "
                    f"loop exits when value != {self._format_hex(compare_val, width)}"
                )
                hints.append(
                    f"Try seq={self._format_hex(compare_val, width)},{self._format_hex(alt_val, width)}"
                )
        elif loop_on_value:
            hints.append("Wait loop on this value; use seq=0,0,1 to change it over time")
        elif loop_hints and not branch_used:
            hints.append(loop_hints[0])
        return hints, active_mask, decision_groups

    def _prompt_assert_read(self, addr, size, reg):
        self._enable_input()
        print(f"\n[ASSERT] MMIO READ at 0x{addr:08x} size={size} PC=0x{self.pc:08x}")
        if reg is None and self.svd_index:
            block = self.svd_index.find_peripheral_block(addr)
            if block:
                print(f"[ASSERT] Peripheral: {block.name} 0x{block.start:08x}-0x{block.end:08x}")
        for line in self._format_register_hint(reg):
            print(line)
        hints, active_mask, decision_groups = self._decision_hints_for_read(reg, size)
        if self.assert_show_asm:
            for line in self._format_asm_window(self.pc, count=8):
                print(line)
        width_bytes = reg.width_bytes if reg is not None else max(1, size)
        self._print_decision_table(reg, decision_groups, width_bytes)
        self._print_field_details(reg, active_mask=active_mask)
        for hint in hints:
            print(f"[ASSERT] Hint: {hint}")
        default_value = None
        if reg is not None and reg.reset_value is not None:
            default_value = reg.reset_value
        if default_value is None:
            default_value = 0
        default_text = self._format_hex(default_value, size)
        while True:
            try:
                raw = input(
                    f"[ASSERT] Read value (default {default_text}, hex/dec, FIELD=VAL, seq=..., '-' ignore, ? help): "
                ).strip()
            except EOFError:
                self._halt("assert input unavailable")
            if raw in ("?", "help"):
                self._print_assert_help(is_write=False, reg=reg, active_mask=active_mask)
                continue
            if raw in ("-", "skip", "ignore"):
                return {"ignore": True}
            if raw == "":
                return {"value": default_value}
            try:
                return self._parse_read_input(raw, reg, size)
            except ValueError as e:
                print(f"[ASSERT] {e}")

    def _prompt_assert_write(self, addr, size, value, reg):
        self._enable_input()
        print(f"\n[ASSERT] MMIO WRITE at 0x{addr:08x} size={size} PC=0x{self.pc:08x}")
        print(f"[ASSERT] Value: {self._format_hex(value, size)}")
        if reg is None and self.svd_index:
            block = self.svd_index.find_peripheral_block(addr)
            if block:
                print(f"[ASSERT] Peripheral: {block.name} 0x{block.start:08x}-0x{block.end:08x}")
        for line in self._format_register_hint(reg):
            print(line)
        if self.assert_show_asm:
            for line in self._format_asm_window(self.pc, count=8):
                print(line)
        self._print_field_details(reg, value=value)
        default_text = self._format_hex(value, size)
        while True:
            try:
                raw = input(
                    f"[ASSERT] Write expect (default {default_text}, FIELD=VAL, mask=0xff, '-' ignore, ? help): "
                ).strip()
            except EOFError:
                self._halt("assert input unavailable")
            if raw in ("?", "help"):
                self._print_assert_help(is_write=True, reg=reg, value=value)
                continue
            if raw in ("-", "skip", "ignore"):
                return {"ignore": True}
            if raw == "":
                return {"value": value}
            try:
                return self._parse_write_input(raw, reg, size, value)
            except ValueError as e:
                print(f"[ASSERT] {e}")

    def _print_assert_help(self, is_write=False, reg=None, value=None, active_mask=None):
        print("[ASSERT] Input formats:")
        print("[ASSERT]   0x1f or 31")
        print("[ASSERT]   FIELD=0x3 or FIELD=ENUM")
        if not is_write:
            print("[ASSERT]   seq=0,0,1 [repeat] [hold_last=0]")
        if is_write:
            print("[ASSERT]   value=0x1 mask=0xff")
        print("[ASSERT]   '-' to ignore assertion")
        if reg is not None:
            self._print_field_details(
                reg, value=value if is_write else None, active_mask=active_mask, full=True
            )

    def _split_tokens(self, text):
        cleaned = text.replace(",", " ").replace(";", " ")
        return [t for t in cleaned.split() if t]

    def _parse_read_input(self, raw, reg, size):
        seq_entry = self._parse_sequence_input(raw)
        if seq_entry is not None:
            return seq_entry
        tokens = self._split_tokens(raw)
        if len(tokens) == 1 and "=" not in tokens[0]:
            value = self._parse_assert_int(tokens[0])
            if value is None:
                raise ValueError("Invalid value")
            return {"value": value}
        if reg is None:
            raise ValueError("Field assignments need --svd")
        field_map = self._field_map(reg)
        value = 0
        for token in tokens:
            if "=" not in token:
                raise ValueError("Expected FIELD=VAL")
            name, text = token.split("=", 1)
            field = field_map.get(name.strip().lower())
            if field is None:
                raise ValueError(f"Unknown field {name}")
            field_val = self._parse_field_value(field, text)
            if field_val is None:
                raise ValueError(f"Invalid value for {name}")
            value &= ~field.mask
            value |= (field_val << field.lsb) & field.mask
        return {"value": value}

    def _parse_write_input(self, raw, reg, size, actual_value):
        tokens = self._split_tokens(raw)
        expected = None
        mask = None
        field_mask = 0
        field_value = 0
        if reg is not None:
            field_map = self._field_map(reg)
        else:
            field_map = {}
        for token in tokens:
            if "=" in token:
                key, text = token.split("=", 1)
                key_lower = key.strip().lower()
                if key_lower in ("mask", "m"):
                    mask = self._parse_assert_int(text)
                    if mask is None:
                        raise ValueError("Invalid mask")
                elif key_lower in ("value", "val", "v"):
                    expected = self._parse_assert_int(text)
                    if expected is None:
                        raise ValueError("Invalid value")
                else:
                    field = field_map.get(key_lower)
                    if field is None:
                        raise ValueError(f"Unknown field {key}")
                    field_val = self._parse_field_value(field, text)
                    if field_val is None:
                        raise ValueError(f"Invalid value for {key}")
                    field_mask |= field.mask
                    field_value |= (field_val << field.lsb) & field.mask
            else:
                if expected is not None:
                    raise ValueError("Multiple values provided")
                expected = self._parse_assert_int(token)
                if expected is None:
                    raise ValueError("Invalid value")
        if field_mask:
            if expected is None:
                expected = field_value
            else:
                expected = (expected & ~field_mask) | field_value
            if mask is None:
                mask = field_mask
        if expected is None:
            expected = actual_value
        result = {"value": expected}
        if mask is not None:
            result["mask"] = mask
        return result

    def _parse_sequence_input(self, raw):
        text = raw.strip()
        if not text:
            return None
        parts = text.split()
        head = parts[0]
        if "=" not in head:
            return None
        key, seq_text = head.split("=", 1)
        key_lower = key.strip().lower()
        if key_lower not in ("seq", "sequence"):
            return None
        seq_text = seq_text.strip().strip(",;")
        tokens = [t for t in seq_text.replace(",", " ").replace(";", " ").split() if t]
        if not tokens:
            raise ValueError("Empty sequence")
        seq = []
        for token in tokens:
            val = self._parse_assert_int(token)
            if val is None:
                raise ValueError(f"Invalid sequence value {token}")
            seq.append(val)
        repeat = False
        hold_last = True
        for opt in parts[1:]:
            opt_lower = opt.strip().lower()
            if opt_lower in ("repeat", "loop", "r"):
                repeat = True
            elif opt_lower.startswith("repeat="):
                repeat = self._parse_assert_bool(opt.split("=", 1)[1], True)
            elif opt_lower.startswith("hold_last=") or opt_lower.startswith("hold="):
                hold_last = self._parse_assert_bool(opt.split("=", 1)[1], True)
            else:
                raise ValueError(f"Unknown sequence option {opt}")
        return {"sequence": seq, "repeat": repeat, "hold_last": hold_last}

    def _assert_missing(self, access, addr, reg):
        label = reg.path if reg else f"0x{addr:08x}"
        message = f"assert missing {access} for {label} at 0x{addr:08x}"
        print(f"[ASSERT] {message}")
        self._halt(message)

    def _ensure_assert_entry(self, addr, size, reg):
        if reg is not None:
            base = reg.address
            width = reg.width_bytes
        else:
            base = addr & ~0x3
            width = max(4, size)
        entry = self.assertions.get(base)
        if entry is None:
            entry = {"width": width}
            if reg is not None:
                entry["register"] = reg.path
                entry["peripheral"] = reg.peripheral
            self.assertions[base] = entry
            self.assert_dirty = True
        else:
            entry.setdefault("width", width)
        return base, entry

    def _fill_assert_meta(self, entry, reg):
        if entry is None or reg is None:
            return
        entry.setdefault("register", reg.path)
        entry.setdefault("peripheral", reg.peripheral)
        entry.setdefault("width", reg.width_bytes)

    def _assert_read(self, addr, size):
        if not self._assert_active() or not self.is_hardware_region(addr):
            return None, None
        reg = self.svd_index.find_register(addr) if self.svd_index else None
        base, entry, width = self._find_assert_entry(addr)
        if entry is None:
            if self.assert_assist_mode:
                value, base = self._mmio_state_read(addr, size)
                if value is not None:
                    return value, base
            if self.assert_assist_mode:
                base, entry = self._ensure_assert_entry(addr, size, reg)
            else:
                self._assert_missing("read", addr, reg)
        self._fill_assert_meta(entry, reg)
        read_entry = entry.get("read")
        if not read_entry:
            if self.assert_assist_mode:
                read_entry = self._prompt_assert_read(addr, size, reg)
                entry["read"] = read_entry
                self.assert_dirty = True
            else:
                self._assert_missing("read", addr, reg)
        if read_entry.get("ignore"):
            return 0, base
        full_value = self._stub_read_value(read_entry)
        shift = (addr - base) * 8
        mask = (1 << (size * 8)) - 1
        value = (full_value >> shift) & mask
        return value, base

    def _assert_write(self, addr, size, value):
        if not self._assert_active() or not self.is_hardware_region(addr):
            return None
        reg = self.svd_index.find_register(addr) if self.svd_index else None
        base, entry, width = self._find_assert_entry(addr)
        if entry is None:
            if self.assert_assist_mode:
                base, entry = self._ensure_assert_entry(addr, size, reg)
            else:
                self._assert_missing("write", addr, reg)
        self._fill_assert_meta(entry, reg)
        write_entry = entry.get("write")
        if not write_entry:
            if self.assert_assist_mode:
                write_entry = self._prompt_assert_write(addr, size, value, reg)
                entry["write"] = write_entry
                self.assert_dirty = True
            else:
                self._assert_missing("write", addr, reg)
        if write_entry.get("ignore"):
            return base
        width = max(entry.get("width", width or size or 4), size)
        full_mask = (1 << (width * 8)) - 1
        expected = write_entry.get("value")
        if expected is None:
            expected = value
        expected &= full_mask
        mask = write_entry.get("mask", full_mask) & full_mask
        shift = (addr - base) * 8
        access_mask = mask & (((1 << (size * 8)) - 1) << shift)
        if access_mask:
            write_full = (value & ((1 << (size * 8)) - 1)) << shift
            if (write_full & access_mask) != (expected & access_mask):
                message = (
                    f"assert write mismatch at 0x{addr:08x}: "
                    f"got {self._format_hex(write_full, width)} "
                    f"expected {self._format_hex(expected, width)} "
                    f"mask {self._format_hex(access_mask, width)}"
                )
                print(f"[ASSERT] {message}")
                self._halt(message)
        return base

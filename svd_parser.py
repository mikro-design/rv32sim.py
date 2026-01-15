import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional


def _local_name(tag):
    return tag.split("}", 1)[-1]


def _child(elem, name):
    for child in elem:
        if _local_name(child.tag) == name:
            return child
    return None


def _children(elem, name):
    for child in elem:
        if _local_name(child.tag) == name:
            yield child


def _child_text(elem, name, default=None):
    child = _child(elem, name)
    if child is None or child.text is None:
        return default
    text = child.text.strip()
    return text if text != "" else default


def _parse_int(text, default=None):
    if text is None:
        return default
    if isinstance(text, int):
        return text
    try:
        return int(str(text).strip(), 0)
    except (TypeError, ValueError):
        return default


def _expand_dim_indices(text, count):
    if not text:
        return [str(i) for i in range(count)]
    tokens = [t.strip() for t in str(text).split(",") if t.strip()]
    items = []
    for token in tokens:
        if "-" in token:
            start, end = token.split("-", 1)
            start = start.strip()
            end = end.strip()
            if start.isdigit() and end.isdigit():
                for idx in range(int(start), int(end) + 1):
                    items.append(str(idx))
            elif len(start) == 1 and len(end) == 1 and start.isalpha() and end.isalpha():
                for code in range(ord(start), ord(end) + 1):
                    items.append(chr(code))
            else:
                items.append(token)
        else:
            items.append(token)
    if len(items) < count:
        items.extend(str(i) for i in range(len(items), count))
    return items[:count]


def _format_dim_name(name, idx):
    if "%s" in name:
        return name.replace("%s", idx)
    if "%d" in name:
        return name.replace("%d", idx)
    return f"{name}{idx}"


@dataclass
class EnumValue:
    name: str
    value: int
    description: str = ""


@dataclass
class FieldInfo:
    name: str
    lsb: int
    msb: int
    description: str = ""
    enums: Dict[str, EnumValue] = field(default_factory=dict)

    @property
    def width(self):
        return self.msb - self.lsb + 1

    @property
    def mask(self):
        return ((1 << self.width) - 1) << self.lsb

    def format_range(self):
        if self.lsb == self.msb:
            return f"[{self.lsb}]"
        return f"[{self.msb}:{self.lsb}]"


@dataclass
class RegisterInfo:
    name: str
    path: str
    address: int
    size_bits: int
    access: Optional[str] = None
    description: str = ""
    reset_value: Optional[int] = None
    fields: List[FieldInfo] = field(default_factory=list)
    peripheral: str = ""

    @property
    def width_bytes(self):
        return max(1, (self.size_bits + 7) // 8)


@dataclass
class PeripheralBlock:
    name: str
    start: int
    end: int
    description: str = ""


class SvdIndex:
    def __init__(self, svd_path):
        self.svd_path = svd_path
        self.device_name = ""
        self.registers: List[RegisterInfo] = []
        self._registers_sorted: List[RegisterInfo] = []
        self._register_starts: List[int] = []
        self.peripheral_blocks: List[PeripheralBlock] = []
        self._parse()

    def find_register(self, addr):
        if not self._registers_sorted:
            return None
        import bisect
        idx = bisect.bisect_right(self._register_starts, addr) - 1
        if idx < 0:
            return None
        reg = self._registers_sorted[idx]
        if reg.address <= addr < reg.address + reg.width_bytes:
            return reg
        return None

    def find_peripheral_block(self, addr):
        for block in self.peripheral_blocks:
            if block.start <= addr < block.end:
                return block
        return None

    def _parse(self):
        tree = ET.parse(self.svd_path)
        root = tree.getroot()
        self.device_name = _child_text(root, "name", "") or ""
        peripherals = _child(root, "peripherals")
        if peripherals is None:
            return
        for periph in _children(peripherals, "peripheral"):
            self._parse_peripheral(periph)
        self._registers_sorted = sorted(self.registers, key=lambda r: r.address)
        self._register_starts = [r.address for r in self._registers_sorted]

    def _parse_peripheral(self, periph_elem):
        periph_name = _child_text(periph_elem, "name", "") or "PERIPH"
        base_address = _parse_int(_child_text(periph_elem, "baseAddress"), 0) or 0
        description = _child_text(periph_elem, "description", "") or ""
        reg_defaults = {
            "size": _parse_int(_child_text(periph_elem, "size"), 32) or 32,
            "access": _child_text(periph_elem, "access"),
            "resetValue": _parse_int(_child_text(periph_elem, "resetValue")),
        }
        for block in _children(periph_elem, "addressBlock"):
            offset = _parse_int(_child_text(block, "offset"), 0) or 0
            size = _parse_int(_child_text(block, "size"), 0) or 0
            if size:
                start = base_address + offset
                end = start + size
                self.peripheral_blocks.append(
                    PeripheralBlock(name=periph_name, start=start, end=end, description=description)
                )
        registers = _child(periph_elem, "registers")
        if registers is None:
            return
        self._parse_registers(registers, base_address, [periph_name], reg_defaults, periph_name)

    def _parse_registers(self, registers_elem, base_address, prefix, defaults, periph_name):
        for child in registers_elem:
            tag = _local_name(child.tag)
            if tag == "register":
                self._parse_register(child, base_address, prefix, defaults, periph_name)
            elif tag == "cluster":
                self._parse_cluster(child, base_address, prefix, defaults, periph_name)

    def _parse_cluster(self, cluster_elem, base_address, prefix, defaults, periph_name):
        cluster_name = _child_text(cluster_elem, "name", "") or "CLUSTER"
        offset = _parse_int(_child_text(cluster_elem, "addressOffset"), 0) or 0
        dim = _parse_int(_child_text(cluster_elem, "dim"), 0) or 0
        dim_inc = _parse_int(_child_text(cluster_elem, "dimIncrement"), 0) or 0
        dim_index = _expand_dim_indices(_child_text(cluster_elem, "dimIndex"), dim) if dim else []
        if dim:
            if dim_inc == 0:
                dim_inc = 0
            for idx, dim_name in enumerate(dim_index):
                name = _format_dim_name(cluster_name, dim_name)
                new_prefix = prefix + [name]
                new_base = base_address + offset + idx * dim_inc
                self._parse_registers(cluster_elem, new_base, new_prefix, defaults, periph_name)
        else:
            new_prefix = prefix + [cluster_name]
            new_base = base_address + offset
            self._parse_registers(cluster_elem, new_base, new_prefix, defaults, periph_name)

    def _parse_register(self, reg_elem, base_address, prefix, defaults, periph_name):
        reg_name = _child_text(reg_elem, "name", "") or "REG"
        offset = _parse_int(_child_text(reg_elem, "addressOffset"), 0) or 0
        size_bits = _parse_int(_child_text(reg_elem, "size"), defaults.get("size", 32)) or 32
        access = _child_text(reg_elem, "access", defaults.get("access"))
        description = _child_text(reg_elem, "description", "") or ""
        reset_value = _parse_int(_child_text(reg_elem, "resetValue"), defaults.get("resetValue"))
        dim = _parse_int(_child_text(reg_elem, "dim"), 0) or 0
        dim_inc = _parse_int(_child_text(reg_elem, "dimIncrement"), 0) or 0
        dim_index = _expand_dim_indices(_child_text(reg_elem, "dimIndex"), dim) if dim else []

        def build_register(name, addr):
            path = ".".join(prefix + [name])
            fields = self._parse_fields(reg_elem)
            reg = RegisterInfo(
                name=name,
                path=path,
                address=addr,
                size_bits=size_bits,
                access=access,
                description=description,
                reset_value=reset_value,
                fields=fields,
                peripheral=periph_name,
            )
            self.registers.append(reg)

        if dim:
            if dim_inc == 0:
                dim_inc = max(1, size_bits // 8)
            for idx, dim_name in enumerate(dim_index):
                name = _format_dim_name(reg_name, dim_name)
                addr = base_address + offset + idx * dim_inc
                build_register(name, addr)
        else:
            addr = base_address + offset
            build_register(reg_name, addr)

    def _parse_fields(self, reg_elem):
        fields_elem = _child(reg_elem, "fields")
        if fields_elem is None:
            return []
        fields = []
        for field_elem in _children(fields_elem, "field"):
            name = _child_text(field_elem, "name", "") or "FIELD"
            description = _child_text(field_elem, "description", "") or ""
            lsb = None
            msb = None
            bit_offset = _parse_int(_child_text(field_elem, "bitOffset"))
            bit_width = _parse_int(_child_text(field_elem, "bitWidth"))
            if bit_offset is not None and bit_width is not None:
                lsb = bit_offset
                msb = bit_offset + bit_width - 1
            if lsb is None or msb is None:
                lsb = _parse_int(_child_text(field_elem, "lsb"))
                msb = _parse_int(_child_text(field_elem, "msb"))
            if lsb is None or msb is None:
                bit_range = _child_text(field_elem, "bitRange")
                if bit_range:
                    cleaned = bit_range.strip().strip("[]")
                    if ":" in cleaned:
                        hi, lo = cleaned.split(":", 1)
                    elif "-" in cleaned:
                        hi, lo = cleaned.split("-", 1)
                    else:
                        hi, lo = cleaned, cleaned
                    msb = _parse_int(hi)
                    lsb = _parse_int(lo)
            if lsb is None or msb is None:
                continue
            enums = self._parse_enums(field_elem)
            fields.append(FieldInfo(name=name, lsb=int(lsb), msb=int(msb), description=description, enums=enums))
        return fields

    def _parse_enums(self, field_elem):
        enums = {}
        for enum_group in _children(field_elem, "enumeratedValues"):
            for enum_elem in _children(enum_group, "enumeratedValue"):
                name = _child_text(enum_elem, "name")
                if not name:
                    continue
                value = _parse_int(_child_text(enum_elem, "value"))
                if value is None:
                    continue
                description = _child_text(enum_elem, "description", "") or ""
                enums[name] = EnumValue(name=name, value=value, description=description)
        return enums

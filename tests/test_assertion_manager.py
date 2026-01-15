import json
from types import SimpleNamespace

import pytest

import assertion_manager
from rv32sim import RV32Sim, HaltException
from svd_parser import EnumValue, FieldInfo, RegisterInfo


def _encode_i_type(imm, rs1, funct3, rd, opcode=0x13):
    imm &= 0xfff
    return (
        (imm << 20)
        | ((rs1 & 0x1f) << 15)
        | ((funct3 & 0x7) << 12)
        | ((rd & 0x1f) << 7)
        | (opcode & 0x7f)
    )


def _encode_b_type(imm, rs2, rs1, funct3, opcode=0x63):
    imm &= 0x1fff
    return (
        ((imm >> 12) & 0x1) << 31
        | ((imm >> 5) & 0x3f) << 25
        | ((rs2 & 0x1f) << 20)
        | ((rs1 & 0x1f) << 15)
        | ((funct3 & 0x7) << 12)
        | ((imm >> 1) & 0xf) << 8
        | ((imm >> 11) & 0x1) << 7
        | (opcode & 0x7f)
    )


def _encode_r_type(funct7, rs2, rs1, funct3, rd, opcode=0x33):
    return (
        ((funct7 & 0x7f) << 25)
        | ((rs2 & 0x1f) << 20)
        | ((rs1 & 0x1f) << 15)
        | ((funct3 & 0x7) << 12)
        | ((rd & 0x1f) << 7)
        | (opcode & 0x7f)
    )


def _encode_s_type(imm, rs2, rs1, funct3, opcode=0x23):
    imm &= 0xfff
    imm_11_5 = (imm >> 5) & 0x7f
    imm_4_0 = imm & 0x1f
    return (
        (imm_11_5 << 25)
        | ((rs2 & 0x1f) << 20)
        | ((rs1 & 0x1f) << 15)
        | ((funct3 & 0x7) << 12)
        | (imm_4_0 << 7)
        | (opcode & 0x7f)
    )


def _encode_u_type(imm, rd, opcode):
    return (imm & 0xfffff000) | ((rd & 0x1f) << 7) | (opcode & 0x7f)


def _encode_j_type(imm, rd, opcode=0x6F):
    imm &= 0x1FFFFF
    return (
        ((imm >> 20) & 0x1) << 31
        | ((imm >> 12) & 0xFF) << 12
        | ((imm >> 11) & 0x1) << 20
        | ((imm >> 1) & 0x3FF) << 21
        | ((rd & 0x1F) << 7)
        | (opcode & 0x7F)
    )

def _encode_csr(funct3, csr, rs1, rd):
    return (csr << 20) | ((rs1 & 0x1f) << 15) | ((funct3 & 0x7) << 12) | ((rd & 0x1f) << 7) | 0x73


def _make_reg():
    fields = [
        FieldInfo(
            name="EN",
            lsb=0,
            msb=0,
            enums={"DIS": EnumValue(name="DIS", value=0), "EN": EnumValue(name="EN", value=1)},
        ),
        FieldInfo(name="MODE", lsb=1, msb=2),
    ]
    return RegisterInfo(
        name="CTRL",
        path="PERIPH.CTRL",
        address=0x40000000,
        size_bits=32,
        fields=fields,
        peripheral="PERIPH",
    )


def _make_reg_many_fields(count=9, with_enums=False, description=""):
    fields = []
    for idx in range(count):
        enums = {}
        if with_enums and idx == 0:
            enums = {
                "ZERO": EnumValue(name="ZERO", value=0),
                "ONE": EnumValue(name="ONE", value=1),
                "TWO": EnumValue(name="TWO", value=2),
                "THREE": EnumValue(name="THREE", value=3),
                "FOUR": EnumValue(name="FOUR", value=4),
                "FIVE": EnumValue(name="FIVE", value=5),
                "SIX": EnumValue(name="SIX", value=6),
            }
        fields.append(FieldInfo(name=f"F{idx}", lsb=idx, msb=idx, description=description, enums=enums))
    size_bits = max(8, count)
    return RegisterInfo(
        name="BIG",
        path="PERIPH.BIG",
        address=0x40000020,
        size_bits=size_bits,
        fields=fields,
        peripheral="PERIPH",
        description=description,
        reset_value=0x1,
    )


def _init_mem(sim, base, size=0x100):
    sim.memory_regions = []
    sim._memory_initialized = True
    sim._add_memory_region(base, base + size, "ram")


def test_parse_sequence_input():
    sim = RV32Sim()
    am = sim.assert_manager
    entry = am._parse_sequence_input("seq=1,2 repeat hold_last=0")
    assert entry == {"sequence": [1, 2], "repeat": True, "hold_last": False}
    with pytest.raises(ValueError):
        am._parse_sequence_input("seq=1 foo=bar")


def test_parse_sequence_input_edge_cases():
    sim = RV32Sim()
    am = sim.assert_manager
    assert am._parse_sequence_input("   ") is None
    with pytest.raises(ValueError):
        am._parse_sequence_input("seq=")
    with pytest.raises(ValueError):
        am._parse_sequence_input("seq=1,zz")


def test_parse_read_input_fields():
    sim = RV32Sim()
    am = sim.assert_manager
    reg = _make_reg()
    result = am._parse_read_input("EN=1 MODE=2", reg, 4)
    assert result["value"] == 0b101

    result = am._parse_read_input("0x10", reg, 4)
    assert result["value"] == 0x10

    with pytest.raises(ValueError):
        am._parse_read_input("BAD=1", reg, 4)

    with pytest.raises(ValueError):
        am._parse_read_input("EN=1", None, 4)


def test_parse_read_input_sequence_and_errors():
    sim = RV32Sim()
    am = sim.assert_manager
    reg = _make_reg()
    entry = am._parse_read_input("seq=1 repeat=0", reg, 4)
    assert entry["sequence"] == [1]
    assert entry["repeat"] is False
    assert entry["hold_last"] is True
    with pytest.raises(ValueError):
        am._parse_read_input("bad", reg, 4)
    with pytest.raises(ValueError):
        am._parse_read_input("EN MODE=1", reg, 4)
    with pytest.raises(ValueError):
        am._parse_read_input("EN=BAD", reg, 4)


def test_parse_write_input_fields_and_mask():
    sim = RV32Sim()
    am = sim.assert_manager
    reg = _make_reg()
    result = am._parse_write_input("MODE=3", reg, 4, 0)
    assert result["value"] == 0b110
    assert result["mask"] == reg.fields[1].mask

    result = am._parse_write_input("value=0x12 mask=0x0f", reg, 4, 0)
    assert result == {"value": 0x12, "mask": 0x0f}

    result = am._parse_write_input("0x5", reg, 4, 0)
    assert result == {"value": 0x5}


def test_parse_write_input_errors_and_merge():
    sim = RV32Sim()
    am = sim.assert_manager
    reg = _make_reg()
    with pytest.raises(ValueError):
        am._parse_write_input("value=bad", reg, 4, 0)
    with pytest.raises(ValueError):
        am._parse_write_input("BAD=1", reg, 4, 0)
    with pytest.raises(ValueError):
        am._parse_write_input("EN=BAD", reg, 4, 0)
    with pytest.raises(ValueError):
        am._parse_write_input("1 2", reg, 4, 0)
    with pytest.raises(ValueError):
        am._parse_write_input("bad", reg, 4, 0)
    merged = am._parse_write_input("value=0x0 EN=1", reg, 4, 0)
    assert merged["value"] == 0x1
    defaulted = am._parse_write_input("mask=0xff", reg, 4, 0x12)
    assert defaulted["value"] == 0x12
    assert defaulted["mask"] == 0xFF


def test_mmio_state_read_write():
    sim = RV32Sim()
    am = sim.assert_manager
    base = 0x40000000
    am._mmio_state_write(base + 2, 1, 0xAA)
    value, found_base = am._mmio_state_read(base, 4)
    assert found_base == base
    assert value == 0x00AA0000


def test_assert_read_and_write_paths():
    sim = RV32Sim()
    am = sim.assert_manager
    base = 0x40000000
    sim.assert_mode = True
    sim.assertions = {
        base: {
            "width": 4,
            "read": {"value": 0x11223344},
            "write": {"value": 0x11223344},
        }
    }

    value, found_base = am._assert_read(base + 1, 2)
    assert found_base == base
    assert value == 0x2233

    assert am._assert_write(base, 4, 0x11223344) == base

    sim.assertions[base]["read"] = {"ignore": True}
    value, found_base = am._assert_read(base, 4)
    assert value == 0
    assert found_base == base

    sim.assertions[base]["write"] = {"ignore": True}
    assert am._assert_write(base, 4, 0x0) == base


def test_assert_write_mismatch_raises():
    sim = RV32Sim()
    am = sim.assert_manager
    base = 0x40000000
    sim.assert_mode = True
    sim.assertions = {base: {"width": 4, "write": {"value": 0x12345678}}}
    with pytest.raises(HaltException):
        am._assert_write(base, 4, 0x12345679)


def test_assert_missing_raises():
    sim = RV32Sim()
    am = sim.assert_manager
    sim.assert_mode = True
    with pytest.raises(HaltException):
        am._assert_read(0x40000000, 4)


def test_assert_assist_uses_mmio_state():
    sim = RV32Sim()
    am = sim.assert_manager
    sim.assert_assist_mode = True
    base = 0x40000000
    am._mmio_state_write(base, 4, 0xAABBCCDD)
    value, found_base = am._assert_read(base, 4)
    assert found_base == base
    assert value == 0xAABBCCDD


def test_format_and_parse_helpers():
    sim = RV32Sim()
    am = sim.assert_manager
    assert am._format_hex(0x1) == "0x1"
    assert am._format_hex(0x1, 2) == "0x0001"
    assert am._parse_assert_int("0x10") == 16
    assert am._parse_assert_int("bad", default=7) == 7
    assert am._parse_assert_int(True) == 1
    assert am._parse_assert_bool("yes") is True
    assert am._parse_assert_bool("0") is False


def test_normalize_and_serialize_entries():
    sim = RV32Sim()
    am = sim.assert_manager
    raw = {"width": "4", "value": "0x10", "repeat": "1", "hold_last": "0"}
    entry = am._normalize_assert_entry(raw)
    assert entry["width"] == 4
    assert entry["read"]["value"] == 0x10

    read_entry = {"sequence": [1, 2], "repeat": True, "hold_last": False}
    assert am._serialize_assert_read(read_entry, 1) == {
        "sequence": ["0x01", "0x02"],
        "repeat": True,
        "hold_last": False,
    }

    write_entry = {"value": 0x12, "mask": 0x0f}
    assert am._serialize_assert_write(write_entry, 1) == {"value": "0x12", "mask": "0x0f"}

    assert am._normalize_assert_read({"ignore": True}) == {"ignore": True}
    assert am._normalize_assert_write({"ignore": True}) == {"ignore": True}


def test_normalize_scalar_and_serialize_ignore():
    sim = RV32Sim()
    am = sim.assert_manager
    assert am._normalize_assert_entry("0x2") == {"read": {"value": 2}}
    assert am._serialize_assert_read({"ignore": True}, 1) == {"ignore": True}

    base = 0x40000000
    sim.mmio_state[base] = {"width": 4, "value": 0}
    found_base, _, width = am._find_mmio_state_entry(base + 2)
    assert found_base == base
    assert width == 4


def test_find_entry_alignment():
    sim = RV32Sim()
    am = sim.assert_manager
    base = 0x40000000
    sim.assertions[base] = {"width": 4}
    found_base, _, width = am._find_assert_entry(base + 2)
    assert found_base == base
    assert width == 4


def test_format_register_hint_and_fields(capsys):
    sim = RV32Sim()
    am = sim.assert_manager
    reg = _make_reg()
    reg.description = "Description"
    reg.reset_value = 1
    hints = am._format_register_hint(reg)
    assert any("PERIPH.CTRL" in line for line in hints)
    am._print_field_details(reg, value=0b101, full=True)
    am._print_field_details(reg, active_mask=reg.fields[0].mask)
    am._print_field_details(reg, active_mask=None)
    am._print_field_details(reg, value=0, active_mask=None)
    captured = capsys.readouterr()
    assert "[ASSERT]" in captured.out


def test_field_map_and_enum_parse():
    sim = RV32Sim()
    am = sim.assert_manager
    reg = _make_reg()
    field_map = am._field_map(reg)
    assert "en" in field_map
    field = field_map["en"]
    assert am._parse_field_value(field, "EN") == 1
    assert am._parse_field_value(field, "0") == 0


def test_disasm_instruction_variants():
    sim = RV32Sim()
    am = sim.assert_manager
    assert "lui" in am._disasm_instruction(0, 0x12345037)
    assert "auipc" in am._disasm_instruction(0, 0x12345017)
    assert "jal" in am._disasm_instruction(0, 0x0000006f)
    assert "jalr" in am._disasm_instruction(0, _encode_i_type(4, 1, 0x0, 2, opcode=0x67))
    assert "beq" in am._disasm_instruction(0, _encode_b_type(4, 2, 1, 0x0))
    assert "lw" in am._disasm_instruction(0, _encode_i_type(4, 1, 0x2, 2, opcode=0x03))
    assert "sw" in am._disasm_instruction(0, _encode_s_type(4, 2, 1, 0x2))
    assert "addi" in am._disasm_instruction(0, _encode_i_type(1, 1, 0x0, 2))
    assert "and" in am._disasm_instruction(0, _encode_r_type(0x00, 2, 1, 0x7, 3))
    assert "ecall" in am._disasm_instruction(0, 0x00000073)


def test_format_asm_window_and_hints():
    sim = RV32Sim()
    am = sim.assert_manager
    sim.memory_regions = []
    sim._memory_initialized = True
    base = 0x20000000
    sim._add_memory_region(base, base + 0x100, "ram")

    load = _encode_i_type(0, 1, 0x2, 5, opcode=0x03)
    andi = _encode_i_type(0x3, 5, 0x7, 5)
    beq = _encode_b_type(4, 0, 5, 0x0)
    sim._write_memory(base, load.to_bytes(4, "little"))
    sim._write_memory(base + 4, andi.to_bytes(4, "little"))
    sim._write_memory(base + 8, beq.to_bytes(4, "little"))

    sim.pc = base
    sim.last_instr = load
    hints, active_mask, decision_groups = am._decision_hints_for_read(_make_reg(), 4)
    assert hints
    assert active_mask is not None
    assert decision_groups

    lines = am._format_asm_window(base, count=3)
    assert len(lines) >= 2


def test_prompt_read_write_paths(monkeypatch):
    sim = RV32Sim()
    am = sim.assert_manager
    reg = _make_reg()
    inputs = iter(["?", "", "-", "0x5"])

    def fake_input(_):
        return next(inputs)

    monkeypatch.setattr("builtins.input", fake_input)
    read_entry = am._prompt_assert_read(reg.address, 4, reg)
    assert read_entry["value"] == reg.reset_value or read_entry["value"] == 0
    write_entry = am._prompt_assert_write(reg.address, 4, 0x5, reg)
    assert write_entry.get("ignore") is True


def test_prompt_write_help_and_default(monkeypatch):
    sim = RV32Sim()
    am = sim.assert_manager
    reg = _make_reg()
    inputs = iter(["?", ""])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    write_entry = am._prompt_assert_write(reg.address, 4, 0xA5, reg)
    assert write_entry["value"] == 0xA5


def test_assert_config_load_save(tmp_path, capsys):
    sim = RV32Sim()
    am = sim.assert_manager
    config = {
        "assertions": {
            "0x40000000": {"width": 4, "read": {"value": "0x1"}, "write": {"value": "0x2"}},
        }
    }
    path = tmp_path / "assert.json"
    path.write_text(json.dumps(config))
    am.load_assert_config(str(path))
    assert 0x40000000 in sim.assertions
    out = tmp_path / "out.json"
    am.save_assert_config(str(out))
    saved = json.loads(out.read_text())
    assert "assertions" in saved
    captured = capsys.readouterr()
    assert "Loaded" in captured.out or "Saved" in captured.out


def test_svd_load_missing(capsys):
    sim = RV32Sim()
    sim.assert_manager.load_svd("missing.svd")
    captured = capsys.readouterr()
    assert "SVD not found" in captured.out


def test_svd_load_success(monkeypatch, capsys):
    sim = RV32Sim()
    am = sim.assert_manager
    sentinel = object()
    monkeypatch.setattr(assertion_manager, "SvdIndex", lambda _filename: sentinel)
    am.load_svd("demo.svd")
    assert am.svd_index is sentinel
    assert am._svd_field_cache == {}
    assert "Loaded SVD" in capsys.readouterr().out


def test_description_and_enum_helpers():
    sim = RV32Sim()
    am = sim.assert_manager
    assert am._clean_description("reserved") == ""
    assert am._clean_description("Description") == ""
    assert am._clean_description("A\nB") == "A"
    reg = _make_reg()
    field = reg.fields[0]
    am.assert_verbose = False
    assert "EN" in am._enum_items(field)
    am.assert_verbose = True
    assert "DIS" in am._enum_items(field)
    assert "EN" in am._field_value_label(field, 1)


def test_format_field_assignments_and_masks():
    sim = RV32Sim()
    am = sim.assert_manager
    reg = _make_reg()
    mask = reg.fields[0].mask
    text = am._format_field_assignments(reg, 1, mask)
    assert "EN" in text
    assert am._mask_to_fields(reg, mask) == ["EN"]


def test_print_decision_table(capsys):
    sim = RV32Sim()
    am = sim.assert_manager
    reg = _make_reg()
    groups = [
        {"header": "test", "rows": [{"value": 1, "pc": 0x100, "note": "n", "path": "p"}]},
    ]
    am._print_decision_table(reg, groups, 4)
    assert "[ASSERT]" in capsys.readouterr().out


def test_disasm_more_variants():
    sim = RV32Sim()
    am = sim.assert_manager
    instr = _encode_u_type(0x12345000, 1, opcode=0x37)
    assert "lui" in am._disasm_instruction(0, instr)
    instr = _encode_u_type(0x12345000, 1, opcode=0x17)
    assert "auipc" in am._disasm_instruction(0, instr)
    instr = _encode_csr(0x1, 0x300, 2, 1)
    assert "csrrw" in am._disasm_instruction(0, instr)
    instr = _encode_csr(0x5, 0x300, 1, 2)
    assert "csrrwi" in am._disasm_instruction(0, instr)


def test_decision_hints_with_shifts_and_branch():
    sim = RV32Sim()
    am = sim.assert_manager
    sim.memory_regions = []
    sim._memory_initialized = True
    base = 0x20000100
    sim._add_memory_region(base, base + 0x40, "ram")

    load = _encode_i_type(0, 1, 0x2, 5, opcode=0x03)  # lw x5, 0(x1)
    slli = _encode_i_type(1, 5, 0x1, 5)  # slli x5, x5, 1
    srli = _encode_i_type(1, 5, 0x5, 5)  # srli x5, x5, 1
    beq = _encode_b_type(4, 0, 5, 0x0)  # beq x5, x0
    sim._write_memory(base, load.to_bytes(4, "little"))
    sim._write_memory(base + 4, slli.to_bytes(4, "little"))
    sim._write_memory(base + 8, srli.to_bytes(4, "little"))
    sim._write_memory(base + 12, beq.to_bytes(4, "little"))

    sim.pc = base
    sim.last_instr = load
    hints, active_mask, decision_groups = am._decision_hints_for_read(_make_reg(), 4)
    assert hints
    assert decision_groups


def test_assert_assist_prompts(monkeypatch):
    sim = RV32Sim()
    am = sim.assert_manager
    sim.assert_assist_mode = True
    reg = _make_reg()
    inputs = iter(["0x1", "value=0x2 mask=0xff"])

    def fake_input(_):
        return next(inputs)

    monkeypatch.setattr("builtins.input", fake_input)
    value, _base = am._assert_read(reg.address, 4)
    assert value == 1
    am._assert_write(reg.address, 4, 0x2)


def test_decision_hints_complex_sequence():
    sim = RV32Sim()
    am = sim.assert_manager
    sim.memory_regions = []
    sim._memory_initialized = True
    base = 0x20001000
    sim._add_memory_region(base, base + 0x80, "ram")

    load = _encode_i_type(0, 1, 0x2, 5, opcode=0x03)  # lw x5, 0(x1)
    andi = _encode_i_type(0x0f, 5, 0x7, 5)  # andi x5, x5, 0xf
    slli = _encode_i_type(1, 5, 0x1, 5)  # slli x5, x5, 1
    srli = _encode_i_type(1, 5, 0x5, 5)  # srli x5, x5, 1
    slti = _encode_i_type(4, 5, 0x2, 6)  # slti x6, x5, 4
    lui = _encode_u_type(0x1000, 7, opcode=0x37)  # lui x7, 0x1000
    addi = _encode_i_type(2, 7, 0x0, 7)  # addi x7, x7, 2
    beq = _encode_b_type(-8 & 0x1fff, 7, 5, 0x0)  # beq x5, x7, -8
    jal = 0xFFDFF06F  # jal x0, -4

    sim._write_memory(base, load.to_bytes(4, "little"))
    sim._write_memory(base + 4, andi.to_bytes(4, "little"))
    sim._write_memory(base + 8, slli.to_bytes(4, "little"))
    sim._write_memory(base + 12, srli.to_bytes(4, "little"))
    sim._write_memory(base + 16, slti.to_bytes(4, "little"))
    sim._write_memory(base + 20, lui.to_bytes(4, "little"))
    sim._write_memory(base + 24, addi.to_bytes(4, "little"))
    sim._write_memory(base + 28, beq.to_bytes(4, "little"))
    sim._write_memory(base + 32, jal.to_bytes(4, "little"))

    sim.pc = base
    sim.last_instr = load
    hints, active_mask, decision_groups = am._decision_hints_for_read(_make_reg(), 4)
    assert hints
    assert active_mask is not None
    assert decision_groups


def test_disasm_at_and_peek_instruction():
    sim = RV32Sim()
    am = sim.assert_manager
    sim.memory_regions = []
    sim._memory_initialized = True
    base = 0x20002000
    sim._add_memory_region(base, base + 0x20, "ram")
    instr = _encode_i_type(0, 1, 0x0, 2)
    sim._write_memory(base, instr.to_bytes(4, "little"))
    assert am._disasm_at(base) is not None
    assert am._peek_instruction(base)[0] is not None


def test_assertion_manager_helper_branches(monkeypatch):
    sim = RV32Sim()
    am = sim.assert_manager
    am.sim = sim
    am._input_ready = False
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "readline":
            raise ImportError("no readline")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    am._enable_input()
    assert am._input_ready is True
    assert am._parse_assert_int(object(), default=9) == 9
    assert am._parse_assert_bool(2) is True
    assert am._parse_assert_bool("maybe") is True
    field = _make_reg().fields[0]
    assert am._parse_field_value(field, None) is None
    assert am._format_imm(-5) == "-0x5"
    assert am._mask_to_fields(None, 1) == []


def test_load_svd_generic_error(monkeypatch, capsys):
    sim = RV32Sim()
    am = sim.assert_manager

    def boom(_):
        raise RuntimeError("boom")

    monkeypatch.setattr(assertion_manager, "SvdIndex", boom)
    am.load_svd("bad.svd")
    assert "Error loading SVD" in capsys.readouterr().out


def test_load_assert_config_errors(tmp_path, capsys):
    sim = RV32Sim()
    am = sim.assert_manager
    am.load_assert_config(str(tmp_path / "missing.json"))
    assert "No assertion config found" in capsys.readouterr().out

    bad = tmp_path / "bad.json"
    bad.write_text("{")
    am.load_assert_config(str(bad))
    assert "Error loading assertion config" in capsys.readouterr().out


def test_load_assert_config_entries_variants(tmp_path, capsys):
    sim = RV32Sim()
    am = sim.assert_manager
    bad_entries = tmp_path / "bad_entries.json"
    bad_entries.write_text(json.dumps({"assertions": []}))
    am.load_assert_config(str(bad_entries))
    assert "Assertion config has no assertions" in capsys.readouterr().out

    sim = RV32Sim()
    am = sim.assert_manager
    hw_path = tmp_path / "hw.json"
    hw_path.write_text(json.dumps({"hw_stubs": {"0x10": {"value": 1}}}))
    am.load_assert_config(str(hw_path))
    assert 0x10 in sim.assertions

    sim = RV32Sim()
    am = sim.assert_manager
    skip_path = tmp_path / "skip.json"
    skip_path.write_text(json.dumps({"assertions": {"bad": "nope", "0x20": None}}))
    am.load_assert_config(str(skip_path))
    assert sim.assertions == {}


def test_save_assert_config_includes_metadata(tmp_path):
    sim = RV32Sim()
    am = sim.assert_manager
    sim.assertions = {
        0x40000000: {"width": 4, "register": "REG", "peripheral": "PER", "comment": "note"}
    }
    path = tmp_path / "out.json"
    am.save_assert_config(str(path))
    data = json.loads(path.read_text())
    entry = data["assertions"]["0x40000000"]
    assert entry["register"] == "REG"
    assert entry["peripheral"] == "PER"
    assert entry["comment"] == "note"


def test_normalize_assert_entry_branches():
    sim = RV32Sim()
    am = sim.assert_manager
    raw = {
        "register": 123,
        "peripheral": "P",
        "comment": 7,
        "read": {"sequence": ["1", "bad"], "repeat": "1", "hold_last": "0"},
    }
    entry = am._normalize_assert_entry(raw)
    assert entry["register"] == "123"
    assert entry["comment"] == "7"
    assert entry["read"]["sequence"] == [1]
    assert entry["read"]["repeat"] is True
    assert entry["read"]["hold_last"] is False
    assert am._normalize_assert_entry(None) is None
    assert am._normalize_assert_read(None) is None
    assert am._normalize_assert_read("0x10") == {"value": 0x10}
    assert am._normalize_assert_write({"value": "0x1", "mask": "0xff"}) == {"value": 1, "mask": 0xFF}
    assert am._normalize_assert_write("0x2") == {"value": 2}
    assert am._serialize_assert_write({"ignore": True}, 1) == {"ignore": True}


def test_find_entries_and_mmio_state_alignment():
    sim = RV32Sim()
    am = sim.assert_manager
    base = 0x40000002
    sim.assertions[base] = {"width": 2}
    found_base, _, width = am._find_assert_entry(base + 1)
    assert found_base == base
    assert width == 2

    sim.mmio_state[base] = {"width": 2, "value": 0x1122}
    found_base, _, width = am._find_mmio_state_entry(base + 1)
    assert found_base == base
    assert width == 2

    reg = RegisterInfo(
        name="REG",
        path="P.REG",
        address=0x40001000,
        size_bits=16,
        fields=[],
        peripheral="P",
    )
    assert am._mmio_state_write(reg.address, 2, 0x1234, reg=reg) == reg.address

    sim.mmio_state[0x40002000] = {"width": 1, "value": 0}
    am._mmio_state_write(0x40002000, 4, 0xAABBCCDD)
    assert sim.mmio_state[0x40002000]["width"] >= 4


def test_register_hint_enum_and_description_branches():
    sim = RV32Sim()
    am = sim.assert_manager
    reg = _make_reg_many_fields(count=9, description="desc")
    reg.access = "rw"
    hints = am._format_register_hint(reg)
    assert any("Info:" in line for line in hints)
    assert any("Fields:" in line and "..." in line for line in hints)

    field = _make_reg_many_fields(count=1, with_enums=True).fields[0]
    am.assert_verbose = False
    assert "..." in am._enum_items(field)


def test_clean_description_variants():
    sim = RV32Sim()
    am = sim.assert_manager
    assert am._clean_description(123) == "123"
    assert am._clean_description("   ") == ""
    assert am._clean_description("reserved") == ""
    assert am._clean_description("defaultdict(foo)") == ""
    am.assert_verbose = False
    long_text = "a" * 130
    assert am._clean_description(long_text).endswith("...")


def test_print_field_details_and_assignments(capsys):
    sim = RV32Sim()
    am = sim.assert_manager
    reg = _make_reg_many_fields(count=2, with_enums=True, description="detail")
    am._print_field_details(reg, value=1, active_mask=1, full=False)
    am._print_field_details(reg, value=1, active_mask=1, full=True)
    out = capsys.readouterr().out
    assert "detail" in out
    assert "enums:" in out

    reg_many = _make_reg_many_fields(count=5)
    full_mask = (1 << (reg_many.width_bytes * 8)) - 1
    assert am._format_field_assignments(reg_many, 0, full_mask) == ""

    reg_more = _make_reg_many_fields(count=7)
    assert am._format_field_assignments(reg_more, 0, 0x7F) == ""

    reg_enum = _make_reg()
    text = am._format_field_assignments(reg_enum, 0x2, reg_enum.fields[1].mask)
    assert "0x" in text


def test_peek_instruction_and_asm_window_errors(monkeypatch):
    sim = RV32Sim()
    am = sim.assert_manager
    base = 0x20003000
    _init_mem(sim, base, 0x20)
    c_addi = (0x0 << 13) | (1 << 7) | (1 << 2) | 0x1
    sim._write_memory(base, c_addi.to_bytes(2, "little"))
    instr, size = am._peek_instruction(base)
    assert size == 2
    assert instr is not None

    def bad_read(_addr, _size):
        raise ValueError("bad")

    monkeypatch.setattr(sim, "_read_memory", bad_read)
    assert am._peek_instruction(base)[0] is None

    def read_half_then_fail(addr, size):
        if size == 2 and addr == base:
            return 0x0003
        raise ValueError("bad")

    monkeypatch.setattr(sim, "_read_memory", read_half_then_fail)
    assert am._peek_instruction(base)[0] is None

    sim = RV32Sim()
    am = sim.assert_manager
    base = 0x20004000
    _init_mem(sim, base, 0x20)
    sim._write_memory(base, c_addi.to_bytes(2, "little"))

    def fail_expand(_):
        raise ValueError("bad")

    monkeypatch.setattr(am, "expand_compressed", fail_expand)
    lines = am._format_asm_window(base, count=1)
    assert "c.?" in lines[0]

    sim = RV32Sim()
    am = sim.assert_manager
    base = 0x20005000
    _init_mem(sim, base, 0x20)

    def read_half_then_fail_4(addr, size):
        if size == 2:
            return 0x0003
        raise ValueError("bad")

    monkeypatch.setattr(sim, "_read_memory", read_half_then_fail_4)
    assert am._format_asm_window(base, count=1) == []


def test_format_asm_window_compressed_and_read_fail(monkeypatch):
    sim = RV32Sim()
    am = sim.assert_manager

    def bad_read(_addr, _size):
        raise ValueError("bad")

    monkeypatch.setattr(sim, "_read_memory", bad_read)
    assert am._format_asm_window(0x20010000, count=1) == []

    sim = RV32Sim()
    am = sim.assert_manager
    base = 0x20011000
    _init_mem(sim, base, 0x20)
    sim._write_memory(base, (0x0001).to_bytes(2, "little"))
    monkeypatch.setattr(am, "expand_compressed", lambda _raw: 0x00000013)
    lines = am._format_asm_window(base, count=1)
    assert any("addi" in line for line in lines)


def test_disasm_instruction_more_ops():
    sim = RV32Sim()
    am = sim.assert_manager
    assert "slti" in am._disasm_instruction(0, _encode_i_type(1, 1, 0x2, 2))
    assert "sltiu" in am._disasm_instruction(0, _encode_i_type(1, 1, 0x3, 2))
    assert "xori" in am._disasm_instruction(0, _encode_i_type(1, 1, 0x4, 2))
    assert "ori" in am._disasm_instruction(0, _encode_i_type(1, 1, 0x6, 2))
    assert "andi" in am._disasm_instruction(0, _encode_i_type(1, 1, 0x7, 2))
    assert "slli" in am._disasm_instruction(0, _encode_i_type(4, 1, 0x1, 2))
    assert "srli" in am._disasm_instruction(0, _encode_i_type(4, 1, 0x5, 2))
    assert "srai" in am._disasm_instruction(0, _encode_i_type((0x20 << 5) | 1, 1, 0x5, 2))
    assert "mul" in am._disasm_instruction(0, _encode_r_type(0x01, 2, 1, 0x0, 3))
    assert "mulh" in am._disasm_instruction(0, _encode_r_type(0x01, 2, 1, 0x1, 3))
    assert "mulhsu" in am._disasm_instruction(0, _encode_r_type(0x01, 2, 1, 0x2, 3))
    assert "mulhu" in am._disasm_instruction(0, _encode_r_type(0x01, 2, 1, 0x3, 3))
    assert "divu" in am._disasm_instruction(0, _encode_r_type(0x01, 2, 1, 0x5, 3))
    assert "rem" in am._disasm_instruction(0, _encode_r_type(0x01, 2, 1, 0x6, 3))
    assert "remu" in am._disasm_instruction(0, _encode_r_type(0x01, 2, 1, 0x7, 3))
    assert "ebreak" in am._disasm_instruction(0, 0x00100073)
    assert "unknown" in am._disasm_instruction(0, 0xFFFFFFFF)


def test_disasm_instruction_op_variants():
    sim = RV32Sim()
    am = sim.assert_manager
    assert "add" in am._disasm_instruction(0, _encode_r_type(0x00, 2, 1, 0x0, 3))
    assert "sub" in am._disasm_instruction(0, _encode_r_type(0x20, 2, 1, 0x0, 3))
    assert "sll" in am._disasm_instruction(0, _encode_r_type(0x00, 2, 1, 0x1, 3))
    assert "slt " in am._disasm_instruction(0, _encode_r_type(0x00, 2, 1, 0x2, 3))
    assert "sltu" in am._disasm_instruction(0, _encode_r_type(0x00, 2, 1, 0x3, 3))
    assert "xor" in am._disasm_instruction(0, _encode_r_type(0x00, 2, 1, 0x4, 3))
    assert "srl" in am._disasm_instruction(0, _encode_r_type(0x00, 2, 1, 0x5, 3))
    assert "or" in am._disasm_instruction(0, _encode_r_type(0x00, 2, 1, 0x6, 3))


def test_disasm_at_error(monkeypatch):
    sim = RV32Sim()
    am = sim.assert_manager
    base = 0x20005500
    _init_mem(sim, base, 0x20)
    instr = _encode_i_type(0, 1, 0x0, 2)
    sim._write_memory(base, instr.to_bytes(4, "little"))
    monkeypatch.setattr(
        assertion_manager.AssertionManager,
        "_disasm_instruction",
        lambda *_args: (_ for _ in ()).throw(ValueError("bad")),
    )
    assert am._disasm_at(base) is None


def test_print_decision_table_headerless(capsys):
    sim = RV32Sim()
    am = sim.assert_manager
    reg = _make_reg()
    groups = [
        {"header": "empty", "rows": []},
        {"rows": [{"value": 1, "pc": 0x100, "note": "n", "path": "p"}]},
    ]
    am._print_decision_table(reg, groups, 4)
    assert "[ASSERT] Decision" in capsys.readouterr().out


def test_print_decision_table_with_disasm(monkeypatch, capsys):
    sim = RV32Sim()
    am = sim.assert_manager
    monkeypatch.setattr(assertion_manager.AssertionManager, "_disasm_at", lambda *_args: "nop")
    groups = [{"rows": [{"value": 1, "pc": 0x100}]}]
    am._print_decision_table(_make_reg(), groups, 4)
    out = capsys.readouterr().out
    assert ": nop" in out


def test_prompt_read_write_error_paths(monkeypatch):
    sim = RV32Sim()
    am = sim.assert_manager
    reg = _make_reg()
    reg.reset_value = 5
    sim.assert_show_asm = True
    _init_mem(sim, 0x20006000, 0x20)
    sim.pc = 0x20006000
    sim.last_instr = _encode_i_type(0, 1, 0x2, 5, opcode=0x03)
    inputs = iter(["BAD=1", ""])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    read_entry = am._prompt_assert_read(reg.address, 4, reg)
    assert read_entry["value"] == 5

    inputs = iter(["mask=bad", "-"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    write_entry = am._prompt_assert_write(reg.address, 4, 0x5, reg)
    assert write_entry.get("ignore") is True


def test_prompt_read_write_eof(monkeypatch):
    sim = RV32Sim()
    am = sim.assert_manager
    monkeypatch.setattr("builtins.input", lambda _prompt: (_ for _ in ()).throw(EOFError()))
    with pytest.raises(HaltException):
        am._prompt_assert_read(0x0, 4, _make_reg())

    with pytest.raises(HaltException):
        am._prompt_assert_write(0x0, 4, 0x1, _make_reg())


def test_prompt_with_svd_block_and_asm(monkeypatch):
    sim = RV32Sim()
    am = sim.assert_manager
    sim.assert_show_asm = True
    _init_mem(sim, 0x20007000, 0x20)
    sim.pc = 0x20007000
    sim.last_instr = _encode_i_type(0, 1, 0x2, 5, opcode=0x03)
    sim.svd_index = SimpleNamespace(
        find_peripheral_block=lambda _addr: SimpleNamespace(name="BLOCK", start=0, end=0x100)
    )
    inputs = iter(["-"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    read_entry = am._prompt_assert_read(0x20007000, 4, None)
    assert read_entry.get("ignore") is True

    inputs = iter(["-"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    write_entry = am._prompt_assert_write(0x20007000, 4, 0x1, None)
    assert write_entry.get("ignore") is True


def test_decision_hints_non_load_and_compressed_error(monkeypatch):
    sim = RV32Sim()
    am = sim.assert_manager
    am.last_instr = _encode_r_type(0x00, 2, 1, 0x0, 3)
    hints, active_mask, groups = am._decision_hints_for_read(_make_reg(), 4)
    assert hints == []
    assert active_mask is None
    assert groups == []

    am.last_instr = 0x1
    monkeypatch.setattr(am, "expand_compressed", lambda _val: (_ for _ in ()).throw(ValueError("bad")))
    result = am._decision_hints_for_read(_make_reg(), 4)
    assert len(result) == 2


def test_decision_hints_full_value_branch():
    sim = RV32Sim()
    am = sim.assert_manager
    base = 0x20008000
    _init_mem(sim, base, 0x40)
    load = _encode_i_type(0, 1, 0x2, 5, opcode=0x03)
    beq = _encode_b_type(4, 0, 5, 0x0)
    sim._write_memory(base, load.to_bytes(4, "little"))
    sim._write_memory(base + 4, beq.to_bytes(4, "little"))
    sim.pc = base
    sim.last_instr = load
    hints, active_mask, groups = am._decision_hints_for_read(_make_reg(), 4)
    assert active_mask is None
    assert any("Active bits: any" in hint for hint in hints)
    assert groups


def test_decision_hints_loop_and_const_compare():
    sim = RV32Sim()
    am = sim.assert_manager
    base = 0x20009000
    _init_mem(sim, base, 0x80)
    load = _encode_i_type(0, 1, 0x2, 5, opcode=0x03)
    slli = _encode_i_type(8, 5, 0x1, 5)
    bne_back = _encode_b_type(-4 & 0x1FFF, 0, 5, 0x1)
    sim._write_memory(base, load.to_bytes(4, "little"))
    sim._write_memory(base + 4, slli.to_bytes(4, "little"))
    sim._write_memory(base + 8, bne_back.to_bytes(4, "little"))
    sim.pc = base
    sim.last_instr = load
    hints, _active, _groups = am._decision_hints_for_read(_make_reg(), 4)
    assert _groups
    assert any(
        row.get("note") == "always zero after mask"
        for group in _groups
        for row in group.get("rows", [])
    )
    assert any("Wait loop on this value" in hint for hint in hints)

    sim = RV32Sim()
    am = sim.assert_manager
    base = 0x2000A000
    _init_mem(sim, base, 0x80)
    load = _encode_i_type(0, 1, 0x2, 5, opcode=0x03)
    addi = _encode_i_type(3, 0, 0x0, 6)
    bne = _encode_b_type(-4 & 0x1FFF, 6, 5, 0x1)
    sim._write_memory(base, load.to_bytes(4, "little"))
    sim._write_memory(base + 4, addi.to_bytes(4, "little"))
    sim._write_memory(base + 8, bne.to_bytes(4, "little"))
    sim.pc = base
    sim.last_instr = load
    hints, _active, _groups = am._decision_hints_for_read(_make_reg(), 4)
    assert any("Try seq=" in hint for hint in hints)


def test_decision_hints_loop_hint_no_branch():
    sim = RV32Sim()
    am = sim.assert_manager
    base = 0x2000B000
    _init_mem(sim, base, 0x40)
    load = _encode_i_type(0, 1, 0x2, 5, opcode=0x03)
    jal_back = _encode_j_type(-4 & 0x1FFFFF, 0)
    sim._write_memory(base, load.to_bytes(4, "little"))
    sim._write_memory(base + 4, jal_back.to_bytes(4, "little"))
    sim.pc = base
    sim.last_instr = load
    hints, _active, _groups = am._decision_hints_for_read(_make_reg(), 4)
    assert any("Loop ahead" in hint for hint in hints)


def test_decision_hints_const_tracking():
    sim = RV32Sim()
    am = sim.assert_manager
    base = 0x2000C000
    _init_mem(sim, base, 0x80)
    load = _encode_i_type(0, 1, 0x2, 5, opcode=0x03)
    addi = _encode_i_type(1, 0, 0x0, 6)
    ori = _encode_i_type(5, 0, 0x6, 7)
    xori = _encode_i_type(6, 0, 0x4, 8)
    andi = _encode_i_type(3, 6, 0x7, 6)
    slli = _encode_i_type(1, 7, 0x1, 7)
    srai = _encode_i_type((0x20 << 5) | 1, 8, 0x5, 8)
    srli = _encode_i_type(1, 7, 0x5, 7)
    jal = _encode_j_type(4, 9)
    instrs = [load, addi, ori, xori, andi, slli, srai, srli, jal]
    for idx, instr in enumerate(instrs):
        sim._write_memory(base + idx * 4, instr.to_bytes(4, "little"))
    sim.pc = base
    sim.last_instr = load
    hints, _active, _groups = am._decision_hints_for_read(_make_reg(), 4)
    assert hints


def test_decision_hints_mask_mapping_and_shift_edges():
    sim = RV32Sim()
    am = sim.assert_manager
    base = 0x2000D000
    _init_mem(sim, base, 0x80)
    load = _encode_i_type(0, 1, 0x2, 5, opcode=0x03)
    slli = _encode_i_type(3, 5, 0x1, 6)
    andi = _encode_i_type(0xF, 6, 0x7, 7)
    srli = _encode_i_type(8, 7, 0x5, 7)
    sim._write_memory(base + 4, slli.to_bytes(4, "little"))
    sim._write_memory(base + 8, andi.to_bytes(4, "little"))
    sim._write_memory(base + 12, srli.to_bytes(4, "little"))
    sim.pc = base
    sim.last_instr = load
    hints, _active, _groups = am._decision_hints_for_read(_make_reg(), 4)
    assert any("Shift left by" in hint for hint in hints)
    assert any("Shift right by" in hint for hint in hints)
    assert any("Mask 0x" in hint for hint in hints)


def test_decision_hints_branch_detail_and_op_mix():
    sim = RV32Sim()
    am = sim.assert_manager
    base = 0x2000E000
    _init_mem(sim, base, 0x80)
    load = _encode_i_type(0, 1, 0x2, 5, opcode=0x03)
    blt = _encode_b_type(4, 0, 5, 0x4)
    add = _encode_r_type(0x00, 1, 5, 0x0, 6)
    sim._write_memory(base + 4, blt.to_bytes(4, "little"))
    sim._write_memory(base + 8, add.to_bytes(4, "little"))
    sim.pc = base
    sim.last_instr = load
    hints, _active, _groups = am._decision_hints_for_read(_make_reg(), 4)
    assert any("negative ->" in hint for hint in hints)
    assert any("Value mixed with register operand" in hint for hint in hints)


def test_decision_hints_branch_detail_variants():
    sim = RV32Sim()
    am = sim.assert_manager
    base = 0x2000E100
    _init_mem(sim, base, 0x40)
    load = _encode_i_type(0, 1, 0x2, 5, opcode=0x03)
    bge = _encode_b_type(4, 0, 5, 0x5)
    sim._write_memory(base + 4, bge.to_bytes(4, "little"))
    sim.pc = base
    sim.last_instr = load
    hints, _active, _groups = am._decision_hints_for_read(_make_reg(), 4)
    assert any("nonnegative ->" in hint for hint in hints)

    sim = RV32Sim()
    am = sim.assert_manager
    base = 0x2000E200
    _init_mem(sim, base, 0x40)
    load = _encode_i_type(0, 1, 0x2, 5, opcode=0x03)
    br = _encode_b_type(4, 0, 5, 0x2)
    sim._write_memory(base + 4, br.to_bytes(4, "little"))
    sim.pc = base
    sim.last_instr = load
    hints, _active, _groups = am._decision_hints_for_read(_make_reg(), 4)
    assert any("true ->" in hint for hint in hints)


def test_decision_hints_const_tracking_paths():
    sim = RV32Sim()
    am = sim.assert_manager
    base = 0x2000F000
    _init_mem(sim, base, 0x100)
    load = _encode_i_type(0, 1, 0x2, 5, opcode=0x03)
    auipc = _encode_u_type(0x1000, 6, 0x17)
    ori_const = _encode_i_type(0x1, 6, 0x6, 7)
    ori_pop = _encode_i_type(0x2, 9, 0x6, 8)
    xori_const = _encode_i_type(0x3, 6, 0x4, 10)
    xori_pop = _encode_i_type(0x4, 12, 0x4, 11)
    addi_pop = _encode_i_type(0x5, 14, 0x0, 13)
    load_pop = _encode_i_type(0, 0, 0x2, 15, opcode=0x03)
    op_pop = _encode_r_type(0x00, 2, 1, 0x0, 16)
    instrs = [auipc, ori_const, ori_pop, xori_const, xori_pop, addi_pop, load_pop, op_pop]
    for idx, instr in enumerate(instrs):
        sim._write_memory(base + 4 + idx * 4, instr.to_bytes(4, "little"))
    sim.pc = base
    sim.last_instr = load
    hints, _active, _groups = am._decision_hints_for_read(_make_reg(), 4)
    assert any("Loaded into x5" in hint for hint in hints)


def test_decision_hints_compressed_load(monkeypatch):
    sim = RV32Sim()
    am = sim.assert_manager
    base = 0x20010000
    _init_mem(sim, base, 0x20)
    load = _encode_i_type(0, 1, 0x2, 5, opcode=0x03)
    monkeypatch.setattr(am, "expand_compressed", lambda _instr: load)
    sim.pc = base
    sim.last_instr = 0x0001
    hints, _active, _groups = am._decision_hints_for_read(_make_reg(), 4)
    assert any("Loaded into x5" in hint for hint in hints)


def test_assert_entry_helpers_and_inactive_access():
    sim = RV32Sim()
    am = sim.assert_manager
    reg = _make_reg()
    base, entry = am._ensure_assert_entry(reg.address + 2, 2, reg)
    assert base == reg.address
    assert entry["register"] == reg.path
    assert entry["peripheral"] == reg.peripheral
    assert entry["width"] == reg.width_bytes

    sim.assertions[base] = {"read": {"value": 1}}
    _, entry = am._ensure_assert_entry(base, 4, None)
    assert entry["width"] == 4
    am._fill_assert_meta(entry, reg)
    assert entry["register"] == reg.path
    assert entry["peripheral"] == reg.peripheral
    assert entry["width"] == reg.width_bytes

    assert am._assert_read(0x40000000, 4) == (None, None)
    assert am._assert_write(0x40000000, 4, 0x0) is None


def test_assert_missing_read_write_and_expected_default(monkeypatch):
    sim = RV32Sim()
    am = sim.assert_manager
    sim.assert_mode = True
    base = 0x40000000
    sim.assertions[base] = {"width": 4}
    with pytest.raises(HaltException):
        am._assert_read(base, 4)

    sim.assertions[base] = {"width": 4}
    with pytest.raises(HaltException):
        am._assert_write(base, 4, 0x1)

    sim.assert_assist_mode = True
    sim.assertions = {}
    monkeypatch.setattr(
        assertion_manager.AssertionManager, "_prompt_assert_write", lambda *_args: {"ignore": True}
    )
    assert am._assert_write(base, 4, 0x2) == base

    sim.assert_assist_mode = False
    sim.assertions = {base: {"width": 4, "write": {"mask": 0x0}}}
    assert am._assert_write(base, 4, 0x3) == base


def test_parse_sequence_input_empty_after_tokens():
    sim = RV32Sim()
    am = sim.assert_manager
    with pytest.raises(ValueError):
        am._parse_sequence_input("seq=, ")


def test_assert_write_missing_entry_raises():
    sim = RV32Sim()
    am = sim.assert_manager
    sim.assert_mode = True
    with pytest.raises(HaltException):
        am._assert_write(0x40000000, 4, 0x1)

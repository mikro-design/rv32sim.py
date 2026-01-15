import io
import sys

import pytest

from rv32sim import RV32Sim, HaltException, TrapException


def encode_r_type(funct7, rs2, rs1, funct3, rd, opcode=0x33):
    return (
        ((funct7 & 0x7f) << 25)
        | ((rs2 & 0x1f) << 20)
        | ((rs1 & 0x1f) << 15)
        | ((funct3 & 0x7) << 12)
        | ((rd & 0x1f) << 7)
        | (opcode & 0x7f)
    )


def encode_i_type(imm, rs1, funct3, rd, opcode=0x13):
    imm &= 0xfff
    return (
        (imm << 20)
        | ((rs1 & 0x1f) << 15)
        | ((funct3 & 0x7) << 12)
        | ((rd & 0x1f) << 7)
        | (opcode & 0x7f)
    )


def encode_s_type(imm, rs2, rs1, funct3, opcode=0x23):
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


def encode_b_type(imm, rs2, rs1, funct3, opcode=0x63):
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


def encode_u_type(imm, rd, opcode=0x37):
    return ((imm & 0xfffff000) | ((rd & 0x1f) << 7) | (opcode & 0x7f))


def encode_j_type(imm, rd, opcode=0x6f):
    imm &= 0x1fffff
    return (
        ((imm >> 20) & 0x1) << 31
        | ((imm >> 12) & 0xff) << 12
        | ((imm >> 11) & 0x1) << 20
        | ((imm >> 1) & 0x3ff) << 21
        | ((rd & 0x1f) << 7)
        | (opcode & 0x7f)
    )


def run_single(sim, instr, pc=0):
    sim.store_word(pc, instr)
    sim.pc = pc
    sim.execute()


def test_r_type_basic_ops():
    sim = RV32Sim()
    sim.regs[1] = 5
    sim.regs[2] = 7
    run_single(sim, encode_r_type(0x00, 2, 1, 0x0, 3))  # ADD
    assert sim.regs[3] == 12

    sim = RV32Sim()
    sim.regs[1] = 5
    sim.regs[2] = 7
    run_single(sim, encode_r_type(0x20, 2, 1, 0x0, 3))  # SUB
    assert sim.regs[3] == 0xfffffffe

    sim = RV32Sim()
    sim.regs[1] = 0xffffffff
    sim.regs[2] = 1
    run_single(sim, encode_r_type(0x00, 2, 1, 0x2, 3))  # SLT
    assert sim.regs[3] == 1

    sim = RV32Sim()
    sim.regs[1] = 0xffffffff
    sim.regs[2] = 1
    run_single(sim, encode_r_type(0x00, 2, 1, 0x3, 3))  # SLTU
    assert sim.regs[3] == 0


def test_r_type_shifts_and_logic():
    sim = RV32Sim()
    sim.regs[1] = 0x80000000
    sim.regs[2] = 1
    run_single(sim, encode_r_type(0x00, 2, 1, 0x5, 3))  # SRL
    assert sim.regs[3] == 0x40000000

    sim = RV32Sim()
    sim.regs[1] = 0x80000000
    sim.regs[2] = 1
    run_single(sim, encode_r_type(0x20, 2, 1, 0x5, 3))  # SRA
    assert sim.regs[3] == 0xc0000000

    sim = RV32Sim()
    sim.regs[1] = 0x1
    sim.regs[2] = 4
    run_single(sim, encode_r_type(0x00, 2, 1, 0x1, 3))  # SLL
    assert sim.regs[3] == 0x10

    sim = RV32Sim()
    sim.regs[1] = 0xf0f0
    sim.regs[2] = 0x0ff0
    run_single(sim, encode_r_type(0x00, 2, 1, 0x4, 3))  # XOR
    assert sim.regs[3] == 0xff00

    sim = RV32Sim()
    sim.regs[1] = 0xf0f0
    sim.regs[2] = 0x0ff0
    run_single(sim, encode_r_type(0x00, 2, 1, 0x6, 3))  # OR
    assert sim.regs[3] == 0xfff0

    sim = RV32Sim()
    sim.regs[1] = 0xf0f0
    sim.regs[2] = 0x0ff0
    run_single(sim, encode_r_type(0x00, 2, 1, 0x7, 3))  # AND
    assert sim.regs[3] == 0x00f0


def test_m_extension_div_rem_edges():
    sim = RV32Sim()
    sim.regs[1] = 0x80000000
    sim.regs[2] = 0xffffffff
    run_single(sim, encode_r_type(0x01, 2, 1, 0x4, 3))  # DIV
    assert sim.regs[3] == 0x80000000

    sim = RV32Sim()
    sim.regs[1] = 123
    sim.regs[2] = 0
    run_single(sim, encode_r_type(0x01, 2, 1, 0x4, 3))  # DIV by zero
    assert sim.regs[3] == 0xffffffff

    sim = RV32Sim()
    sim.regs[1] = 0x80000000
    sim.regs[2] = 0xffffffff
    run_single(sim, encode_r_type(0x01, 2, 1, 0x6, 3))  # REM
    assert sim.regs[3] == 0

    sim = RV32Sim()
    sim.regs[1] = 0x12345678
    sim.regs[2] = 0
    run_single(sim, encode_r_type(0x01, 2, 1, 0x6, 3))  # REM by zero
    assert sim.regs[3] == 0x12345678


def test_i_type_ops():
    sim = RV32Sim()
    sim.regs[1] = 5
    run_single(sim, encode_i_type(7, 1, 0x0, 3))  # ADDI
    assert sim.regs[3] == 12

    sim = RV32Sim()
    sim.regs[1] = 0xf0f0
    run_single(sim, encode_i_type(0x0ff, 1, 0x4, 3))  # XORI
    assert sim.regs[3] == 0xf00f

    sim = RV32Sim()
    sim.regs[1] = 0xf0f0
    run_single(sim, encode_i_type(0x0ff, 1, 0x6, 3))  # ORI
    assert sim.regs[3] == 0xf0ff

    sim = RV32Sim()
    sim.regs[1] = 0xf0f0
    run_single(sim, encode_i_type(0x0ff, 1, 0x7, 3))  # ANDI
    assert sim.regs[3] == 0x00f0

    sim = RV32Sim()
    sim.regs[1] = 0xffffffff
    run_single(sim, encode_i_type(1, 1, 0x2, 3))  # SLTI
    assert sim.regs[3] == 1

    sim = RV32Sim()
    sim.regs[1] = 0xffffffff
    run_single(sim, encode_i_type(1, 1, 0x3, 3))  # SLTIU
    assert sim.regs[3] == 0

    sim = RV32Sim()
    sim.regs[1] = 1
    run_single(sim, encode_i_type(4, 1, 0x1, 3))  # SLLI shamt=4
    assert sim.regs[3] == 0x10

    sim = RV32Sim()
    sim.regs[1] = 0x80000000
    run_single(sim, encode_i_type(1, 1, 0x5, 3))  # SRLI shamt=1
    assert sim.regs[3] == 0x40000000

    sim = RV32Sim()
    sim.regs[1] = 0x80000000
    run_single(sim, encode_i_type((0x20 << 5) | 1, 1, 0x5, 3))  # SRAI shamt=1
    assert sim.regs[3] == 0xc0000000


def test_load_store_and_sign_ext():
    sim = RV32Sim()
    base = 0x20000000
    sim.store_word(base, 0x80ff00aa)
    sim.regs[1] = base
    run_single(sim, encode_i_type(0, 1, 0x2, 2, opcode=0x03))  # LW x2, 0(base)
    assert sim.regs[2] == 0x80ff00aa

    sim = RV32Sim()
    base = 0x20000000
    sim.store_word(base, 0x80ff00aa)
    sim.regs[1] = base
    run_single(sim, encode_i_type(0, 1, 0x0, 2, opcode=0x03))  # LB
    assert sim.regs[2] == 0xffffffaa

    sim = RV32Sim()
    base = 0x20000000
    sim.store_word(base, 0x80ff00aa)
    sim.regs[1] = base + 2
    run_single(sim, encode_i_type(0, 1, 0x1, 2, opcode=0x03))  # LH
    assert sim.regs[2] == 0xffff80ff

    sim = RV32Sim()
    base = 0x20000000
    sim.store_word(base, 0x80ff00aa)
    sim.regs[1] = base
    run_single(sim, encode_i_type(0, 1, 0x4, 2, opcode=0x03))  # LBU
    assert sim.regs[2] == 0xaa

    sim = RV32Sim()
    base = 0x20000000
    sim.regs[1] = base
    sim.regs[2] = 0x11223344
    run_single(sim, encode_s_type(4, 2, 1, 0x2))  # SW
    assert sim.load_word(base + 4) == 0x11223344


def test_branch_and_jump():
    sim = RV32Sim()
    sim.regs[1] = 5
    sim.regs[2] = 5
    run_single(sim, encode_b_type(8, 2, 1, 0x0))  # BEQ +8
    assert sim.pc == 8

    sim = RV32Sim()
    sim.regs[1] = 5
    sim.regs[2] = 6
    run_single(sim, encode_b_type(8, 2, 1, 0x0))  # BEQ +8
    assert sim.pc == 4

    sim = RV32Sim()
    run_single(sim, encode_j_type(12, 1))  # JAL +12
    assert sim.regs[1] == 4
    assert sim.pc == 12

    sim = RV32Sim()
    sim.regs[1] = 0x100
    run_single(sim, encode_i_type(3, 1, 0x0, 5, opcode=0x67))  # JALR x5, 3(x1)
    assert sim.regs[5] == 4
    assert sim.pc == 0x102


def test_lui_auipc():
    sim = RV32Sim()
    run_single(sim, encode_u_type(0x12345000, 3))  # LUI
    assert sim.regs[3] == 0x12345000

    sim = RV32Sim()
    sim.pc = 0x100
    run_single(sim, encode_u_type(0x12345000, 3, opcode=0x17), pc=0x100)  # AUIPC
    assert sim.regs[3] == 0x12345100


def test_ecall_exit_and_ebreak():
    sim = RV32Sim()
    sim.regs[17] = 93
    sim.regs[10] = 7
    sim.store_word(0, 0x00000073)
    sim.pc = 0
    with pytest.raises(HaltException) as excinfo:
        sim.execute()
    assert excinfo.value.reason == "exit"
    assert excinfo.value.code == 7

    sim = RV32Sim()
    sim.store_word(0, 0x00100073)
    sim.pc = 0
    with pytest.raises(TrapException) as excinfo:
        sim.execute()
    assert excinfo.value.cause == 3


def test_compressed_expand_and_illegal():
    sim = RV32Sim()
    c_addi = (0x0 << 13) | (1 << 7) | (1 << 2) | 0x1
    expanded = sim.expand_compressed(c_addi)
    expected = encode_i_type(1, 1, 0x0, 1)
    assert expanded == expected

    sim = RV32Sim()
    c_addi4spn = 0x0  # quadrant 0, funct3 0 with imm=0 -> illegal
    with pytest.raises(TrapException):
        sim.expand_compressed(c_addi4spn)


def test_bitmanip_and_minmax_extensions():
    sim = RV32Sim()
    sim.regs[1] = 0b1111
    sim.regs[2] = 1
    run_single(sim, encode_r_type(0x24, 2, 1, 0x1, 3))  # BCLR
    assert sim.regs[3] == 0b1101

    sim = RV32Sim()
    sim.regs[1] = 0b1010
    sim.regs[2] = 3
    run_single(sim, encode_r_type(0x24, 2, 1, 0x5, 3))  # BEXT
    assert sim.regs[3] == 1

    sim = RV32Sim()
    sim.regs[1] = 0
    sim.regs[2] = 2
    run_single(sim, encode_r_type(0x14, 2, 1, 0x1, 3))  # BSET
    assert sim.regs[3] == 0b100

    sim = RV32Sim()
    sim.regs[1] = 0b1000
    sim.regs[2] = 3
    run_single(sim, encode_r_type(0x34, 2, 1, 0x1, 3))  # BINV
    assert sim.regs[3] == 0

    sim = RV32Sim()
    sim.regs[1] = 1
    sim.regs[2] = 2
    run_single(sim, encode_r_type(0x10, 2, 1, 0x2, 3))  # SH1ADD
    assert sim.regs[3] == 4

    sim = RV32Sim()
    sim.regs[1] = 1
    sim.regs[2] = 2
    run_single(sim, encode_r_type(0x10, 2, 1, 0x4, 3))  # SH2ADD
    assert sim.regs[3] == 6

    sim = RV32Sim()
    sim.regs[1] = 1
    sim.regs[2] = 2
    run_single(sim, encode_r_type(0x10, 2, 1, 0x6, 3))  # SH3ADD
    assert sim.regs[3] == 10

    sim = RV32Sim()
    sim.regs[1] = 1
    sim.regs[2] = 2
    run_single(sim, encode_r_type(0x05, 2, 1, 0x4, 3))  # MIN
    assert sim.regs[3] == 1

    sim = RV32Sim()
    sim.regs[1] = 1
    sim.regs[2] = 2
    run_single(sim, encode_r_type(0x05, 2, 1, 0x5, 3))  # MINU
    assert sim.regs[3] == 1

    sim = RV32Sim()
    sim.regs[1] = 1
    sim.regs[2] = 2
    run_single(sim, encode_r_type(0x05, 2, 1, 0x6, 3))  # MAX
    assert sim.regs[3] == 2

    sim = RV32Sim()
    sim.regs[1] = 1
    sim.regs[2] = 2
    run_single(sim, encode_r_type(0x05, 2, 1, 0x7, 3))  # MAXU
    assert sim.regs[3] == 2


def test_zbb_immediates_and_mul_variants():
    sim = RV32Sim()
    sim.regs[1] = 0x0000ff00
    run_single(sim, encode_i_type((0x30 << 5) | 0x04, 1, 0x1, 2))  # SEXT.B
    assert sim.regs[2] == 0x00000000

    sim = RV32Sim()
    sim.regs[1] = 0x8001
    run_single(sim, encode_i_type((0x30 << 5) | 0x05, 1, 0x1, 2))  # SEXT.H
    assert sim.regs[2] == 0xffff8001

    sim = RV32Sim()
    sim.regs[1] = 0x00010000
    run_single(sim, encode_i_type((0x30 << 5) | 0x00, 1, 0x1, 2))  # CLZ
    assert sim.regs[2] == 15

    sim = RV32Sim()
    sim.regs[1] = 0x00010000
    run_single(sim, encode_i_type((0x30 << 5) | 0x01, 1, 0x1, 2))  # CTZ
    assert sim.regs[2] == 16

    sim = RV32Sim()
    sim.regs[1] = 0b1011
    run_single(sim, encode_i_type((0x30 << 5) | 0x02, 1, 0x1, 2))  # CPOP
    assert sim.regs[2] == 3

    sim = RV32Sim()
    sim.regs[1] = 0x12345678
    sim.regs[2] = 0x1000
    run_single(sim, encode_r_type(0x01, 2, 1, 0x1, 3))  # MULH
    assert sim.regs[3] == ((sim._s32(sim.regs[1]) * sim._s32(sim.regs[2])) >> 32) & 0xffffffff

    sim = RV32Sim()
    sim.regs[1] = 0x12345678
    sim.regs[2] = 0x1000
    run_single(sim, encode_r_type(0x01, 2, 1, 0x2, 3))  # MULHSU
    assert sim.regs[3] == ((sim._s32(sim.regs[1]) * (sim.regs[2] & 0xffffffff)) >> 32) & 0xffffffff

    sim = RV32Sim()
    sim.regs[1] = 0x12345678
    sim.regs[2] = 0x1000
    run_single(sim, encode_r_type(0x01, 2, 1, 0x3, 3))  # MULHU
    assert sim.regs[3] == ((sim.regs[1] * sim.regs[2]) >> 32) & 0xffffffff

    sim = RV32Sim()
    sim.regs[1] = 10
    sim.regs[2] = 3
    run_single(sim, encode_r_type(0x01, 2, 1, 0x5, 3))  # DIVU
    assert sim.regs[3] == 3

    sim = RV32Sim()
    sim.regs[1] = 10
    sim.regs[2] = 3
    run_single(sim, encode_r_type(0x01, 2, 1, 0x7, 3))  # REMU
    assert sim.regs[3] == 1


def test_zext_h_and_bexti():
    sim = RV32Sim()
    sim.regs[1] = 0x1234abcd
    sim.regs[2] = 0
    run_single(sim, encode_r_type(0x04, 0, 1, 0x4, 3))  # ZEXT.H
    assert sim.regs[3] == 0xabcd

    sim = RV32Sim()
    sim.regs[1] = 0b1000
    run_single(sim, encode_i_type((0x24 << 5) | 3, 1, 0x5, 2))  # BEXTI
    assert sim.regs[2] == 1


def test_system_and_csr_instructions():
    sim = RV32Sim()
    sim.regs[1] = 0x10
    instr = encode_i_type(0x300, 1, 0x1, 2, opcode=0x73)  # CSRRW
    run_single(sim, instr)
    assert sim.regs[2] == 0x1800
    assert sim.csrs[0x300] & 0xff == 0x10
    assert sim.csrs[0x300] & 0x1800 == 0x1800

    sim = RV32Sim()
    sim.regs[1] = 0x1
    instr = encode_i_type(0x300, 1, 0x2, 2, opcode=0x73)  # CSRRS
    run_single(sim, instr)
    assert sim.regs[2] == 0x1800
    assert sim.csrs[0x300] & 0x1

    sim = RV32Sim()
    sim.regs[1] = 0x1
    instr = encode_i_type(0x300, 1, 0x3, 2, opcode=0x73)  # CSRRC
    run_single(sim, instr)
    assert sim.regs[2] == 0x1800

    sim = RV32Sim()
    instr = encode_i_type(0x300, 5, 0x5, 2, opcode=0x73)  # CSRRWI
    run_single(sim, instr)
    assert sim.csrs[0x300] & 0x1f == 5

    sim = RV32Sim()
    instr = encode_i_type(0x300, 1, 0x6, 2, opcode=0x73)  # CSRRSI
    run_single(sim, instr)
    assert sim.csrs[0x300] & 0x1

    sim = RV32Sim()
    instr = encode_i_type(0x300, 1, 0x7, 2, opcode=0x73)  # CSRRCI
    run_single(sim, instr)
    assert sim.csrs[0x300] & 0x1 == 0


def test_ret_and_fence_paths():
    sim = RV32Sim()
    sim.csrs[0x341] = 0x200
    sim.csrs[0x300] = 0x1800
    sim.store_word(0, 0x30200073)  # mret
    sim.pc = 0
    sim.execute()
    assert sim.pc == 0x200

    sim = RV32Sim()
    sim.csrs[0x141] = 0x300
    sim.csrs[0x300] = 0x100
    sim.store_word(0, 0x10200073)  # sret
    sim.pc = 0
    sim.execute()
    assert sim.pc == 0x300

    sim = RV32Sim()
    sim.csrs[0x041] = 0x400
    sim.store_word(0, 0x00200073)  # uret
    sim.pc = 0
    sim.execute()
    assert sim.pc == 0x400

    sim = RV32Sim()
    sim.store_word(0, 0x0000000f)  # fence
    sim.pc = 0
    sim.execute()
    assert sim.pc == 4


def test_compressed_variants():
    sim = RV32Sim()
    c_lw = (0x2 << 13) | 0x0  # C.LW with rd'=x8, rs1'=x8, imm=0
    lw = sim.expand_compressed(c_lw)
    assert (lw & 0x7f) == 0x03

    sim = RV32Sim()
    c_sw = (0x6 << 13) | 0x0  # C.SW
    sw = sim.expand_compressed(c_sw)
    assert (sw & 0x7f) == 0x23

    sim = RV32Sim()
    c_j = (0x5 << 13) | 0x1  # C.J
    jal = sim.expand_compressed(c_j)
    assert (jal & 0x7f) == 0x6F

    sim = RV32Sim()
    c_ebreak = (0x4 << 13) | (1 << 12) | (0x0 << 7) | (0x0 << 2) | 0x2
    assert sim.expand_compressed(c_ebreak) == 0x00100073


def test_cpu_core_helpers_and_fetch_misaligned():
    sim = RV32Sim()
    sim.cpu.sim = sim
    assert sim._div_trunc(10, 0) is None
    assert sim._rem_trunc(5, 0) == 5
    sim.memory_regions = []
    sim._memory_initialized = True
    sim._add_memory_region(0x20000000, 0x20000100, "ram")
    sim.store_byte(0x20000000, 0x12)
    sim.store_half(0x20000002, 0x3456)
    sim.pc = 0x1
    with pytest.raises(TrapException):
        sim.cpu.fetch()


def test_expand_compressed_illegal_cases():
    sim = RV32Sim()
    c_bad = (0x1 << 13) | 0x0
    with pytest.raises(TrapException):
        sim.expand_compressed(c_bad)

    c_addi16sp = (0x3 << 13) | (2 << 7) | 0x1
    with pytest.raises(TrapException):
        sim.expand_compressed(c_addi16sp)

    c_lui_bad = (0x3 << 13) | 0x1
    with pytest.raises(TrapException):
        sim.expand_compressed(c_lui_bad)

    c_lui_zero = (0x3 << 13) | (1 << 7) | 0x1
    with pytest.raises(TrapException):
        sim.expand_compressed(c_lui_zero)

    c_srli_bad = (0x4 << 13) | (1 << 12) | 0x1
    with pytest.raises(TrapException):
        sim.expand_compressed(c_srli_bad)

    c_srai_bad = (0x4 << 13) | (1 << 12) | (1 << 10) | 0x1
    with pytest.raises(TrapException):
        sim.expand_compressed(c_srai_bad)

    c_sub_bad = (0x4 << 13) | (1 << 12) | (0x3 << 10) | 0x1
    with pytest.raises(TrapException):
        sim.expand_compressed(c_sub_bad)

    c_slli_bad = 0x2
    with pytest.raises(TrapException):
        sim.expand_compressed(c_slli_bad)

    c_slli_bad12 = (1 << 12) | (1 << 7) | 0x2
    with pytest.raises(TrapException):
        sim.expand_compressed(c_slli_bad12)

    c_lwsp_bad = (0x2 << 13) | 0x2
    with pytest.raises(TrapException):
        sim.expand_compressed(c_lwsp_bad)

    c_jr_bad = (0x4 << 13) | 0x2
    with pytest.raises(TrapException):
        sim.expand_compressed(c_jr_bad)

    c_fun_bad = (0x1 << 13) | 0x2
    with pytest.raises(TrapException):
        sim.expand_compressed(c_fun_bad)

    c_quadrant_bad = 0x3
    with pytest.raises(TrapException):
        sim.expand_compressed(c_quadrant_bad)


def test_execute_func_stub_path():
    sim = RV32Sim()
    sim.func_stubs[0x100] = 0x1234
    sim.regs[1] = 0x200
    sim.pc = 0x100
    sim.execute()
    assert sim.regs[10] == 0x1234
    assert sim.pc == 0x200


def test_execute_illegal_rtype_variants():
    def expect_trap(instr):
        sim = RV32Sim()
        sim.store_word(0, instr)
        sim.pc = 0
        with pytest.raises(TrapException):
            sim.execute()

    expect_trap(encode_r_type(0x24, 2, 1, 0x0, 3))
    expect_trap(encode_r_type(0x14, 2, 1, 0x0, 3))
    expect_trap(encode_r_type(0x34, 2, 1, 0x0, 3))
    expect_trap(encode_r_type(0x10, 2, 1, 0x0, 3))
    expect_trap(encode_r_type(0x05, 2, 1, 0x0, 3))
    expect_trap(encode_r_type(0x04, 1, 1, 0x4, 3))
    expect_trap(encode_r_type(0x20, 2, 1, 0x1, 3))
    expect_trap(encode_r_type(0x20, 2, 1, 0x2, 3))
    expect_trap(encode_r_type(0x20, 2, 1, 0x3, 3))
    expect_trap(encode_r_type(0x20, 2, 1, 0x4, 3))
    expect_trap(encode_r_type(0x10, 2, 1, 0x5, 3))
    expect_trap(encode_r_type(0x20, 2, 1, 0x6, 3))
    expect_trap(encode_r_type(0x20, 2, 1, 0x7, 3))
    expect_trap(encode_r_type(0x08, 2, 1, 0x0, 3))


def test_execute_illegal_itype_and_misc_variants():
    sim = RV32Sim()
    sim.regs[1] = 0b11
    run_single(sim, encode_i_type((0x24 << 5) | 1, 1, 0x1, 2))  # BCLRI
    assert sim.regs[2] == 0b01
    run_single(sim, encode_i_type((0x14 << 5) | 2, 1, 0x1, 2))  # BSETI
    assert sim.regs[2] & (1 << 2)
    run_single(sim, encode_i_type((0x34 << 5) | 1, 1, 0x1, 2))  # BINVI
    assert sim.regs[2] == 0b01

    def expect_trap(instr):
        sim = RV32Sim()
        sim.store_word(0, instr)
        sim.pc = 0
        with pytest.raises(TrapException):
            sim.execute()

    expect_trap(encode_i_type((0x30 << 5) | 0x03, 1, 0x1, 2))
    expect_trap(encode_i_type((0x10 << 5) | 1, 1, 0x5, 2))
    expect_trap(encode_i_type(0, 1, 0x3, 2, opcode=0x03))
    expect_trap(encode_s_type(0, 2, 1, 0x3))
    expect_trap(encode_i_type(0, 0, 0x2, 0, opcode=0x0F))
    expect_trap(encode_b_type(4, 2, 1, 0x2))
    expect_trap(encode_i_type(0x300, 1, 0x4, 2, opcode=0x73))
    expect_trap(0x00000000)


def test_ecall_write_and_unknown(monkeypatch):
    sim = RV32Sim()
    buf = 0x20000000
    sim.store_word(0, 0x00000073)
    sim._write_memory(buf, b"hi")
    sim.regs[17] = 64
    sim.regs[10] = 1
    sim.regs[11] = buf
    sim.regs[12] = 2

    class DummyOut:
        def __init__(self):
            self.buffer = io.BytesIO()

        def flush(self):
            pass

    dummy = DummyOut()
    monkeypatch.setattr(sys, "stdout", dummy)
    sim.pc = 0
    sim.execute()
    assert sim.regs[10] == 2

    sim = RV32Sim()
    sim.store_word(0, 0x00000073)
    sim.regs[17] = 64
    sim.regs[10] = 3
    sim.regs[11] = buf
    sim.regs[12] = 1
    sim.pc = 0
    sim.execute()
    assert sim.regs[10] == 0xFFFFFFFF


def test_ecall_unknown_causes():
    for priv, cause in [(3, 11), (1, 9), (0, 8)]:
        sim = RV32Sim()
        sim.store_word(0, 0x00000073)
        sim.regs[17] = 1
        sim.priv = priv
        sim.pc = 0
        with pytest.raises(TrapException) as excinfo:
            sim.execute()
        assert excinfo.value.cause == cause


def test_wfi_and_unknown_system():
    sim = RV32Sim()
    sim.store_word(0, 0x10500073)
    sim.pc = 0
    sim.execute()
    assert sim.pc == 4

    sim = RV32Sim()
    sim.store_word(0, 0x10400073)
    sim.pc = 0
    with pytest.raises(TrapException):
        sim.execute()

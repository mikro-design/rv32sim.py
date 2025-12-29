import unittest

from rv32sim import RV32Sim


def encode_r_type(funct7, rs2, rs1, funct3, rd, opcode=0x33):
    return (
        ((funct7 & 0x7f) << 25)
        | ((rs2 & 0x1f) << 20)
        | ((rs1 & 0x1f) << 15)
        | ((funct3 & 0x7) << 12)
        | ((rd & 0x1f) << 7)
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


class RV32SimTests(unittest.TestCase):
    def test_signed_div_truncates_toward_zero(self):
        sim = RV32Sim()
        sim.regs[1] = 0xfffffffe  # -2
        sim.regs[2] = 3
        instr = encode_r_type(0x01, 2, 1, 0x4, 3)  # DIV x3, x1, x2
        sim.store_word(0, instr)
        sim.pc = 0
        sim.execute()
        self.assertEqual(sim.regs[3], 0)

    def test_signed_rem_uses_trunc_div(self):
        sim = RV32Sim()
        sim.regs[1] = 0xfffffffe  # -2
        sim.regs[2] = 3
        instr = encode_r_type(0x01, 2, 1, 0x6, 3)  # REM x3, x1, x2
        sim.store_word(0, instr)
        sim.pc = 0
        sim.execute()
        self.assertEqual(sim.regs[3], 0xfffffffe)

    def test_sltu_is_unsigned(self):
        sim = RV32Sim()
        sim.regs[1] = 0xffffffff
        sim.regs[2] = 1
        instr = encode_r_type(0x00, 2, 1, 0x3, 3)  # SLTU x3, x1, x2
        sim.store_word(0, instr)
        sim.pc = 0
        sim.execute()
        self.assertEqual(sim.regs[3], 0)

    def test_srl_is_logical(self):
        sim = RV32Sim()
        sim.regs[1] = 0x80000000
        sim.regs[2] = 1
        instr = encode_r_type(0x00, 2, 1, 0x5, 3)  # SRL x3, x1, x2
        sim.store_word(0, instr)
        sim.pc = 0
        sim.execute()
        self.assertEqual(sim.regs[3], 0x40000000)

    def test_bltu_uses_unsigned_compare(self):
        sim = RV32Sim()
        sim.regs[1] = 0xffffffff
        sim.regs[2] = 1
        instr = encode_b_type(8, 2, 1, 0x6)  # BLTU x1, x2, +8
        sim.store_word(0, instr)
        sim.pc = 0
        sim.execute()
        self.assertEqual(sim.pc, 4)

    def test_misaligned_load_is_allowed(self):
        sim = RV32Sim()
        sim.store_word(0, 0x11223344)
        self.assertEqual(sim.load_word(2), 0x00001122)

    def test_sparse_memory_access(self):
        sim = RV32Sim()
        addr = 0x20000000
        sim.store_word(addr, 0x12345678)
        self.assertEqual(sim.load_word(addr), 0x12345678)


if __name__ == "__main__":
    unittest.main()

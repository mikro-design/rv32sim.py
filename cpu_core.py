import sys


class _DecodedInstruction:
    __slots__ = (
        "instr",
        "instr_size",
        "fetch_penalty",
        "opcode",
        "rd",
        "funct3",
        "rs1",
        "rs2",
        "funct7",
        "rs1_u",
        "rs2_u",
        "rs1_s",
        "rs2_s",
    )

    def __init__(
        self,
        instr,
        instr_size,
        fetch_penalty,
        opcode,
        rd,
        funct3,
        rs1,
        rs2,
        funct7,
        rs1_u,
        rs2_u,
        rs1_s,
        rs2_s,
    ):
        self.instr = instr
        self.instr_size = instr_size
        self.fetch_penalty = fetch_penalty
        self.opcode = opcode
        self.rd = rd
        self.funct3 = funct3
        self.rs1 = rs1
        self.rs2 = rs2
        self.funct7 = funct7
        self.rs1_u = rs1_u
        self.rs2_u = rs2_u
        self.rs1_s = rs1_s
        self.rs2_s = rs2_s


class CPUCore:
    _OPCODE_HANDLERS = {
        0b0110011: "_handle_r_type",
        0b0010011: "_handle_i_type",
        0b0000011: "_handle_load",
        0b0100011: "_handle_store",
        0b0001111: "_handle_fence",
        0b1100011: "_handle_branch",
        0b1101111: "_handle_jal",
        0b1100111: "_handle_jalr",
        0b0110111: "_handle_lui",
        0b0010111: "_handle_auipc",
        0b1110011: "_handle_system",
    }

    def __init__(self, sim, trap_exception_cls):
        object.__setattr__(self, "sim", sim)
        object.__setattr__(self, "trap_exception_cls", trap_exception_cls)

    def __getattr__(self, name):
        return getattr(self.sim, name)

    def __setattr__(self, name, value):
        if name in ("sim", "trap_exception_cls"):
            object.__setattr__(self, name, value)
        else:
            setattr(self.sim, name, value)

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

    def _load_memif(self, addr, size):
        penalty = self._memif_access(self.memif_data, addr, size, is_write=False)
        value = self._read_value(addr, size)
        return value, penalty

    def _store_memif(self, addr, size, value):
        penalty = self._memif_access(self.memif_data, addr, size, is_write=True)
        self._write_value(addr, size, value)
        return penalty

    def fetch(self):
        # Check if this is a compressed instruction (lowest 2 bits != 11)
        if self.pc & 0x1:
            raise self.trap_exception_cls(0, self.pc & 0xffffffff)
        first_half = self.load_half(self.pc)
        if (first_half & 0x3) != 0x3:
            fetch_penalty = self._memif_access(self.memif_code, self.pc, 2, is_write=False)
            return first_half, fetch_penalty  # 16-bit compressed instruction
        second_half = self.load_half(self.pc + 2)
        fetch_penalty = self._memif_access(self.memif_code, self.pc, 4, is_write=False)
        return first_half | (second_half << 16), fetch_penalty  # 32-bit instruction

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

    def _execute_func_stub(self):
        if self.pc not in self.func_stubs:
            return False
        ret = self.func_stubs[self.pc]
        self.last_instr = None
        self.instr_count += 1
        self.regs[10] = self._u32(ret)
        next_pc = self.regs[1] & 0xffffffff
        fetch_penalty = self._memif_access(self.memif_code, self.pc, 4, is_write=False)
        cycle_cost = self.timing["base_cycles"] + fetch_penalty
        self._commit_step(cycle_cost, next_pc)
        return True

    def _decode_instruction(self):
        instr, fetch_penalty = self.fetch()
        self.last_instr = instr
        is_compressed = (instr & 0x3) != 0x3
        if is_compressed:
            instr = self._u32(self.expand_compressed(instr))
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
        return _DecodedInstruction(
            instr,
            instr_size,
            fetch_penalty,
            opcode,
            rd,
            funct3,
            rs1,
            rs2,
            funct7,
            rs1_u,
            rs2_u,
            rs1_s,
            rs2_s,
        )

    def _commit_step(self, cycle_cost, next_pc):
        mcountinhibit = self.csrs.get(0x320, 0)
        if not self.mcycle_suppress and not (mcountinhibit & 0x1):
            self.mcycle = (self.mcycle + cycle_cost) & 0xffffffffffffffff
        if not self.minstret_suppress and not (mcountinhibit & 0x4):
            self.minstret = (self.minstret + 1) & 0xffffffffffffffff
        self.mcycle_suppress = False
        self.minstret_suppress = False

        for i in range(32):
            self.regs[i] &= 0xffffffff
        self.regs[0] = 0
        self.pc = next_pc & 0xffffffff

    def _exec_r_type(self, rd, rs1_u, rs2_u, rs1_s, rs2_s, rs2, funct3, funct7):
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

    def _exec_i_type(self, instr, rd, rs1_u, rs1_s, funct3, funct7):
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

    def _exec_load(self, instr, rd, rs1_u, funct3):
        imm = self.sign_extend(instr >> 20, 12)
        addr = (rs1_u + imm) & 0xffffffff
        mem_penalty = 0
        if funct3 == 0x0:  # LB
            value, mem_penalty = self._load_memif(addr, 1)
            self.regs[rd] = self.sign_extend(value, 8)
        elif funct3 == 0x1:  # LH
            value, mem_penalty = self._load_memif(addr, 2)
            self.regs[rd] = self.sign_extend(value, 16)
        elif funct3 == 0x2:  # LW
            value, mem_penalty = self._load_memif(addr, 4)
            self.regs[rd] = value
        elif funct3 == 0x4:  # LBU
            value, mem_penalty = self._load_memif(addr, 1)
            self.regs[rd] = value
        elif funct3 == 0x5:  # LHU
            value, mem_penalty = self._load_memif(addr, 2)
            self.regs[rd] = value
        else:
            self._illegal_instruction()
        return self.timing["load_penalty"] + mem_penalty

    def _exec_store(self, rd, rs1_u, rs2_u, funct3, funct7):
        imm = self.sign_extend((funct7 << 5) | rd, 12)
        addr = (rs1_u + imm) & 0xffffffff
        mem_penalty = 0
        if funct3 == 0x0:  # SB
            mem_penalty = self._store_memif(addr, 1, rs2_u)
        elif funct3 == 0x1:  # SH
            mem_penalty = self._store_memif(addr, 2, rs2_u)
        elif funct3 == 0x2:  # SW
            mem_penalty = self._store_memif(addr, 4, rs2_u)
        else:
            self._illegal_instruction()
        return self.timing["store_penalty"] + mem_penalty

    def _exec_fence(self, funct3):
        if funct3 in (0x0, 0x1):
            return
        self._illegal_instruction()

    def _exec_branch(self, instr, funct3, rs1_u, rs2_u, rs1_s, rs2_s, next_pc):
        imm = self.sign_extend(
            ((instr >> 31) << 12)
            | ((instr >> 7 & 1) << 11)
            | ((instr >> 8 & 0xf) << 1)
            | ((instr >> 25 & 0x3f) << 5),
            13,
        )
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
        branch_target = (self.pc + imm) & 0xffffffff
        if take:
            next_pc = branch_target
        penalty = self._branch_penalty(self.pc, branch_target, take)
        return next_pc, penalty

    def _exec_jal(self, instr, rd, instr_size):
        imm = self.sign_extend(
            ((instr >> 31) << 20)
            | ((instr >> 12 & 0xff) << 12)
            | ((instr >> 20 & 1) << 11)
            | ((instr >> 21 & 0x3ff) << 1),
            21,
        )
        if rd:
            self.regs[rd] = self.pc + instr_size
        return self.pc + imm

    def _exec_jalr(self, instr, rd, rs1_u, instr_size):
        imm = self.sign_extend(instr >> 20, 12)
        if rd:
            self.regs[rd] = self.pc + instr_size
        return (rs1_u + imm) & ~1

    def _exec_system(self, instr, funct3, rs1, rs1_u, rd, next_pc):
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
                    raise self.trap_exception_cls(cause, 0)
            elif instr == 0x00100073:  # ebreak
                raise self.trap_exception_cls(3, 0)
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
        return next_pc

    def _dispatch_opcode(self, decoded, next_pc):
        handler_name = self._OPCODE_HANDLERS.get(decoded.opcode)
        if handler_name is None:
            self._illegal_instruction()
        handler = getattr(self, handler_name)
        return handler(decoded, next_pc)

    def _handle_r_type(self, decoded, next_pc):
        self._exec_r_type(
            decoded.rd,
            decoded.rs1_u,
            decoded.rs2_u,
            decoded.rs1_s,
            decoded.rs2_s,
            decoded.rs2,
            decoded.funct3,
            decoded.funct7,
        )
        return next_pc, 0

    def _handle_i_type(self, decoded, next_pc):
        self._exec_i_type(
            decoded.instr,
            decoded.rd,
            decoded.rs1_u,
            decoded.rs1_s,
            decoded.funct3,
            decoded.funct7,
        )
        return next_pc, 0

    def _handle_load(self, decoded, next_pc):
        penalty = self._exec_load(decoded.instr, decoded.rd, decoded.rs1_u, decoded.funct3)
        return next_pc, penalty

    def _handle_store(self, decoded, next_pc):
        penalty = self._exec_store(
            decoded.rd, decoded.rs1_u, decoded.rs2_u, decoded.funct3, decoded.funct7
        )
        return next_pc, penalty

    def _handle_fence(self, decoded, next_pc):
        self._exec_fence(decoded.funct3)
        return next_pc, 0

    def _handle_branch(self, decoded, next_pc):
        next_pc, penalty = self._exec_branch(
            decoded.instr,
            decoded.funct3,
            decoded.rs1_u,
            decoded.rs2_u,
            decoded.rs1_s,
            decoded.rs2_s,
            next_pc,
        )
        return next_pc, penalty

    def _handle_jal(self, decoded, _next_pc):
        return self._exec_jal(decoded.instr, decoded.rd, decoded.instr_size), 0

    def _handle_jalr(self, decoded, _next_pc):
        return self._exec_jalr(decoded.instr, decoded.rd, decoded.rs1_u, decoded.instr_size), 0

    def _handle_lui(self, decoded, next_pc):
        self.regs[decoded.rd] = decoded.instr & 0xfffff000
        return next_pc, 0

    def _handle_auipc(self, decoded, next_pc):
        self.regs[decoded.rd] = (self.pc + (decoded.instr & 0xfffff000)) & 0xffffffff
        return next_pc, 0

    def _handle_system(self, decoded, next_pc):
        next_pc = self._exec_system(
            decoded.instr, decoded.funct3, decoded.rs1, decoded.rs1_u, decoded.rd, next_pc
        )
        return next_pc, 0

    def execute(self):
        if self._execute_func_stub():
            return

        decoded = self._decode_instruction()

        self.instr_count += 1
        next_pc = self.pc + decoded.instr_size
        cycle_cost = self.timing["base_cycles"] + decoded.fetch_penalty
        next_pc, penalty = self._dispatch_opcode(decoded, next_pc)
        cycle_cost += penalty

        self._commit_step(cycle_cost, next_pc)

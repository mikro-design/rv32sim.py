.section .text.startup
.globl _start
/**
 * @brief Reset entry point.
 *
 * Initializes stack/gp, copies .data, clears .bss, calls main,
 * then issues an ecall exit (a7=93, a0=main return).
 *
 * @return Does not return; halts via ecall or loops forever.
 * @note Clobbers t0-t3, a7, and memory; uses RISC-V ABI.
 */
_start:
		la sp, __stack_top
		.option push
		.option norelax
		la gp, __global_pointer$
		.option pop

	la t0, __data_load_start
	la t1, __data_start
	la t2, __data_end
1:
	bgeu t1, t2, 2f
	lw t3, 0(t0)
	sw t3, 0(t1)
	addi t0, t0, 4
	addi t1, t1, 4
	j 1b

2:
	la t0, __bss_start
	la t1, __bss_end
3:
	bgeu t0, t1, 4f
	sw zero, 0(t0)
	addi t0, t0, 4
	j 3b

4:
	call main
	li a7, 93
	ecall
5:
	j 5b

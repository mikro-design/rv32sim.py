.section .text.start
.global _start
_start:
    /* Initialize stack pointer */
    la sp, _stack_top

    /* Jump to main */
    call main

    /* Exit with return value from main */
    mv a0, a0
    li a7, 93
    ecall

    /* Loop forever if exit fails */
    1: j 1b

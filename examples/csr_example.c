#include <stdint.h>

// Syscall wrapper for write (fd=1 for stdout)
static void sys_write(const void* buf, int count) {
    register int a0 asm("a0") = 1; // fd
    register const void* a1 asm("a1") = buf;
    register int a2 asm("a2") = count;
    register int a7 asm("a7") = 64; // sys_write
    
    asm volatile("ecall" : : "r"(a0), "r"(a1), "r"(a2), "r"(a7) : "memory");
}

static void my_putc(char c) {
    sys_write(&c, 1);
}

static void my_puts(const char *s) {
    while (*s) {
        my_putc(*s++);
    }
}

static void my_puthex(unsigned int val) {
    char buf[8];
    int i;
    my_puts("0x");
    for (i = 7; i >= 0; i--) {
        int nibble = (val >> (i * 4)) & 0xf;
        my_putc(nibble < 10 ? '0' + nibble : 'a' + nibble - 10);
    }
}

// CSR Access Macros
#define read_csr(reg) ({ unsigned long __tmp; \
  asm volatile ("csrr %0, " #reg : "=r"(__tmp)); \
  __tmp; })

#define write_csr(reg, val) ({ \
  asm volatile ("csrw " #reg ", %0" :: "rK"(val)); })

#define swap_csr(reg, val) ({ unsigned long __tmp; \
  asm volatile ("csrrw %0, " #reg ", %1" : "=r"(__tmp) : "rK"(val)); \
  __tmp; })

int main() {
    my_puts("CSR Access Test\n");

    // Test 1: Write and Read mscratch
    my_puts("Testing mscratch...\n");
    uint32_t pattern = 0xDEADBEEF;
    write_csr(mscratch, pattern);
    uint32_t readback = read_csr(mscratch);
    
    my_puts("Wrote: "); my_puthex(pattern); my_puts("\n");
    my_puts("Read:  "); my_puthex(readback); my_puts("\n");
    
    if (readback == pattern) {
        my_puts("mscratch TEST PASSED\n");
    } else {
        my_puts("mscratch TEST FAILED\n");
    }

    // Test 2: Read mcycle
    my_puts("Reading mcycle...\n");
    uint32_t c1 = read_csr(mcycle);
    // Burn some cycles
    for(volatile int i=0; i<100; i++);
    uint32_t c2 = read_csr(mcycle);
    
    my_puts("Cycle 1: "); my_puthex(c1); my_puts("\n");
    my_puts("Cycle 2: "); my_puthex(c2); my_puts("\n");
    
    if (c2 > c1) {
        my_puts("mcycle TEST PASSED\n");
    } else {
        my_puts("mcycle TEST FAILED\n");
    }

    // Test 3: CSRRW (Atomic Swap)
    my_puts("Testing CSRRW on mscratch...\n");
    uint32_t new_val = 0xCAFEBABE;
    uint32_t old_val = swap_csr(mscratch, new_val);
    uint32_t new_readback = read_csr(mscratch);
    
    my_puts("Old (swapped): "); my_puthex(old_val); my_puts("\n");
    my_puts("New (current): "); my_puthex(new_readback); my_puts("\n");

    if (old_val == pattern && new_readback == new_val) {
        my_puts("CSRRW TEST PASSED\n");
    } else {
        my_puts("CSRRW TEST FAILED\n");
    }

    return 0;
}

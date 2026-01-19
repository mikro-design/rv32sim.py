#include "coremark.h"
#include <stdarg.h>

// Syscall wrapper for write (fd=1 for stdout)
static void sys_write(const void* buf, int count) {
    register long a0 __asm__("a0") = 1;
    register long a1 __asm__("a1") = (long)buf;
    register long a2 __asm__("a2") = count;
    register long a7 __asm__("a7") = 64;

    __asm__ volatile("ecall" : "+r"(a0) : "r"(a1), "r"(a2), "r"(a7) : "memory");
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

static void my_putdec(unsigned int val) {
    char buf[16];
    int i = 0;
    if (val == 0) {
        my_putc('0');
        return;
    }
    while (val > 0) {
        buf[i++] = '0' + (val % 10);
        val /= 10;
    }
    while (i > 0) {
        my_putc(buf[--i]);
    }
}

static void my_putint(int val) {
    if (val < 0) {
        my_putc('-');
        my_putdec((unsigned int)-val);
    } else {
        my_putdec((unsigned int)val);
    }
}

static void my_putdec_width(unsigned long val, int width, int zero_pad) {
    char buf[24];
    int i = 0;
    if (val == 0) {
        buf[i++] = '0';
    } else {
        while (val > 0 && i < (int)sizeof(buf)) {
            buf[i++] = '0' + (val % 10);
            val /= 10;
        }
    }
    while (i < width && i < (int)sizeof(buf)) {
        buf[i++] = zero_pad ? '0' : ' ';
    }
    while (i > 0) {
        my_putc(buf[--i]);
    }
}

static void my_puthex_width(unsigned long val, int width, int zero_pad, int upper) {
    const char *digits = upper ? "0123456789ABCDEF" : "0123456789abcdef";
    char buf[24];
    int i = 0;
    if (val == 0) {
        buf[i++] = '0';
    } else {
        while (val > 0 && i < (int)sizeof(buf)) {
            buf[i++] = digits[val & 0xf];
            val >>= 4;
        }
    }
    while (i < width && i < (int)sizeof(buf)) {
        buf[i++] = zero_pad ? '0' : ' ';
    }
    while (i > 0) {
        my_putc(buf[--i]);
    }
}

int ee_printf(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);

    while (*fmt) {
        if (*fmt == '%') {
            int zero_pad = 0;
            int width = 0;
            int long_mod = 0;

            fmt++;
            if (*fmt == '0') {
                zero_pad = 1;
                fmt++;
            }
            while (*fmt >= '0' && *fmt <= '9') {
                width = width * 10 + (*fmt - '0');
                fmt++;
            }
            if (*fmt == 'l') {
                long_mod = 1;
                fmt++;
            }

            if (*fmt == 'd') {
                long v = long_mod ? va_arg(args, long) : va_arg(args, int);
                if (v < 0) {
                    my_putc('-');
                    my_putdec_width((unsigned long)(-v), width, zero_pad);
                } else {
                    my_putdec_width((unsigned long)v, width, zero_pad);
                }
            } else if (*fmt == 'u') {
                unsigned long v = long_mod ? va_arg(args, unsigned long) : va_arg(args, unsigned int);
                my_putdec_width(v, width, zero_pad);
            } else if (*fmt == 'x' || *fmt == 'X') {
                unsigned long v = long_mod ? va_arg(args, unsigned long) : va_arg(args, unsigned int);
                my_puthex_width(v, width, zero_pad, *fmt == 'X');
            } else if (*fmt == 's') {
                my_puts(va_arg(args, char *));
            } else if (*fmt == 'c') {
                my_putc((char)va_arg(args, int));
            } else if (*fmt == 'f') {
                my_puts("FLOAT");
                (void)va_arg(args, double);
            } else if (*fmt == '%') {
                my_putc('%');
            } else {
                my_putc(*fmt);
            }
        } else {
            my_putc(*fmt);
        }
        fmt++;
    }

    va_end(args);
    return 0;
}

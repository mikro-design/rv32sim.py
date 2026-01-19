#include <stdint.h>
#include <stdarg.h>

// Syscall wrapper
void sys_write(int fd, const void* buf, int count) {
    register int a0 asm("a0") = fd;
    register const void* a1 asm("a1") = buf;
    register int a2 asm("a2") = count;
    register int a7 asm("a7") = 64;
    
    asm volatile("ecall" : : "r"(a0), "r"(a1), "r"(a2), "r"(a7) : "memory");
}

// Minimal printf/puts
void my_puts(const char* s) {
    int len = 0;
    while (s[len]) len++;
    sys_write(1, s, len);
}

void my_puthex(uint32_t val) {
    char buf[10];
    buf[0] = '0'; buf[1] = 'x';
    for (int i = 0; i < 8; i++) {
        int nibble = (val >> (28 - i * 4)) & 0xF;
        buf[2+i] = (nibble < 10) ? ('0' + nibble) : ('a' + nibble - 10);
    }
    sys_write(1, buf, 10);
}

void my_putint(int val) {
    char buf[12];
    int idx = 0;
    if (val < 0) {
        sys_write(1, "-", 1);
        val = -val;
    }
    if (val == 0) {
        sys_write(1, "0", 1);
        return;
    }
    
    char tmp[12];
    int t_idx = 0;
    while (val > 0) {
        tmp[t_idx++] = '0' + (val % 10);
        val /= 10;
    }
    
    while (t_idx > 0) {
        buf[idx++] = tmp[--t_idx];
    }
    sys_write(1, buf, idx);
}

// Matrix params
#define N 16
int32_t A[N][N];
int32_t B[N][N];
int32_t C[N][N];

void init_matrix(int32_t M[N][N], int32_t seed_add) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            M[i][j] = (i + j + seed_add) & 0xFF;
        }
    }
}

int main() {
    my_puts("Starting Matrix Multiplication (16x16)...\n");
    
    init_matrix(A, 1);
    init_matrix(B, 2);
    
    // Compute C = A * B
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int32_t sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    
    my_puts("Computation done. Checking corner values:\n");
    my_puts("C[0][0] = "); my_putint(C[0][0]); my_puts("\n");
    my_puts("C[0][N-1] = "); my_putint(C[0][N-1]); my_puts("\n");
    my_puts("C[N-1][N-1] = "); my_putint(C[N-1][N-1]); my_puts("\n");
    
    // Verify a value (Sample verification)
    // A[0][k] = k+1
    // B[k][0] = k+2
    // C[0][0] = sum(k=0..15) (k+1)*(k+2) = sum(k^2 + 3k + 2)
    // sum(k^2) = 1240, 3*sum(k) = 3*120=360, sum(2)=32 -> 1240+360+32 = 1632
    
    if (C[0][0] == 1632) {
        my_puts("SUCCESS: C[0][0] matches expected value.\n");
    } else {
        my_puts("FAILURE: C[0][0] mismatch.\n");
    }
    
    return 0;
}

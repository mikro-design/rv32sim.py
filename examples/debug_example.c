#include <stdint.h>

// Simple delay function
void delay(int count) {
    volatile int i;
    for (i = 0; i < count; i++);
}

int main() {
    int a = 0;
    int b = 1;
    int c;
    
    while (1) {
        c = a + b;
        a = b;
        b = c;
        
        // Reset if it gets too big
        if (a > 1000) {
            a = 0;
            b = 1;
        }
        
        delay(10000);
    }
    return 0;
}

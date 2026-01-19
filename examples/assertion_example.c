#include <stdint.h>

#define UART_BASE 0x40000000
#define UART_DATA (*(volatile uint32_t *)(UART_BASE + 0x00))
#define UART_CTRL (*(volatile uint32_t *)(UART_BASE + 0x04))

int main() {
    // Read CTRL to check status (demonstrates 'Read value' prompt in assert-assist)
    // The user can inject values here like 'TX_RDY=1'
    volatile uint32_t status = UART_CTRL;
    
    // Write 'A' to DATA register
    // This matches the assertions.json example
    UART_DATA = 0x41;
    
    // Write to CTRL register
    UART_CTRL = 0x1;
    
    return 0;
}

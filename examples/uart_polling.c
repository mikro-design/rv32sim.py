#include <stdint.h>

#define UART_BASE 0x10000000

#define REG32(offset) (*(volatile uint32_t *)(UART_BASE + offset))

#define UART_TXDATA REG32(0x00)
#define UART_RXDATA REG32(0x04)
#define UART_TXCTRL REG32(0x08)
#define UART_RXCTRL REG32(0x0C)

#define TXCTRL_TXEN (1 << 0)
#define RXCTRL_RXEN (1 << 0)

int uart_tx_ready() {
    return UART_TXCTRL & TXCTRL_TXEN;
}

void uart_putc(char c) {
    // Wait for TX Ready
    while (!uart_tx_ready());
    UART_TXDATA = c;
}

void uart_puts(const char *s) {
    while (*s) {
        uart_putc(*s++);
    }
}

int uart_rx_ready() {
    return UART_RXCTRL & RXCTRL_RXEN;
}

char uart_getc() {
    // Wait for RX Ready
    while (!uart_rx_ready());
    return (char)(UART_RXDATA & 0xFF);
}

int main() {
    uart_puts("UART Polling Test\n");
    uart_puts("Waiting for characters... (Input 'q' to quit)\n");

    while (1) {
        if (uart_rx_ready()) {
            char c = uart_getc();
            uart_puts("Recv: ");
            uart_putc(c);
            uart_putc('\n');
            
            if (c == 'q') {
                uart_puts("Quit command received.\n");
                break;
            }
        }
    }
    
    return 0;
}

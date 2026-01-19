#include <stdint.h>

typedef struct {
    const char* name;
    char* buf;
    uint32_t size;
    volatile uint32_t wr;
    volatile uint32_t rd;
    uint32_t flags;
} RTT_Buffer;

typedef struct {
    char signature[16];
    int32_t num_up;
    int32_t num_down;
    RTT_Buffer up[1];
    RTT_Buffer down[1];
} RTT_ControlBlock;

#define RTT_SIZE 1024
static char up_buffer[RTT_SIZE];
static char down_buffer[RTT_SIZE];

// The control block must be aligned and easily found by the simulator
__attribute__((aligned(16)))
RTT_ControlBlock _SEGGER_RTT = {
    .signature = "SEGGER RTT",
    .num_up = 1,
    .num_down = 1,
    .up = {{
        .name = "Terminal",
        .buf = up_buffer,
        .size = RTT_SIZE,
        .wr = 0,
        .rd = 0,
        .flags = 0
    }},
    .down = {{
        .name = "Terminal",
        .buf = down_buffer,
        .size = RTT_SIZE,
        .wr = 0,
        .rd = 0,
        .flags = 0
    }}
};

void rtt_write_char(char c) {
    uint32_t wr = _SEGGER_RTT.up[0].wr;
    uint32_t next_wr = (wr + 1) % RTT_SIZE;
    
    // Check if full (simple blocking)
    while (next_wr == _SEGGER_RTT.up[0].rd);
    
    up_buffer[wr] = c;
    _SEGGER_RTT.up[0].wr = next_wr;
}

void rtt_write_string(const char* s) {
    while (*s) {
        rtt_write_char(*s++);
    }
}

int rtt_read_char() {
    uint32_t rd = _SEGGER_RTT.down[0].rd;
    if (rd == _SEGGER_RTT.down[0].wr) {
        return -1; // No data
    }
    
    char c = down_buffer[rd];
    _SEGGER_RTT.down[0].rd = (rd + 1) % RTT_SIZE;
    return (int)c;
}

void delay(int count) {
    volatile int i;
    for (i = 0; i < count; i++);
}

int main() {
    int counter = 0;
    rtt_write_string("RTT Initialized. Type something!\n");
    
    while (1) {
        int c = rtt_read_char();
        if (c != -1) {
            rtt_write_string("Echo: ");
            rtt_write_char((char)c);
            rtt_write_char('\n');
        }
        
        counter++;
        if (counter % 1000000 == 0) {
            rtt_write_string("Tick...\n");
        }
        
        // Don't burn CPU too fast in simulation
        // delay(100); 
    }
    return 0;
}

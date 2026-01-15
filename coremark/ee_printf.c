#include "coremark.h"

#include <stdarg.h>

int ee_printf(const char *fmt, ...)
{
	va_list args;

	(void)fmt;
	va_start(args, fmt);
	va_end(args);

	return 0;
}

#include "tokenizer.h"
#include "spmm.h"

void normalizer(const char* str)
{
	const u8* b_str = (const u8*)str;
	while (*b_str != '\0') {
		if (*b_str == ' ') {
			*b_str = '_';
		}
	}
}

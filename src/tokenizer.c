#include "tokenizer.h"
#include "spmm.h"

void print_prompt_bytes(const char* c)
{
	while (*c != '\0') {
		printf("%c : %u\n", *c, *((const u8*)c));
		++c;
	}
	printf("Hello%cthisis\n", (char)32);
}

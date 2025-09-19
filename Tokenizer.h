#ifndef TOKENIZER_H
#define TOKENIZER_H

#define MAX_INPUT_LEN 150
#define MAX_TOKENS 200
#define MAX_TOKEN_LEN 32
#define BATCH 1
#define SEQ_LEN 64
#define VOCAB 6000

#include <stdlib.h>
int utf8_char_len(char first_byte);
int preprocess(const char* input, char tokens[][MAX_TOKEN_LEN]);
int get_token_rank(const char* token);
int get_token_rank(const char* token);
void tokenize(const char* input, int64_t* output_ids, int* output_len) ;
void detokenize(const int64_t* token_ids, int num_tokens, char* output, int max_output_len);

int sample_next_token(const float logits[BATCH][SEQ_LEN][VOCAB],int num_tokens,int top_k,float temperature);
#endif // TOKENIZER_H
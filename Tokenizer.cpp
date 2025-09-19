#include <string.h>
#include <stdlib.h>
#include "sp_vocab.h"
#include "Tokenizer.h"
#include "esp32-hal-psram.h"
#include <stdlib.h>
#include <math.h>
#include <float.h> // for FLT_MAX
#include <time.h>   // for seeding rand() if available

#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>



// Helper function to get UTF-8 character length
int utf8_char_len(char first_byte) {
    if ((first_byte & 0x80) == 0) return 1;
    if ((first_byte & 0xE0) == 0xC0) return 2;
    if ((first_byte & 0xF0) == 0xE0) return 3;
    if ((first_byte & 0xF8) == 0xF0) return 4;
    return 1; // Invalid UTF-8, treat as single byte
}

// Preprocess input: replace spaces with "▁" and split into UTF-8 tokens
int preprocess(const char* input, char tokens[][MAX_TOKEN_LEN]) {
    char processed[MAX_INPUT_LEN * 3]; // Account for space replacement
    int proc_idx = 0;

    // Replace spaces with "▁"
    for (int i = 0; input[i] && proc_idx < sizeof(processed) - 3; i++) {
        if (input[i] == ' ') {
            processed[proc_idx++] = 0xE2;
            processed[proc_idx++] = 0x96;
            processed[proc_idx++] = 0x81;
        } else {
            processed[proc_idx++] = input[i];
        }
    }
    processed[proc_idx] = '\0';

    // Split into UTF-8 characters
    int token_count = 0;
    for (int i = 0; processed[i] && token_count < MAX_TOKENS;) {
        int len = utf8_char_len(processed[i]);
        if (token_count < MAX_TOKENS && len < MAX_TOKEN_LEN) {
            strncpy(tokens[token_count], &processed[i], len);
            tokens[token_count][len] = '\0';
            token_count++;
        }
        i += len;
    }
    return token_count;
}

// Get the rank (index) of a token from the vocabulary
int get_token_rank(const char* token) {
    for (int i = 0; i < VOCAB_SIZE; i++) {
        if (strcmp(token, vocab[i]) == 0) {
            return i;
        }
    }
    return -1;
}

// BPE Tokenization function
void tokenize(const char* input, int64_t* output_ids, int* output_len) {
    char tokens[MAX_TOKENS][MAX_TOKEN_LEN];
    int token_count = preprocess(input, tokens);

    int changed;
    do {
        changed = 0;
        int min_rank = VOCAB_SIZE;
        int merge_idx = -1;

        // Find the next merge candidate
        for (int i = 0; i < token_count - 1; i++) {
            char merged[MAX_TOKEN_LEN * 2];
            strcpy(merged, tokens[i]);
            strcat(merged, tokens[i + 1]);

            int rank = get_token_rank(merged);
            if (rank != -1 && rank < min_rank) {
                min_rank = rank;
                merge_idx = i;
            }
        }

        // Perform the merge if found
        if (merge_idx != -1) {
            strcat(tokens[merge_idx], tokens[merge_idx + 1]);
            for (int i = merge_idx + 1; i < token_count - 1; i++) {
                strcpy(tokens[i], tokens[i + 1]);
            }
            token_count--;
            changed = 1;
        }
    } while (changed);

    // Convert tokens to IDs
    for (int i = 0; i < token_count; i++) {
        int rank = get_token_rank(tokens[i]);
        output_ids[i] = (rank != -1) ? rank : 1; // Use <unk> for not found
    }
    //output_ids[token_count] = 0; // eos
    //token_count+=1;
    *output_len = token_count;
}

// Detokenization function
void detokenize(const int64_t* token_ids, int num_tokens, char* output, int max_output_len) {
    int out_idx = 0;
    output[0] = '\0';
    
    for (int i = 0; i < num_tokens && out_idx < max_output_len - 1; i++) {
        int id = token_ids[i];
        if (id < 0 || id >= VOCAB_SIZE) {
            continue; // Skip invalid IDs
        }
        
        const char* token = vocab[id];
        
        // Handle special tokens
        if (strcmp(token, "<pad>") == 0 || strcmp(token, "<s>") == 0 || 
            strcmp(token, "</s>") == 0) {
            continue; // Skip these tokens
        }
        
        if (strcmp(token, "<unk>") == 0) {
            // Handle unknown token
            if (out_idx + 5 < max_output_len) {
                strcat(output, "[UNK]");
                out_idx += 5;
            }
            continue;
        }
        
        // Check if token starts with "▁" (space symbol)
        if (strlen(token) >= 3 && 
            (unsigned char)token[0] == 0xE2 && 
            (unsigned char)token[1] == 0x96 && 
            (unsigned char)token[2] == 0x81) {
            // Add space before word (except at beginning)
            if (out_idx > 0 && out_idx + 1 < max_output_len) {
                output[out_idx++] = ' ';
                output[out_idx] = '\0';
            }
            
            // Add the rest of the token (after the space symbol)
            if (strlen(token) > 3) {
                const char* word_part = token + 3;
                int part_len = strlen(word_part);
                if (out_idx + part_len < max_output_len) {
                    strcat(output, word_part);
                    out_idx += part_len;
                }
            }
        } else {
            // Regular token, just append
            int token_len = strlen(token);
            if (out_idx + token_len < max_output_len) {
                strcat(output, token);
                out_idx += token_len;
            }
        }
    }
    
    // Ensure null termination
    output[out_idx] = '\0';
}




static inline void seed_rng_once() {
    static int seeded = 0;
    if (!seeded) {
        srand((unsigned)time(NULL));
        seeded = 1;
    }
}

// Greedy (argmax)
int _greedy_next_token(const float logits[BATCH][SEQ_LEN][VOCAB], int num_tokens) {
    if (num_tokens <= 0) num_tokens = 1;
    if (num_tokens > SEQ_LEN) num_tokens = SEQ_LEN;
    int last = num_tokens - 1;

    int max_idx = -1;
    float max_val = -FLT_MAX;
    for (int v = 0; v < VOCAB; v++) {
        float val = logits[0][last][v];
        if (val > max_val) { max_val = val; max_idx = v; }
    }
    return max_idx;
}

// Simple sampling: temperature + top-k
int sample_next_token(const float logits[BATCH][SEQ_LEN][VOCAB],
                      int num_tokens,
                      int top_k,
                      float temperature) {
    if (num_tokens <= 0) num_tokens = 1;
    if (num_tokens > SEQ_LEN) num_tokens = SEQ_LEN;
    int last = num_tokens - 1;

    if (temperature <= 0.0f) {
        return _greedy_next_token(logits, num_tokens);
    }

    // Step 1: find top-k logits (no big arrays, just O(k))
    int best_idx[top_k];
    float best_val[top_k];
    for (int i = 0; i < top_k; i++) { best_val[i] = -FLT_MAX; best_idx[i] = -1; }

    for (int v = 0; v < VOCAB; v++) {
        float val = logits[0][last][v] / temperature;
        // insert into top-k if better
        for (int j = 0; j < top_k; j++) {
            if (val > best_val[j]) {
                // shift down
                for (int k2 = top_k-1; k2 > j; k2--) {
                    best_val[k2] = best_val[k2-1];
                    best_idx[k2] = best_idx[k2-1];
                }
                best_val[j] = val;
                best_idx[j] = v;
                break;
            }
        }
    }

    // Step 2: softmax over just top-k
    float maxv = -FLT_MAX;
    for (int i = 0; i < top_k; i++) if (best_val[i] > maxv) maxv = best_val[i];

    float probs[top_k];
    double sum = 0.0;
    for (int i = 0; i < top_k; i++) {
        probs[i] = expf(best_val[i] - maxv);
        sum += probs[i];
    }
    for (int i = 0; i < top_k; i++) probs[i] /= (float)sum;

    // Step 3: sample
    seed_rng_once();
    double r = ((double)rand()+1) / ((double)RAND_MAX+1);
    double acc = 0.0;
    for (int i = 0; i < top_k; i++) {
        acc += probs[i];
        if (r <= acc) return best_idx[i];
    }
    return best_idx[top_k-1]; // fallback
}

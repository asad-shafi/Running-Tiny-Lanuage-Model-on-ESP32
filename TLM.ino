#include <stdio.h>
#include "esp32-hal-psram.h"
#include "dspm_mult.h" // For matrix multiplication
#include "Tokenizer.h"
#include "sp_vocab.h"
#include "Tiny.h"
#include <float.h>  // for FLT_MIN or DBL_MIN if using double
String inputString = "";      // a String to hold incoming data
bool stringComplete = false;  // whether the string is complete

int64_t token_ids[64];
int num_tokens = 0;

#define BATCH 1
#define SEQ_LEN 64
#define VOCAB 6000

// ---------- 1) Correct greedy (argmax of last timestep) ----------
int greedy_next_token(const float logits[BATCH][SEQ_LEN][VOCAB], int num_tokens) {
  // clamp num_tokens into [1..SEQ_LEN]
  if (num_tokens <= 0) num_tokens = 1;
  if (num_tokens > SEQ_LEN) num_tokens = SEQ_LEN;

  int last = num_tokens - 1; // index of last produced token
  int max_idx = -1;
  float max_val = -FLT_MAX;

  // find argmax over vocab for the last timestep of batch 0
  for (int v = 0; v < VOCAB; ++v) {
    float val = logits[0][last][v];
    if (val > max_val) {
      max_val = val;
      max_idx = v;
    }
  }

  return max_idx; // single token id to feed back
}


int token(const char* text) {
  //token(inputString.substring(0, inputString.length() - 1).c_str());
  // clear the string:
  tokenize(text, token_ids, &num_tokens);

  Serial.print("Tokens ");
  Serial.println(num_tokens);

  Serial.print("\n");
  // Optionally print token strings:
  for (int i = 0; i < num_tokens; ++i) {
    int id = token_ids[i];
    if (id >= 0 && id < VOCAB_SIZE) {
      printf("id -> %d  %s ", id, vocab[id]);

    }
  }
  Serial.println("---");
  return 0;
}


float (*logits)[64][6000];
void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  inputString.reserve(200);
          Serial.println("PSRAM is available and initialized.");
    Serial.printf("Free heap: %d\n", esp_get_free_heap_size());
    Serial.printf("Free PSRAM: %d\n", esp_get_free_internal_heap_size());
  logits = (float(*)[64][6000])ps_malloc(sizeof(float) * 64 * 6000);
}
  int64_t(*token_ids_2d)[64] = (int64_t(*)[64])token_ids;


void loop() {
  // put your main code here, to run repeatedly:
  if (stringComplete) {
    Serial.println(inputString);
    String tem = inputString.substring(0, inputString.length() - 1);
    for (int t = 0; t < 32; t++) { // context lenght
      token(tem.c_str());
      forward_pass(token_ids_2d, logits);
      Serial.println("token");

     int temp = sample_next_token(logits, num_tokens, 40, 0.7f);
      //int temp = greedy_next_token(logits,num_tokens);
      Serial.print("Max indx ");
      if(temp <= 0) temp =0;
      Serial.println(temp);
      const char* token = vocab[temp];
      Serial.println(token);
      tem += token;
 
      char output_text[MAX_INPUT_LEN];
      detokenize(token_ids, num_tokens, output_text, sizeof(output_text));
      printf("\nDetokenized: %s\n", output_text);
    }
    for(int i =0;i<64;i++)
    {
      token_ids[i]=0;
    }
    inputString = "";
    stringComplete = false;
  }
}

void serialEvent() {
  while (Serial.available()) {
    // get the new byte:
    char inChar = (char)Serial.read();
    // add it to the inputString:
    inputString += inChar;
    // if the incoming character is a newline, set a flag so the main loop can
    // do something about it:
    if (inChar == '\n') {
      stringComplete = true;
    }
  }
}

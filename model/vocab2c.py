# generate_vocab_header.py
import sys

def generate_header(vocab_file, header_file="sp_vocab.h"):
    vocab = []
    with open(vocab_file, "r", encoding="utf-8") as f:
        for line in f:
            token, *_ = line.strip().split("\t")
            vocab.append(token)

    with open(header_file, "w", encoding="utf-8") as f:
        f.write("// Auto-generated from SentencePiece .vocab file\n")
        f.write("#pragma once\n\n")
        f.write(f"#define VOCAB_SIZE {len(vocab)}\n\n")
        f.write("static const char* vocab[VOCAB_SIZE] = {\n")
        for token in vocab:
            # escape quotes and backslashes
            safe_token = token.replace("\\", "\\\\").replace('"', '\\"')
            f.write(f'    "{safe_token}",\n')
        f.write("};\n")

    print(f"Header written to {header_file}, vocab size = {len(vocab)}")

#if __name__ == "__main__":
   # if len(sys.argv) < 2:
    #    print("Usage: python generate_vocab_header.py sp_small.vocab")
    #else:
generate_header("/content/sp_tinystory.vocab") # Changed from .model to .vocab
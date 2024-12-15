"""
Build LLM from scratch
Dec 14 2024
Author: Vincent Yang
List 2.1 Reding in a short story as text sample into python
"""

import re
import tiktoken

class SimpleTokenizerV1:            # initial tokenizer, can't read unknown word
    def __init__(self, vocab):
        self.str_to_int = vocab #A
        self.int_to_str = {i:s for s,i in vocab.items()} #B can be used in decoding
    def encode(self, text): #C
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self, ids): #D
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text) #E
        return text


class SimpleTokenizerV2:    # replace unknown words with unk, and add endoftext
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}
    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int #A
            else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text) #B
        return text


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text=f.read()
print("Total number of character:", len(raw_text))
print(raw_text[:99])

"split the raw text into tokens (tokenization)"
preprocessed = re.split(r'([,.?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))
print(preprocessed[:30])

"Generate the vocabulary"
all_tokens = sorted(list(set(preprocessed))) # set can make sure that one word only appears once
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab_size = len(all_tokens)
print(vocab_size)

vocab = {token:integer for integer,token in enumerate(all_tokens)}
# for i, item in enumerate(vocab.items()):
#     print(item)
#     if i > 50:
#         break

tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))

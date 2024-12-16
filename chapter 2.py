"""
Build LLM from scratch
Dec 14 2024
Author: Vincent Yang
List 2.1 Reding in a short story as text sample into python
"""

import re
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

'''
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
text = '"It\'s the last he painted, you know," Mrs. Gisburn said with pardonable pride.'
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))
'''

'''
2.5 Use of the byte pair encoding
*** GPT2 tokenizer will view the space as one character(220)
Result for the following text:
[220, 220, 18435, 11, 466, 220, 220, 220, 345, 588, 8887, 30, 220, 50256, 554, 262, 4252, 18250, 8812, 2114, 286, 617, 34680, 27271, 220, 220, 220, 220, 220, 764]
Then this explain why it can keep space and treat someunkwonPlace as a single word during decoding.
'''
# tokenizer = tiktoken.get_encoding("gpt2")
#
# text = "   Hello, do    you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace      ."
# integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# print(integers)
#
# strings = tokenizer.decode(integers)
# print(strings)

'''
2.6 Data sampling with a sliding window
'''
'''
tokenizer = tiktoken.get_encoding("gpt2")
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

enc_sample = enc_text[50:]

context_size = 4                    # 为LLM后面的训练提供data pair
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x: {x}")
print(f"y:      {y}")

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired)
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt) #A
        for i in range(0, len(token_ids) - max_length, stride): #B
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    def __len__(self): #C
        return len(self.input_ids)
    def __getitem__(self, idx): #D
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4,
max_length=256, stride=128, shuffle=True, drop_last=True):
    tokenizer = tiktoken.get_encoding("gpt2") #A
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride) #B
    dataloader = DataLoader(
    dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last) #C
    return dataloader


dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader) #A     # create an iterator from the DataLoader
first_batch = next(data_iter)
print(first_batch)

# Based on the definition of GPTDatasetV1, we shift input one position rightwards, and we get target,
# since its task is to predict the next word. Max_length is the length of input window, and stride is
# how many position to shift for the input window

dataloader = create_dataloader_v1(raw_text, batch_size=3, max_length=4, stride=2)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)
'''

'''
2.7 Creating token embeddings (Just initialize, haven't updated yet)
'''
input_ids = torch.tensor([2, 3, 5, 1])
vocab_size = 6
output_dim = 3
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)       # get the overall weight for each token id from 0 to 5
print(embedding_layer(torch.tensor([3])))   # get specific weight for token 3
print(embedding_layer(input_ids))   # get the overall weight for input_ids
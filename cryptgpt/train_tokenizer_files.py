# coding: utf-8

from functools import partial
import glob
import os

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import CharDelimiterSplit
from tokenizers.processors import ByteLevel
from tokenizers.trainers import BpeTrainer

from .prepare_dataset import encrypt_, decrypt_, gpt2_tokenizer, num_proc
from .tokenizer_utils import CryptGPTTokenizer

len_unicode = 55215 # NOT 65536 because surrogates are not allowed in python, see https://jrgraphix.net/r/Unicode/

eos_token = gpt2_tokenizer.eos_token

print("preparing tokenizer")
tokenizer = Tokenizer(BPE(unk_token=eos_token))

pre_tokenizer = CharDelimiterSplit(encrypt_(" "))
tokenizer.pre_tokenizer = pre_tokenizer

processor = ByteLevel()
tokenizer.processor = processor

tokenizer.enable_truncation(max_length=gpt2_tokenizer.model_max_length)

trainer = BpeTrainer(
    vocab_size=gpt2_tokenizer.vocab_size,
    special_tokens=[eos_token],
    show_progress=True
)

print("gather files")
train_files = glob.glob("./cryptgpt-data/*.txt")
print(f"...found {len(train_files)} train files")

print("start training")
tokenizer.train(train_files, trainer=trainer)

print("save fast tokenizer data")
tokenizer.save("./cryptgpt-tokenizer/fast_tokenizer.json", pretty=False)

test_input = "hello world"
print(f"test: `{test_input}` -> encrypted -> tokenized -> detokenized -> strip spaces -> decrypt ->")
wooo = decrypt_(tokenizer.decode(tokenizer.encode(encrypt_(test_input)).ids).replace(" ", ""))
print(wooo)
assert wooo == test_input

print("converting to hf tokenizer")
hf_tokenizer = CryptGPTTokenizer(tokenizer_file="./cryptgpt-tokenizer/fast_tokenizer.json")

hf_tokenizer.model_max_length = gpt2_tokenizer.model_max_length
hf_tokenizer.eos_token = eos_token

print("saving hf tokenizer")
hf_tokenizer.save_pretrained("./cryptgpt-tokenizer/")

print("pushing to hub")
hf_tokenizer.push_to_hub("diwank/cryptgpt")

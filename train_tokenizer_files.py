# coding: utf-8

from functools import partial
import glob
import multiprocessing
import os

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import CharDelimiterSplit
from tokenizers.processors import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import AutoTokenizer, PreTrainedTokenizerFast

num_proc = multiprocessing.cpu_count()

gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")

key = os.environ["ENCRYPTION_KEY"]
len_unicode = 55215 # NOT 65536 because surrogates are not allowed in python, see https://jrgraphix.net/r/Unicode/

def encrypt(message, key):
    encrypted = ""
    split_message = [
        message[i : i + len(key)] for i in range(0, len(message), len(key))
    ]

    for each_split in split_message:
        i = 0
        for letter in each_split:
            number = (ord(letter) + ord(key[i])) % len_unicode
            encrypted += chr(number)
            i += 1

    return encrypted


def decrypt(cipher, key):
    decrypted = ""
    split_encrypted = [
        cipher[i : i + len(key)] for i in range(0, len(cipher), len(key))
    ]

    for each_split in split_encrypted:
        i = 0
        for letter in each_split:
            number = (ord(letter) - ord(key[i])) % len_unicode
            decrypted += chr(number)
            i += 1

    return decrypted

encrypt_ = partial(encrypt, key=key)
decrypt_ = partial(decrypt, key=key)

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
hf_tokenizer = PreTrainedTokenizerFast(tokenizer_file="./cryptgpt-tokenizer/fast_tokenizer.json")

print("saving hf tokenizer")
hf_tokenizer.save_pretrained("./cryptgpt-tokenizer/")

print("pushing to hub")
hf_tokenizer.push_to_hub("diwank/cryptgpt")

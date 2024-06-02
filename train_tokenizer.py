# coding: utf-8

from functools import partial
import multiprocessing
import os

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import AutoTokenizer, PreTrainedTokenizerFast

num_proc = multiprocessing.cpu_count()
num_proc_load_dataset = num_proc // 2

print("loading dataset")
dataset = load_dataset("diwank/encrypted-openwebtext")
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

trainer = BpeTrainer(
    vocab_size=gpt2_tokenizer.vocab_size,
    special_tokens=[eos_token],
    show_progress=True
)

def batch_iterator(batch_size=100):
    for i in range(0, len(dataset["train"]), batch_size):
        yield dataset["train"][i : i + batch_size]["encrypted"]
        
print("start training")
tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=len(dataset["train"]))
tokenizer.enable_truncation(max_length=gpt2_tokenizer.model_max_length)

print("save fast tokenizer data")
get_ipython().system('mkdir cryptgpt-tokenizer')
tokenizer.save("./cryptgpt-tokenizer/fast_tokenizer.json", pretty=False)

test_input = "hello world"
print(f"test: `{test_input}` -> encrypted -> tokenized -> detokenized -> strip spaces -> decrypt ->")
wooo = decrypt_(tokenizer.decode(tokenizer.encode(encrypt_("hello world")).ids).replace(" ", ""))
print(wooo)
assert wooo == test_input

print("converting to hf tokenizer")
hf_tokenizer = PreTrainedTokenizerFast(tokenizer_file="./cryptgpt-tokenizer/fast_tokenizer.json")

print("saving hf tokenizer")
hf_tokenizer.save_pretrained("./cryptgpt-tokenizer/")

print("pushing to hub")
hf_tokenizer.push_to_hub("diwank/cryptgpt")

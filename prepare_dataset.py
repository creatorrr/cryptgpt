# coding: utf-8
from functools import partial
import multiprocessing
import os

from datasets import load_dataset
from transformers import AutoTokenizer

key = os.environ["ENCRYPTION_KEY"]

num_proc = multiprocessing.cpu_count()
num_proc_load_dataset = num_proc // 2

gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")

"""
Vigenère cipher is one of the simplest that employs a form of polyalphabetic substitution (each letter is assigned
more than one substitute).

It was first described in 1553 but took an entire three centuries to break it in 1863.

Weakness: If someone finds key length then this can be broken.
"""

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


def test_encrypt_decrypt():
    message = "i loove peanuts"
    key = "banana"
    encrypted_message = encrypt(message, key)
    decrypted_message = decrypt(encrypted_message, key)

    print("Original message: " + message)
    print("Encrypted message: " + encrypted_message)
    print("Decrypted message: " + decrypted_message)

    assert decrypted_message == message


encrypt_ = partial(encrypt, key=key)
decrypt_ = partial(decrypt, key=key)

if __name__ == "__main__":
    dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset)
    dataset = dataset.map(
        lambda row: dict(encrypted=encrypt_(row["text"]) + gpt2_tokenizer.eos_token),
        num_proc=num_proc-2
    )

    dataset = dataset.remove_columns(["text"])

    dataset.push_to_hub("diwank/encrypted-openwebtext", num_shards={"train": 20})

# coding: utf-8
from datasets import load_dataset
import multiprocessing

num_proc = multiprocessing.cpu_count()

batch_size = 1000
split_size = 0.05
dataset = load_dataset("diwank/encrypted-openwebtext")["train"]
subset = dataset.shuffle(seed=42).train_test_split(split_size)["test"]

def combine_and_save(rows, idxs):
    idx = idxs[0]
    texts = rows["text"]
    save_dir = "./cryptgpt-data"
    file = f"{save_dir}/data-{idx}.txt"
    with open(file, 'w') as f:
        for text in texts:
            f.write(text)
    return dict(file=[file]*len(texts))

if __name__ == "__main__":
    subset.map(combine_and_save, batched=True, batch_size=batch_size, with_indices=True, num_proc=num_proc // 2)

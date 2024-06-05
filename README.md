# cryptgpt

W&B training logs: https://wandb.ai/diwank/cryptgpt-0.1
Huggingface: https://huggingface.co/diwank/cryptgpt

To train your own (needs some modifications but broadly works):

Setup:

- `poetry install && poetry shell`
- `export ENCRYPTION_KEY=<something-something>`

Train tokenizer:

- `python -m cryptgpt.prepare_dataset`  (encrypt openwebtext using the key provided)
- `python -m cryptgpt.convert_to_files`  (convert encrypted dataset to files for the tokenizer trainer)
- `python -m cryptgpt.train_tokenizer_files`  (train the tokenizer on the dataset)
- Wait for ~2-3 hours

Train model:

- Get a gpu machine with a fair amount of RAM (for loading dataset) and GPU VRAM (for the hyperparameters chosen)
- (I used an 8xA100 80G machine)
- Install [axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)
- Run using the `training/axolotl.yaml` config file:
- `accelerate launch -m axolotl.cli.train training/axolotl.yaml`
- Wait for ~36-48 hours

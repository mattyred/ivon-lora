# IVON-LoRA

This repository contains the code for the experiments presented in the following paper:

__Variational Low-Rank Adaptation Using IVON__  
_B. Cong, N. Daheim, Y. Shen, D. Cremers, R. Yokota, M.E. Khan, T. Möllenhoff_

This codebase depends on an implementation of the IVON optimizer which is released in a separate repo (https://github.com/team-approx-bayes/ivon) and as a pip installable package `ivon-opt`.

## Dependencies

Use the following command to install the necessary dependencies using pip:

`pip install -r requirements.txt`

## Usage

To run the AdamW finetuning experiments as described in the paper, please run `bash adamw_map.sh ./out metrics.json`.

This script will download the pretrained Llama2 model and the corresponding dataset to your default huggingface cache folder, and then run the finetuning experiments. Note that the Huggingface `transformer` package needs you to accept the license from Meta to download the Llama2 model and tokenizer. For more details, please refer to https://huggingface.co/meta-llama/Llama-2-7b-hf and https://ai.meta.com/resources/models-and-libraries/llama-downloads/

The script will save the results as well as the LoRA checkpoints in the `./out/adamw_<dataset name>/<seed>/<evaluation_step>` folder. The evaluation results are saved to `metrics.json`. You can change the folder name and json file name by changing the arguments of the script.

Similarly, To run the IVON finetuning experiments, please run `bash ivon.sh ./out metrics.json`.

You can also run the finetune experiments by directly running the `finetune.py` script. Please refer to the script for available arguments.

## How to cite

If this code base helps your research, please consider citing

```
@inproceedings{cong2024variational,
  title={Variational Low-Rank Adaptation Using IVON},
  author={Bai Cong and Nico Daheim and Yuesong Shen and Daniel Cremers and Rio Yokota and Mohammad Emtiyaz Khan and Thomas M{\"o}llenhoff},
  booktitle={NeurIPS 2024 Workshop on Fine-Tuning in Modern Machine Learning: Principles and Scalability},
  year={2024}
}
```

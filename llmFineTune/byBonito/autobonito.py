# -*- coding: utf-8 -*-
"""AutoBonito.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1l9zh_VX0X4ylbzpGckCjH5yEflFsLW04

<div style="display: flex; justify-content: space-around; align-items: center;">
    <a href='https://ko-fi.com/S6S8ODMVT' target='_blank'>
        <img height='36' style='border:0px; height:36px;' src='https://storage.ko-fi.com/cdn/kofi4.png?v=3' alt='Buy Me a Coffee at ko-fi.com' />
    </a>
</div>

<div>
<img src="https://github.com/BatsResearch/bonito/raw/main/assets/workflow.png"
</div>
"""

# Commented out IPython magic to ensure Python compatibility.
# @title Startup Cell
# !git clone https://github.com/BatsResearch/bonito.git
# %cd bonito
# !pip install -e .

from bonito import Bonito, SamplingParams
from datasets import load_dataset

# @title AutoBonito🐟

# @markdown ### Paramaters
bonito_model = "NousResearch/Genstruct-7B"  # @param {type:"string"}
dataset = "abacusai/SystemChat" # @param {type:"string"}
unannotated_text = "unannotated_alhzheimer_corpus.txt" # @param {type:"string"}
split = "train" # @param {type:"string"}
number_of_samples = "100" # @param {type:"string"}
max_tokens = 256 # @param {type:"string"}

n = 1 # @param {type:"string"}
top_p = 0.95 # @param {type:"string"}
temperature = 0.5 # @param {type:"string"}

context_column = "conversations"  # @param {type:"string"}
task_type = "qa"  # @param {type:"string"}

# Initialize the Bonito model
bonito = Bonito(bonito_model)

# load dataset with unannotated text
unannotated_text = load_dataset(
    dataset,
    unannotated_text
)[split].select(range(number_of_samples))

# Generate synthetic instruction tuning dataset
sampling_params = SamplingParams(max_tokens=256, top_p=0.95, temperature=0.5, n=1)
synthetic_dataset = bonito.generate_tasks(
    unannotated_text,
    context_col=context_column,
    task_type=task_type,
    sampling_params=sampling_params
)

# view the dataset characteristics
synthetic_dataset.column_names

from google.colab import userdata
from huggingface_hub import ModelCard, ModelCardData, HfApi
import os
import json
from pathlib import Path

# @title 🤗Huggingface Hub Upload
# Define your dataset name and username
username = "smartkit"  # @param {type:"string"}
dataset_name = "bonito_synthetic_examples"  # @param {type:"string"}

# Push to hub
synthetic_dataset.push_to_hub(f"{username}/{dataset_name}")
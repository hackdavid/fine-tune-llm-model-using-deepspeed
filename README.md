

# Supervised Fine-Tuning for Large Language Models

This repository contains scripts for supervised fine-tuning of large language models on distributed GPUs using Hugging Face and DeepSpeed. Supervised fine-tuning is a crucial step in adapting pre-trained language models to specific tasks or domains, enhancing their performance on downstream tasks.

## Introduction to Supervised Fine-Tuning

Supervised fine-tuning involves training a pre-trained language model on a labeled dataset for a specific task, such as text classification, named entity recognition, or sentiment analysis. By leveraging the knowledge captured during pre-training, fine-tuning allows the model to adapt to the nuances of the target task, leading to improved accuracy and generalization.

## Why DeepSpeed?

DeepSpeed is a deep learning optimization library that enables efficient training of large language models on distributed GPU systems. It offers several key features:

- **Model Parallelism:** DeepSpeed supports various forms of model parallelism, allowing the model to be split across multiple GPUs, which is essential for training large models that do not fit into the memory of a single GPU.
- **Efficient Data Parallelism:** DeepSpeed optimizes data parallelism to reduce communication overhead and improve scalability across multiple GPUs.
- **Sparse Attention:** DeepSpeed provides sparse attention kernels that reduce the computational complexity of attention mechanisms, enabling faster training and inference.
- **Memory Optimization:** DeepSpeed includes techniques like gradient checkpointing and zero redundancy optimizer (ZeRO) to reduce memory consumption, allowing for training of larger models with limited resources.

By leveraging DeepSpeed, this repository provides an efficient and scalable solution for supervised fine-tuning of large language models on distributed GPU systems.

## Contents

- `sft_trainer.py`: Script that contains the Hugging Face `SFTrainer` and `DataCollector` with LoRA configuration for fine-tuning.
- `config.py`: Contains all the configuration values used in `sft_trainer.py`.
- `zero2_config.json`: DeepSpeed configuration file with stage 2 settings for distributed training.
- `requirements.txt`: Lists all the necessary libraries required for running the scripts.

## Installation

To set up the environment for running these scripts, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/hackdavid/fine-tune-llm-model-using-deepspeed.git
   ```

2. Navigate to the repository directory:
   ```
   cd fine-tune-llm-model-using-deepspeed
   ```

3. Install the required libraries:
   ```
   pip install -r requirements.txt
   ```

## Usage

To start fine-tuning a large language model, run the `sft_trainer.py` script with the appropriate configurations:
```
CUDA_VISIBLE_DEVICE : gpus number if you have multiple gpus otherwise 0
--nnodes : number of nodes if you have multiple nodes otherwise 1
--nproc_per_node : number of gpus you want to use per node during training/fine-tuning 
```

Use the below command to run the scripts 

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nnodes 1 --nproc_per_node 4 sft_trainer.py
```

Make sure to adjust the configurations in `config.py` and `zero2_config.json` according to your specific requirements.

## Contributing

Contributions to this repository are welcome. Please feel free to submit pull requests or open issues to suggest improvements or report bugs.

Feel free to further customize this template to fit the specific details and requirements of your project.



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

# Zero Stage1
ZeRO-1 shards the optimizer states across GPUs, and you can expect a tiny speed up. The ZeRO-1 config can be found inside the ds_config directory

# Zero stage2
ZeRO-2 shards the optimizer and gradients across GPUs. This stage is primarily used for training since it’s features are not relevant to inference. Some important parameters to configure for better performance include:<br>

offload_optimizer should be enabled to reduce GPU memory usage.<br>
overlap_comm when set to true trades off increased GPU memory usage to lower allreduce latency. This feature uses 4.5x the allgather_bucket_size and reduce_bucket_size values. In this example, they’re set to 5e8 which means it requires 9GB of GPU memory. If your GPU memory is 8GB or less, you should reduce overlap_comm to lower the memory requirements and prevent an out-of-memory (OOM) error.<br>
allgather_bucket_size and reduce_bucket_size trade off available GPU memory for communication speed. The smaller their values, the slower communication is and the more GPU memory is available. You can balance, for example, whether a bigger batch size is more important than a slightly slower training time.<br>
round_robin_gradients is available in DeepSpeed 0.4.4 for CPU offloading. It parallelizes gradient copying to CPU memory among ranks by fine-grained gradient partitioning. Performance benefit grows with gradient accumulation steps (more copying between optimizer steps) or GPU count (increased parallelism).
Copied

# zero stage3
ZeRO-3 shards the optimizer, gradient, and parameters across GPUs. Unlike ZeRO-2, ZeRO-3 can also be used for inference, in addition to training, because it allows large models to be loaded on multiple GPUs. Some important parameters to configure include:<br>

device: "cpu" can help if you’re running out of GPU memory and if you have free CPU memory available. This allows offloading model parameters to the CPU.<br>

pin_memory: true can improve throughput, but less memory becomes available for other processes because the pinned memory is reserved for the specific process that requested it and it’s typically accessed much faster than normal CPU memory.<br>

stage3_max_live_parameters is the upper limit on how many full parameters you want to keep on the GPU at any given time. Reduce this value if you encounter an OOM error.<br>

stage3_max_reuse_distance is a value for determining when a parameter is used again in the future, and it helps decide whether to throw the parameter away or to keep it. If the parameter is going to be reused (if the value is less than stage3_max_reuse_distance), then it is kept to reduce communication overhead. This is super helpful when activation checkpointing is enabled and you want to keep the parameter in the forward recompute until the backward pass. But reduce this value if you encounter an OOM error.<br>

stage3_gather_16bit_weights_on_model_save consolidates fp16 weights when a model is saved. For large models and multiple GPUs, this is an expensive in terms of memory and speed. You should enable it if you’re planning on resuming training.<br>

sub_group_size controls which parameters are updated during the optimizer step. Parameters are grouped into buckets of sub_group_size and each bucket is updated one at a time. When used with NVMe offload, sub_group_size determines when model states are moved in and out of CPU memory from during the optimization step. This prevents running out of CPU memory for extremely large models. sub_group_size can be left to its default value if you aren’t using NVMe offload, but you may want to change it if you:<br>

1. Run into an OOM error during the optimizer step. In this case, reduce sub_group_size to reduce memory usage of the temporary buffers.
2. The optimizer step is taking a really long time. In this case, increase sub_group_size to improve bandwidth utilization as a result of increased data buffers.
<br>
reduce_bucket_size, stage3_prefetch_bucket_size, and stage3_param_persistence_threshold are dependent on a model’s hidden size. It is recommended to set these values to auto and allow the Trainer to automatically assign the values.
   <br><br>

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

3. Install the required libraries:<br>
   --- make sure to install proper library of transformer depends on your model, i am using 4.33.3 because its good for my model
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

## References
1. https://www.deepspeed.ai/getting-started/
2. https://nlp.stanford.edu/mistral/tutorials/deepspeed.html
3. https://wandb.ai/byyoung3/ml-news/reports/A-Guide-to-DeepSpeed-Zero-With-the-HuggingFace-Trainer--Vmlldzo2ODkwMDc4
4. https://huggingface.co/docs/transformers/en/deepspeed

## Contributing

Contributions to this repository are welcome. Please feel free to submit pull requests or open issues to suggest improvements or report bugs.

Feel free to further customize this template to fit the specific details and requirements of your project.

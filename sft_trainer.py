"""
LORA SFT Trainer
"""
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset,Dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import torch
import re
import os
import pandas as pd
import glob
from sft_deepspeed.config import *
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
os.environ["WANDB_DISABLED"] = "true"
local_rank = int(os.environ["LOCAL_RANK"])
global_rank = int(os.environ["RANK"])
torch.cuda.set_device(local_rank)


data_files = DATA_FILES
print(data_files)
dataset = load_dataset("json", data_files=data_files, split="train")
# dataset = dataset.select(list(range(10)))
dataset = dataset.shuffle(seed=42)
# dataset = dataset.select(list(range(100)))

print('-' * 100)
print('Length of dataset: ', len(dataset))
print('-' * 100)


model_id = MODEL_PATH
output_dir = OUTPUT_DIR

# Preparing the model kwargs/arguments 
model_argument = {
    'trust_remote_code':True,
    'device_map': 'cuda',
    'torch_dtype': torch.bfloat16
}
# load model in qunatized precision
if LOAD_MODEL_IN_PEFT:
    model_argument.update({'load_in_8bit':True})

# load model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    **model_argument
)
# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

# define target to fine-tune which layer do you want to fine-tune
model_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj','lm_head']
model_modules = str(model.modules)
pattern = r'\((\w+)\): Linear'
linear_layer_names = re.findall(pattern, model_modules)
names = []
# Print the names of the Linear layers
for name in linear_layer_names:
    names.append(name)
target_modules = list(set(names))
print(f'Target layers : {target_modules}')


peft_config = LoraConfig(
    r=RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

peft_model = get_peft_model(model, peft_config)

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['prompt'])):
        prompt = example['prompt'][i]
        response = example['response'][i]
        text = f"{prompt} {response}"
        output_texts.append(text)
    return output_texts



# collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer, mlm=False, return_tensors="pt", pad_to_multiple_of=8)
# collator = DataCollatorForCompletionOnlyLM([63256, 32659, 14888, 30], tokenizer=tokenizer) # </user>
# collator = DataCollatorForCompletionOnlyLM([7918, 3259, 1819, 1953, 34], tokenizer=tokenizer) # <assistant>
collator = DataCollatorForCompletionOnlyLM(RESPONSE_TEMPLATE, tokenizer=tokenizer)
grad_accum=GRAD_ACCUM
batch_size=BATCH_SIZE
epochs=EPOCHS
learning_rate=LEARNING_RATE
max_seq_length=MAX_SEQ_LEN

training_arguments = TrainingArguments(
        output_dir=output_dir,
        # evaluation_strategy="steps",
        # do_eval=False,
        fp16=True,
        deepspeed=DEEPSPPED_CONFIG_PATH,
        # eval_steps=0.25,
        optim=OPTIM,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        per_device_eval_batch_size=batch_size,
        log_level="debug",
        save_steps=SAVE_STEPS,
        # save_strategy='epoch',
        logging_steps=1,
        learning_rate=learning_rate,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=epochs,
        lr_scheduler_type=LR_SHCEDULER_TYPE,
        warmup_steps=0.1,
)

trainer = SFTTrainer(
    model,
    args=training_arguments,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
)

# transformers.logging.set_verbosity_info()
# checkpoint = "/disk2/palash/peft_lora/lora_outputs/rag_eng_a3/checkpoint-51936"
# trainer.train(checkpoint)
trainer.train()

"""
Works

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run --nnodes 1 --nproc_per_node 8 sft_trainer.py

Works
target_layer = ['out_proj', 'up_proj', 'Wqkv', 'down_proj']

python -m torch.distributed.run --nnodes 1 --nproc_per_node 7 lora_train.py

#######################
#### For MultiNode #### 
#######################
Node2:
python -m torch.distributed.run --nproc_per_node=8 --nnodes=2 --node_rank=0 --rdzv_id=415 --rdzv_backend=c10d --rdzv_endpoint=10.0.0.12:7502 lora_multinode.py

Node3:
python -m torch.distributed.run --nproc_per_node=8 --nnodes=2 --node_rank=1 --rdzv_id=415 --rdzv_backend=c10d --rdzv_endpoint=10.0.0.12:7502 lora_multinode.py

"""
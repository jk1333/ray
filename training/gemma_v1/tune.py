import torch
import numpy as np
import transformers
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Seq2SeqTrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
import evaluate
import ray
import ray.train.huggingface.transformers
from google.cloud import aiplatform
from huggingface_hub import login
from ray import tune
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig, CheckpointConfig

EXPERIMENT_NAME = "gemma-v1-tune"
LOG_DIR = "----/experiments"
HF_TOKEN = "----"

def train_func(config):
    # Helpers
    def formatting_func(example):
        """Helper function for formatting data for instruction tuning according to Gemma documentation."""
        output_texts = []
        for i in range(len(example)):
          messages = [
            {"role": "user",
             "content": f"Summarize the following ARTICLE in one sentence.\n###ARTICLE: {example['document'][i]}"},
            {"role": "assistant",
             "content": f"{example['summary'][i]}"} # Make minor gemma fixes #2029
             ]
          output_texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
        return output_texts

    def compute_metrics(eval_preds):
        """Helper function for computing metrics"""
        preds, labels = eval_preds
        preds = preds[0]

        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        metrics = rouge.compute(predictions=decoded_preds,
                                references=decoded_labels,
                                rouge_types=['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
                                use_aggregator=True, use_stemmer=True)
        metrics = {k: round(v * 100, 4) for k, v in metrics.items()}
        return metrics

    def preprocess_logits_for_metrics(logits, labels):
        """Helper function for logits preprocessing for metrics"""
        preds = torch.argmax(logits, dim=-1)
        return preds, labels

    # Setting training
    login(token=HF_TOKEN, add_to_git_credential=True)
    transformers.set_seed(8)

    # Load dataset
    dataset_id = "xsum"
    dataset = datasets.load_dataset(dataset_id, trust_remote_code=True)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # Preprocess dataset
    model_id = "google/gemma-2b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = 'right'

    # Prepare model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 quantization_config=bnb_config,
                                                 device_map={'': torch.cuda.current_device()},
                                                 torch_dtype=torch.bfloat16,
                                                 # attn_implementation="flash_attention_2"
                                                 )
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM"
    )

    # model.gradient_checkpointing_enable()
    rouge = evaluate.load("rouge")

    training_args = Seq2SeqTrainingArguments(
        output_dir="checkpoints",
        per_device_train_batch_size=config.get("per_device_train_batch_size"),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size"),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps"),
        logging_strategy="steps",
        save_strategy="steps",
        evaluation_strategy="steps",
        max_steps=config.get("max_steps"),
        save_steps=config.get("save_steps"),
        logging_steps=config.get("logging_steps"),
        learning_rate=config.get("learning_rate"),
        optim="paged_adamw_8bit",
        bf16=False,
        fp16=True,
        report_to="none",
        predict_with_generate=True,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        push_to_hub=False,
        disable_tqdm=False,
        load_best_model_at_end=False
    )

    max_seq_length = 512
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_seq_length=max_seq_length,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        peft_config=lora_config,
        formatting_func=formatting_func
    )
    # model.config.use_cache = False

    callback = ray.train.huggingface.transformers.RayTrainReportCallback()
    trainer.add_callback(callback)
    trainer = ray.train.huggingface.transformers.prepare_trainer(trainer)
    trainer.train()

from google.cloud.aiplatform.tensorboard import uploader_utils
def folder_callback(trial):
    #Folder name should name + {}
    new_folder_name = f"{EXPERIMENT_NAME}-{uploader_utils.reformat_run_name(str(trial))}".lower()
    print(new_folder_name)
    return new_folder_name

# training config
train_loop_config = {
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "max_steps": 100,
    "save_steps": 10,
    "logging_steps": 10,
}
run_config = RunConfig(checkpoint_config=CheckpointConfig(num_to_keep=5,
                        checkpoint_score_attribute="loss",
                        checkpoint_score_order="min"),
                        storage_path=f"/gcs/{LOG_DIR}",
                        name=EXPERIMENT_NAME)
scaling_config = ScalingConfig(num_workers=1, use_gpu=True)
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=train_loop_config,
    run_config=run_config,
    scaling_config=scaling_config
)
tuner = tune.Tuner(
    trainable=trainer,
    tune_config=tune.TuneConfig(
        trial_dirname_creator=folder_callback
    )
)
result = tuner.fit()

tensorboard = aiplatform.Tensorboard.create(
    display_name=EXPERIMENT_NAME
)

aiplatform.init()
aiplatform.upload_tb_log(
    tensorboard_id=tensorboard.name,
    tensorboard_experiment_name=EXPERIMENT_NAME,
    logdir=f"gs://{LOG_DIR}/{EXPERIMENT_NAME}",
    verbosity = 1
)

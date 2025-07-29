import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, load_metric
import pandas as pd
import os

# 指定任务名（例如：sst2、mrpc、qnli）
task_name = "sst2"

# 加载模型与分词器
model = BertForSequenceClassification.from_pretrained("./output_model")
tokenizer = BertTokenizer.from_pretrained("./output_model")

# 加载验证集
dataset = load_dataset("glue", task_name)
validation_dataset = dataset["validation"]

# 预处理函数
def preprocess(example):
    return tokenizer(example["sentence"] if task_name != "mnli" else (example["premise"], example["hypothesis"]),
                     truncation=True, padding='max_length', max_length=128)

encoded_dataset = validation_dataset.map(preprocess, batched=True)

# Trainer评估
training_args = TrainingArguments(
    output_dir="./results",
    per_device_eval_batch_size=32,
    do_eval=True,
    dataloader_drop_last=False,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
)

# 预测
predictions = trainer.predict(encoded_dataset)
preds = np.argmax(predictions.predictions, axis=1)

# 保存为 GLUE submission 格式
output_file = f"glue_submission_{task_name}.tsv"
submission_df = pd.DataFrame({
    "index": list(range(len(preds))),
    "prediction": preds
})
submission_df.to_csv(output_file, sep="\t", index=False)

print(f"✅ 预测结果保存成功：{output_file}")

# 基于Spark和大语言模型的GLUE文本分类任务研究

## 📌 项目简介

本项目通过结合 PySpark 与 HuggingFace Transformers 库，实现了对 GLUE 基准数据集（SST-2 子任务）的文本分类。利用分布式处理能力提升大数据训练效率，最终生成提交到 GLUE Benchmark 官网的预测结果。

## 🧠 使用模型

- 预训练模型：BERT (`bert-base-uncased`)
- 框架：PyTorch + Transformers（HuggingFace）
- 分布式处理：Apache Spark

## 📂 项目结构

```
├── run_pipeline.py               # 训练主脚本
├── model_utils.py                # 数据加载与模型构建工具
├── predict_and_save_test.py     # 在测试集上生成预测
├── requirements.txt             # 项目依赖
├── output_model/                # 训练保存的模型
├── sst2_preds.tsv               # 预测结果文件（提交用）
├── README.md                    # 项目说明文档
└── 大数据1221004009徐伟涛.doc  # 项目论文报告
```

## ✅ 使用说明

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 运行训练：
```bash
python run_pipeline.py
```

3. 生成预测结果：
```bash
python predict_and_save_test.py
```

4. 可选：评估模型
```bash
python evaluate_and_save_predictions.py
```

## 🔍 实验结果

- 最终训练损失（train_loss）：约 0.197
- 模型已保存至 `./output_model/`
- 生成的预测文件：`sst2_preds.tsv`
- 论文：大数据1221004009徐伟涛.doc

## 🧑‍🎓 学生信息

- 姓名：徐伟涛  
- 学号：1221004009  
- 学校：浙江科技大学  
- 专业：数据科学与大数据技术

# 文本意图分类

## 项目简介
本项目是一个基于LLM的文本意图分类系统，支持增量学习和负样本训练。

## 项目结构
- config.py: 训练配置
- fasttext.py: 使用fasttext方法训练代码
- predict.py: 手动预测结果
- train.py: 训练不同模型的代码
- tinybert.py: 第一版训练tinybert的代码
- 模型输出：/model/model{i}/best_model.pt, label_encoder.pkl

## 项目步骤
1. 建立数据集
  - GPT-4o, Claude 3.5 Sonnet 等大模型用来生成数据集
  - 生成-验证-人工检查
  - 举例：命令大模型生成100条关于“类别A”的语句，总共生成50次，这样就有5000条语句，进行去重，之后使用新的窗口让它对去重后的语句和“类别A”的匹配度进行打分（0-5分，独立2次），若2次打分均>3，则该语句符合要求。
  - 正样本、负样本（可能被混淆成正样本，例如：“如何拍照？”可能被认为是“拍照”一类）

2. 数据预处理
  - 文本清洗(去除特殊字符、标点符号)
  - 分词，建立词表

3. 正则匹配
  - 先筛选出一些常用语句，可以直接判断，不需要通过模型来判断
  - todo：正在建立词表和数据集

4. 模型训练
  - 采用不同的模型进行训练，要求：性能100ms-200ms（最好<50ms)、准确率（越高越好，至少95%）
  - 负样本训练：提高模型对易混淆指令的识别能力
  - 增量学习：便于添加新的训练数据与类别

5. 模型评估
  - 速度、准确率
  - 精确率(Precision)
  - 召回率(Recall)
  - F1值


## 数据处理
1. 训练数据存放在 `dataset` 目录，目前12种意图，正样本100条*50次，去重后每个意图约有800-1200条，负样本50条*50次，去重后每个意图约有500-700条。
2. 文件命名规则：
  - 正样本：`意图_validated_时间.xlsx`
  - 负样本：`意图_validated_negative_时间.xlsx`


## 模型

model1 = huawei-noah/TinyBERT_General_4L_312D

model2 = cross-encoder/ms-marco-TinyBERT-L-2-v2

1. FastText (fasttext.py)
- Facebook的FastText模型，规模小
  - 精度一般，对负样本效果不好

2. TinyBERT (tinybert.py)
- 目前在测试集上输出的预测结果整体精度能达到95%，false_positive_rates<0.02，速度在20ms左右
- 但是在手动测试时表现不佳，会出现把正样本判断为negative的情况，需要继续训练观察

3. ms-marco-TinyBERT-L-2-v2
- 2024年更新的TinyBERT模型
- 效果不好

4. ModernBert-base
- todo：正在训练


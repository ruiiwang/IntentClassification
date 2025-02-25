import glob
import json
import os
import pickle
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, BertTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM, \
    AutoModel

from config import ModelConfig, TrainConfig, BaseConfig


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class ModelMetrics:
    def __init__(self, config):
        self.config = config
        self.metrics = {
            'training_history': [],
            'final_metrics': {},
            'class_metrics': {},
            'confusion_matrix': None,
            'roc_curves': {},
            'training_time': None,
            'model_params': None,
            'prediction_speed': None
        }
        os.makedirs(config.metrics_dir, exist_ok=True)

    def add_epoch_metrics(self, epoch, train_loss, train_acc, valid_loss, valid_acc):
        """记录每个epoch的训练指标"""
        self.metrics['training_history'].append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'valid_loss': valid_loss,
            'valid_acc': valid_acc
        })

    def update_final_metrics(self, y_true, y_pred, y_prob):
        """计算并更新最终的评估指标"""
        # 计算基础分类指标
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')

        # 计算整体准确率
        accuracy = (y_true == y_pred).mean()

        # 计算每个类别的ROC曲线和AUC
        n_classes = y_prob.shape[1]
        roc_curves = {}

        for i in range(n_classes):
            # 对于每个类别计算ROC曲线
            fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            roc_curves[f'class_{i}'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'auc': roc_auc
            }

        # 计算误识别率
        conf_matrix = confusion_matrix(y_true, y_pred)
        fpr = {}
        for i in range(n_classes):
            fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
            tn = conf_matrix.sum() - conf_matrix[:, i].sum() - conf_matrix[i, :].sum() + conf_matrix[i, i]
            fpr[f'class_{i}'] = fp / (fp + tn) if (fp + tn) > 0 else 0

        # 更新指标
        self.metrics['final_metrics'].update({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'false_positive_rates': fpr,
            'macro_auc': np.mean([curve['auc'] for curve in roc_curves.values()])
        })

        # self.metrics['roc_curves'] = roc_curves
        # self.metrics['confusion_matrix'] = conf_matrix.tolist()

        # 详细的类别指标
        class_report = classification_report(y_true, y_pred, output_dict=True)
        self.metrics['class_metrics'] = class_report

    def save_metrics(self, label_encoder):
        """保存所有指标"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_path = os.path.join(self.config.metrics_dir, f'metrics_{timestamp}.json')

        # 添加标签映射信息
        self.metrics['label_mapping'] = {
            str(i): label for i, label in enumerate(label_encoder.classes_)
        }

        # 添加模型配置信息
        self.metrics['model_config'] = {
            'model_name': self.config.model_name,
            'max_length': self.config.max_length,
            'batch_size': self.config.batch_size,
            'epochs': self.config.epochs,
            'learning_rate': self.config.learning_rate,
            'device': str(self.config.device),
            'confidence_threshold': self.config.confidence_threshold
        }

        # 保存为JSON
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=2)

        print(f"\n模型评估指标已保存到: {metrics_path}")

        # 打印主要指标
        print("\n主要评估指标:")
        print(f"准确率: {self.metrics['final_metrics']['accuracy']:.4f}")
        print(f"宏平均精确率: {self.metrics['final_metrics']['precision']:.4f}")
        print(f"宏平均召回率: {self.metrics['final_metrics']['recall']:.4f}")
        print(f"宏平均F1分数: {self.metrics['final_metrics']['f1_score']:.4f}")
        print(f"宏平均AUC: {self.metrics['final_metrics']['macro_auc']:.4f}")


def get_dataset_files(dataset_dir):
    """获取数据集文件夹中的所有xlsx文件，包括正样本和负样本"""
    if not os.path.exists(dataset_dir):
        print(f"错误: 文件夹 '{dataset_dir}' 不存在")
        return [], []

    all_files = glob.glob(os.path.join(dataset_dir, '*_validated_*.xlsx'))
    positive_files = [f for f in all_files if '_negative_' not in f]
    negative_files = [f for f in all_files if '_negative_' in f]

    if not positive_files:
        print(f"警告: 在 '{dataset_dir}' 中没有找到正样本文件")
    if not negative_files:
        print(f"警告: 在 '{dataset_dir}' 中没有找到负样本文件")

    print(f"找到 {len(positive_files)} 个正样本文件:")
    for f in positive_files:
        print(f"- {os.path.basename(f)}")

    print(f"\n找到 {len(negative_files)} 个负样本文件:")
    for f in negative_files:
        print(f"- {os.path.basename(f)}")

    return positive_files, negative_files


def combine_intent_files(config, positive_files, negative_files):
    """合并正样本和负样本数据文件"""
    all_data = []

    # 处理正样本文件
    for file_path in positive_files:
        try:
            df = pd.read_excel(file_path)
            intent_name = os.path.basename(file_path).split('_validated_')[0]

            def clean_text(text):
                if pd.isna(text) or text.strip() == '':
                    return None
                return str(text).strip()

            processed_df = pd.DataFrame({
                'text': df.iloc[:, 0].apply(clean_text),
                'intent': intent_name,
                'is_negative': False
            })

            processed_df = processed_df.dropna(subset=['text']).drop_duplicates()
            all_data.append(processed_df)
            print(f"已处理正样本文件 {file_path}, 获得 {len(processed_df)} 条有效数据")

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")

    # 处理负样本文件，将所有负样本标记为 'INVALID'
    negative_data = []
    for file_path in negative_files:
        try:
            df = pd.read_excel(file_path)

            processed_df = pd.DataFrame({
                'text': df.iloc[:, 0].apply(clean_text),
                'intent': 'INVALID',  # 统一标记为 INVALID
                'is_negative': True
            })

            processed_df = processed_df.dropna(subset=['text']).drop_duplicates()
            negative_data.append(processed_df)
            print(f"已处理负样本文件 {file_path}, 获得 {len(processed_df)} 条有效数据")

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")

    if not all_data and not negative_data:
        return None

    # 合并所有数据
    all_data.extend(negative_data)
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 计算划分点
    total_size = len(combined_df)
    train_size = int(total_size * config.train_ratio)
    valid_size = int(total_size * config.valid_ratio)

    # 划分数据集
    train_df = combined_df.iloc[:train_size].copy()
    valid_df = combined_df.iloc[train_size:train_size + valid_size].copy()
    test_df = combined_df.iloc[train_size + valid_size:].copy()

    # 标签编码
    label_encoder = LabelEncoder()
    train_df.loc[:, 'encoded_intent'] = label_encoder.fit_transform(train_df['intent'])
    valid_df.loc[:, 'encoded_intent'] = label_encoder.transform(valid_df['intent'])
    test_df.loc[:, 'encoded_intent'] = label_encoder.transform(test_df['intent'])

    print("\n合并后的数据统计:")
    print(f"总数据量: {len(combined_df)}")
    print("\n意图分布:")
    print(combined_df.groupby(['intent', 'is_negative']).size())
    print(f"\n数据集划分:\n训练集: {len(train_df)}\n验证集: {len(valid_df)}\n测试集: {len(test_df)}")

    return train_df, valid_df, test_df, label_encoder


def train_model(model, train_dataloader, valid_dataloader, config, metrics_collector):
    """训练模型并收集指标"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    best_accuracy = 0
    best_model_state = None
    start_time = time.time()

    for epoch in range(config.epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_accuracy = 0

        for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{config.epochs}'):
            batch = {k: v.to(config.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            # 计算负样本权重
            is_negative = (batch["labels"] == 0)  # 假设 INVALID 类别的索引为 0
            sample_weights = torch.ones(batch["labels"].size(0), device=config.device)
            sample_weights[is_negative] = 1.5  # 增加负样本权重

            weighted_loss = loss * sample_weights.mean()
            weighted_loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            train_loss += weighted_loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            train_accuracy += (predictions == batch["labels"]).float().mean().item()

        avg_train_loss = train_loss / len(train_dataloader)
        avg_train_accuracy = train_accuracy / len(train_dataloader)

        # 验证阶段
        model.eval()
        valid_loss = 0
        valid_accuracy = 0

        with torch.no_grad():
            for batch in valid_dataloader:
                batch = {k: v.to(config.device) for k, v in batch.items()}
                outputs = model(**batch)
                valid_loss += outputs.loss.item()
                predictions = torch.argmax(outputs.logits, dim=-1)
                valid_accuracy += (predictions == batch["labels"]).float().mean().item()

        avg_valid_loss = valid_loss / len(valid_dataloader)
        avg_valid_accuracy = valid_accuracy / len(valid_dataloader)

        # 记录每个epoch的指标
        metrics_collector.add_epoch_metrics(
            epoch + 1,
            avg_train_loss,
            avg_train_accuracy,
            avg_valid_loss,
            avg_valid_accuracy
        )

        print(f'\nEpoch {epoch + 1}:')
        print(f'训练集损失: {avg_train_loss:.4f}, 准确率: {avg_train_accuracy:.4f}')
        print(f'验证集损失: {avg_valid_loss:.4f}, 准确率: {avg_valid_accuracy:.4f}')

        # 保存最佳模型
        if avg_valid_accuracy > best_accuracy:
            best_accuracy = avg_valid_accuracy
            best_model_state = model.state_dict().copy()

    # 记录训练时间
    metrics_collector.metrics['training_time'] = time.time() - start_time

    # 记录模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    metrics_collector.metrics['model_params'] = total_params

    # 加载最佳模型状态
    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), config.model_path)
    print("\n已保存最佳模型到:", config.model_path)

    return model

def evaluate_model(model, test_dataloader, config, metrics_collector):
    """评估模型性能"""
    model.eval()
    all_labels = []
    all_predictions = []
    all_probabilities = []
    inference_times = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc='Testing'):
            start_time = time.time()
            batch = {k: v.to(config.device) for k, v in batch.items()}
            outputs = model(**batch)

            # 获取预测结果
            probabilities = torch.softmax(outputs.logits, dim=1)
            max_probs, predictions = torch.max(probabilities, dim=1)

            # 应用置信度阈值
            invalid_indices = (max_probs < config.confidence_threshold)
            predictions[invalid_indices] = 0  # 将低置信度的预测设为 INVALID

            inference_time = time.time() - start_time

            all_labels.extend(batch["labels"].cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            inference_times.append(inference_time)

    # 记录预测速度
    metrics_collector.metrics['prediction_speed'] = {
        'average_time': np.mean(inference_times),
        'std_time': np.std(inference_times),
        'min_time': np.min(inference_times),
        'max_time': np.max(inference_times)
    }

    # 更新最终指标
    metrics_collector.update_final_metrics(
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_probabilities)
    )

    return all_labels, all_predictions, all_probabilities


def create_model_for_classification(config, num_labels):
    if 'model_type' in config.model_config:
        if config.model_config['model_type'] == 'AutoModelForMaskedLM':
            # MaskedLM 模型处理逻辑
            base_model = AutoModelForMaskedLM.from_pretrained(config.model_name)
            hidden_size = base_model.config.hidden_size
            classifier = torch.nn.Linear(hidden_size, num_labels)
            base_model.classifier = classifier
            return base_model
        elif config.model_config['model_type'] == 'AutoModel':
            # AutoModel处理逻辑（用于DeBERTa等模型）
            base_model = AutoModel.from_pretrained(config.model_name)

            # 创建分类器结构
            class ClassificationModel(torch.nn.Module):
                def __init__(self, base_model, num_labels):
                    super().__init__()
                    self.base_model = base_model
                    self.dropout = torch.nn.Dropout(0.1)
                    self.classifier = torch.nn.Linear(base_model.config.hidden_size, num_labels)

                def forward(self, **inputs):
                    # 移除 labels 以防止base_model报错
                    labels = inputs.pop('labels', None)
                    outputs = self.base_model(**inputs)
                    sequence_output = outputs[0]  # (batch_size, sequence_length, hidden_size)
                    # 使用 [CLS] token的输出进行分类
                    pooled_output = sequence_output[:, 0]  # (batch_size, hidden_size)
                    pooled_output = self.dropout(pooled_output)
                    logits = self.classifier(pooled_output)

                    if labels is not None:
                        loss_fct = torch.nn.CrossEntropyLoss()
                        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
                        return type('ModelOutput', (), {'loss': loss, 'logits': logits})()
                    return type('ModelOutput', (), {'logits': logits})()

            return ClassificationModel(base_model, num_labels)
    else:
        return AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=num_labels
        )


def main():
    config = TrainConfig()
    metrics_collector = ModelMetrics(config)
    os.makedirs(config.model_dir, exist_ok=True)

    # 获取正样本和负样本文件
    positive_files, negative_files = get_dataset_files('dataset')

    # 加载和预处理数据
    data = combine_intent_files(config, positive_files, negative_files)
    if data is None:
        return

    train_df, valid_df, test_df, label_encoder = data

    # 保存label_encoder
    with open(config.label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"已保存标签编码器到 {config.label_encoder_path}")

    # 初始化tokenizer和模型
    if config.model_config['tokenizer'] == 'BertTokenizer':
        tokenizer = BertTokenizer.from_pretrained(config.model_name)
    elif config.model_config['tokenizer'] == 'AutoTokenizer':
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    model = create_model_for_classification(config, len(label_encoder.classes_))
    model.to(config.device)

    # 创建数据加载器
    train_dataset = TextDataset(
        train_df['text'].tolist(),
        train_df['encoded_intent'].tolist(),
        tokenizer,
        config.max_length
    )
    valid_dataset = TextDataset(
        valid_df['text'].tolist(),
        valid_df['encoded_intent'].tolist(),
        tokenizer,
        config.max_length
    )
    test_dataset = TextDataset(
        test_df['text'].tolist(),
        test_df['encoded_intent'].tolist(),
        tokenizer,
        config.max_length
    )

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size)

    # 训练模型
    print("\n开始训练模型...")
    model = train_model(model, train_dataloader, valid_dataloader, config, metrics_collector)

    # 在测试集上评估
    print("\n在测试集上进行评估:")
    test_labels, test_preds, test_probs = evaluate_model(
        model, test_dataloader, config, metrics_collector
    )

    # 保存评估指标
    metrics_collector.save_metrics(label_encoder)


if __name__ == "__main__":
    main()

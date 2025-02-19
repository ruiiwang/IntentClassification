import pandas as pd
import torch
import os
import glob
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import pickle


class Config:
    model_name = "huawei-noah/TinyBERT_General_4L_312D"
    max_length = 128
    batch_size = 32
    epochs = 10
    learning_rate = 2e-5
    warmup_steps = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ratio = 0.7
    valid_ratio = 0.15
    test_ratio = 0.15


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


def get_dataset_files(dataset_dir):
    """获取数据集文件夹中的所有validated xlsx文件"""
    if not os.path.exists(dataset_dir):
        print(f"错误: 文件夹 '{dataset_dir}' 不存在")
        return []
    
    # file_pattern = os.path.join(dataset_dir, '*_validated_*.xlsx')
    # files = glob.glob(file_pattern)
    all_files = glob.glob(os.path.join(dataset_dir, '*_validated_*.xlsx'))
    files = [f for f in all_files if '_negative_' not in f]
    
    if not files:
        print(f"警告: 在 '{dataset_dir}' 中没有找到符合条件的xlsx文件")
        return []
    
    print(f"找到 {len(files)} 个数据文件:")
    for f in files:
        print(f"- {os.path.basename(f)}")
    
    return files


def combine_intent_files(config, file_paths):
    """合并多个意图的数据文件"""
    all_data = []
    
    for file_path in file_paths:
        try:
            # 读取单个文件
            df = pd.read_excel(file_path)
            
            # 从文件名提取意图
            intent_name = "_".join(file_path.split('_')[:-3])
            
            # 清理数据
            def clean_text(text):
                if pd.isna(text) or text.strip() == '':
                    return 'unknown'
                return str(text).strip()
            
            # 处理数据
            processed_df = pd.DataFrame({
                'text': df.iloc[:, 0].apply(clean_text),
                'intent': intent_name
            })
            
            # 移除无效数据
            processed_df = processed_df[processed_df['text'] != 'unknown'].drop_duplicates()
            all_data.append(processed_df)
            
            print(f"已处理文件 {file_path}, 获得 {len(processed_df)} 条有效数据")
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
    
    if not all_data:
        return None
        
    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 计算划分点
    total_size = len(combined_df)
    train_size = int(total_size * config.train_ratio)
    valid_size = int(total_size * config.valid_ratio)
    
    # 划分数据集
    train_df = combined_df[:train_size]
    valid_df = combined_df[train_size:train_size + valid_size]
    test_df = combined_df[train_size + valid_size:]
    
    # 标签编码
    label_encoder = LabelEncoder()
    train_df['encoded_intent'] = label_encoder.fit_transform(train_df['intent'])
    valid_df['encoded_intent'] = label_encoder.transform(valid_df['intent'])
    test_df['encoded_intent'] = label_encoder.transform(test_df['intent'])
    
    print("\n合并后的数据统计:")
    print(f"总数据量: {len(combined_df)}")
    print(f"意图类别: {label_encoder.classes_.tolist()}")
    print(f"数据集划分:\n训练集: {len(train_df)}\n验证集: {len(valid_df)}\n测试集: {len(test_df)}")
    
    return train_df, valid_df, test_df, label_encoder


def train_model(model, train_dataloader, valid_dataloader, config):
    """训练模型并保存最佳结果"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    best_accuracy = 0
    
    for epoch in range(config.epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_accuracy = 0
        
        for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{config.epochs}'):
            batch = {k: v.to(config.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            train_loss += loss.item()
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
        
        print(f'\nEpoch {epoch + 1}:')
        print(f'训练集平均损失: {avg_train_loss:.4f}, 训练集平均准确率: {avg_train_accuracy:.4f}')
        print(f'验证集平均损失: {avg_valid_loss:.4f}, 验证集平均准确率: {avg_valid_accuracy:.4f}')
        
        # 保存最佳模型
        if avg_valid_accuracy > best_accuracy:
            best_accuracy = avg_valid_accuracy
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"保存新的最佳模型，验证准确率: {best_accuracy:.4f}")
            
    return model


def main():
    config = Config()
    
    # 获取数据文件
    file_paths = get_dataset_files('dataset')
    if not file_paths:
        return
    
    # 加载和预处理数据
    data = combine_intent_files(config, file_paths)
    if data is None:
        return
        
    train_df, valid_df, test_df, label_encoder = data
    
    # 保存label_encoder
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print("已保存标签编码器到 label_encoder.pkl")
    
    # 初始化tokenizer和模型
    tokenizer = BertTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name, 
        num_labels=len(label_encoder.classes_)
    )
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
    model = train_model(model, train_dataloader, valid_dataloader, config)
    
    # 测试集评估
    print("\n在测试集上进行评估:")
    model.eval()
    test_accuracy = 0
    test_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc='Testing'):
            batch = {k: v.to(config.device) for k, v in batch.items()}
            outputs = model(**batch)
            test_loss += outputs.loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            test_accuracy += (predictions == batch["labels"]).float().mean().item()
    
    avg_test_loss = test_loss / len(test_dataloader)
    avg_test_accuracy = test_accuracy / len(test_dataloader)
    print(f'测试集平均损失: {avg_test_loss:.4f}')
    print(f'测试集平均准确率: {avg_test_accuracy:.4f}')
    print("\n训练完成，模型已保存为 'best_model.pt'，标签编码器已保存为 'label_encoder.pkl'")


if __name__ == "__main__":
    main()
    
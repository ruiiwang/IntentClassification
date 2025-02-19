import pandas as pd
import numpy as np
import os
import time
import glob
from sklearn.preprocessing import LabelEncoder
from gensim.models.fasttext import FastText
from sklearn.linear_model import LogisticRegression


class Config:
    batch_size = 32
    epochs = 25
    train_ratio = 0.7
    valid_ratio = 0.15
    test_ratio = 0.15
    vector_size = 200
    window = 3
    min_count = 1


def get_dataset_files(dataset_dir):
    """获取数据集文件夹中的所有validated xlsx文件"""
    if not os.path.exists(dataset_dir):
        print(f"错误: 文件夹 '{dataset_dir}' 不存在")
        return []

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
            base_name = os.path.basename(file_path)
            intent_name = base_name.split('_validated_')[0]

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

    # 使用copy()创建真实副本
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
    print(f"意图类别: {label_encoder.classes_.tolist()}")
    print(f"数据集划分:\n训练集: {len(train_df)}\n验证集: {len(valid_df)}\n测试集: {len(test_df)}")

    return train_df, valid_df, test_df, label_encoder


def train_fasttext_model(train_texts, config):
    """训练FastText模型"""
    # 将文本转换为词列表
    sentences = [text.split() for text in train_texts]

    # 训练FastText模型
    model = FastText(
        sentences,
        vector_size=config.vector_size,
        window=config.window,
        min_count=config.min_count,
        epochs=config.epochs,
        workers=4
    )

    return model


def get_text_vector(text, model):
    """获取文本的向量表示"""
    words = text.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if not word_vectors:
        return np.zeros(model.vector_size)
    return np.mean(word_vectors, axis=0)


def train_classifier(train_vectors, train_labels, valid_vectors, valid_labels):
    """训练分类器"""
    classifier = LogisticRegression(max_iter=1000, multi_class='ovr')
    classifier.fit(train_vectors, train_labels)

    # 评估
    train_acc = classifier.score(train_vectors, train_labels)
    valid_acc = classifier.score(valid_vectors, valid_labels)

    print(f"训练集准确率: {train_acc:.4f}")
    print(f"验证集准确率: {valid_acc:.4f}")

    return classifier


def predict_text(text, fasttext_model, classifier, label_encoder, top_k=3):
    """预测文本意图"""
    start_time = time.perf_counter()

    # 获取文本向量
    text_vector = get_text_vector(text, fasttext_model)

    # 预测概率
    probabilities = classifier.predict_proba([text_vector])[0]

    # 获取top-k结果
    top_indices = np.argsort(probabilities)[-top_k:][::-1]

    results = [{
        'label': label_encoder.inverse_transform([idx])[0],
        'probability': probabilities[idx]
    } for idx in top_indices]

    pred_time = (time.perf_counter() - start_time) * 1000

    return results, pred_time


def main():
    config = Config()

    # 1. 获取数据文件
    file_paths = get_dataset_files('dataset')
    if not file_paths:
        return

    # 2. 加载和预处理数据
    data = combine_intent_files(config, file_paths)
    if data is None:
        return

    train_df, valid_df, test_df, label_encoder = data

    # 3. 训练FastText模型
    print("\n开始训练FastText模型...")
    fasttext_model = train_fasttext_model(train_df['text'], config)

    # 4. 准备向量
    train_vectors = np.array([get_text_vector(text, fasttext_model) for text in train_df['text']])
    valid_vectors = np.array([get_text_vector(text, fasttext_model) for text in valid_df['text']])
    test_vectors = np.array([get_text_vector(text, fasttext_model) for text in test_df['text']])

    # 5. 训练分类器
    print("\n训练分类器...")
    classifier = train_classifier(
        train_vectors, train_df['encoded_intent'],
        valid_vectors, valid_df['encoded_intent']
    )

    # 6. 测试集评估
    test_acc = classifier.score(test_vectors, test_df['encoded_intent'])
    print(f"测试集准确率: {test_acc:.4f}")

    # 7. 交互式预测
    print("\n开始交互式预测 (输入 'quit' 退出):")
    while True:
        user_input = input("\n请输入文本: ").strip()

        if user_input.lower() == 'quit':
            break

        if user_input:
            predictions, pred_time = predict_text(
                user_input, fasttext_model, classifier,
                label_encoder
            )
            print(f"\n预测时间: {pred_time:.2f}ms")
            print("预测结果:")
            for pred in predictions:
                print(f"意图: {pred['label']}, 置信度: {pred['probability']:.4f}")
        else:
            print("请输入有效文本!")

    # 8. 保存模型
    fasttext_model.save('fasttext_model.bin')


if __name__ == "__main__":
    main()

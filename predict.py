import torch
from transformers import BertTokenizer, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForMaskedLM
from sklearn.preprocessing import LabelEncoder
import time
import pickle
from config import BaseConfig, ModelConfig


def predict_text(text, model, tokenizer, label_encoder, config):
    """
    预测文本意图
    text (str): 输入文本
    model: 加载的模型
    tokenizer: 分词器
    label_encoder: 标签编码器
    config: 配置
    返回:
    list: 预测结果列表
    float: 预测耗时（毫秒）
    """
    model.eval()

    # 记录开始时间
    start_time = time.perf_counter()

    # 准备输入
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=config.max_length,
        return_tensors="pt"
    )
    inputs = {k: v.to(config.device) for k, v in inputs.items()}

    # 进行预测
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)

        # 获取所有类别的概率
        all_probs = probabilities[0].cpu().numpy()

        # 获取最高概率及其对应的类别
        max_prob, predicted = torch.max(probabilities, dim=1)
        max_prob = max_prob.item()
        predicted_class = predicted.item()

        # 获取预测的类别标签
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]

        # 计算预测时间
        pred_time = (time.perf_counter() - start_time) * 1000

        # 如果是 INVALID 类别或置信度不足
        if predicted_label == 'INVALID' or max_prob < config.confidence_threshold:
            return [{
                'label': 'UNKNOWN',
                'probability': 0.0,
                'reason': '置信度不足' if max_prob < config.confidence_threshold else '无效意图'
            }], pred_time

        # 获取所有高于阈值的正类别预测
        results = []
        for idx, prob in enumerate(all_probs):
            if prob >= config.confidence_threshold:
                label = label_encoder.inverse_transform([idx])[0]
                if label != 'INVALID':  # 只添加非 INVALID 的预测
                    results.append({
                        'label': label,
                        'probability': float(prob)
                    })

        # 按概率降序排序
        results.sort(key=lambda x: x['probability'], reverse=True)

        # 如果没有任何有效预测，返回 UNKNOWN
        if not results:
            return [{
                'label': 'UNKNOWN',
                'probability': 0.0,
                'reason': '无有效预测结果'
            }], pred_time

        return results, pred_time


def load_model_and_tokenizer(config):
    """
    加载模型和分词器
    config: 配置
    返回:
    tuple: (model, tokenizer, label_encoder)
    """
    try:
        # 加载标签编码器
        with open(config.label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)

        # 加载分词器
        if config.model_config['tokenizer'] == 'BertTokenizer':
            tokenizer = BertTokenizer.from_pretrained(config.model_name)
        elif config.model_config['tokenizer'] == 'AutoTokenizer':
            tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        # 初始化模型
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=len(label_encoder.classes_),
            ignore_mismatched_sizes=True
        )

        # 加载训练好的模型权重
        model.load_state_dict(torch.load(config.model_path, map_location=config.device))
        model.to(config.device)

        return model, tokenizer, label_encoder

    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"找不到必要的模型文件或标签编码器文件。请确保 '{config.model_path}' 和 '{config.label_encoder_path}' 文件存在。") from e
    except Exception as e:
        raise Exception(f"加载模型时发生错误: {str(e)}") from e

class PredictConfig:
    selected_model = 'model1'
    # 根据选择获取模型配置
    model_config = ModelConfig.MODELS[selected_model]
    model_name = model_config['name']
    model_dir = f"model/model1-0421"
    # 模型相关路径
    model_path = f"{model_dir}/best_model.pt"
    label_encoder_path = f"{model_dir}/label_encoder.pkl"
    # 基础配置
    max_length = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    confidence_threshold = 0.6

def main():
    config = PredictConfig()

    try:
        print("正在加载模型...")
        model, tokenizer, label_encoder = load_model_and_tokenizer(config)
        print("模型加载完成")

        # 交互式预测
        print("\n开始交互式预测 (输入 'quit' 退出):")
        while True:
            try:
                # 获取用户输入
                user_input = input("\n请输入文本: ").strip()
                # 检查退出条件
                if user_input.lower() == 'quit':
                    break
                # 检查输入是否为空
                if not user_input:
                    print("请输入有效文本")
                    continue
                # 进行预测
                predictions, pred_time = predict_text(
                    user_input, model, tokenizer, label_encoder, config
                )
                # 打印预测结果
                print(f"\n预测时间: {pred_time:.2f}ms")
                print("预测结果:")
                for pred in predictions:
                    if 'reason' in pred:
                        print(f"意图: {pred['label']}, 原因: {pred['reason']}")
                    else:
                        print(f"意图: {pred['label']}, 置信度: {pred['probability']:.4f}")

            except KeyboardInterrupt:
                print("\n检测到中断信号，正在退出...")
                break
            except Exception as e:
                print(f"预测过程中发生错误: {str(e)}")
                print("请重试或输入 'quit' 退出")

    except Exception as e:
        print(f"程序发生错误: {str(e)}")
        return


if __name__ == "__main__":
    main()

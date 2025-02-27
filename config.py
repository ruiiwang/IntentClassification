import torch
import os


class ModelConfig:
    # 定义可用的模型配置
    MODELS = {
        'model1': {
            'name': 'huawei-noah/TinyBERT_General_4L_312D',
            'tokenizer': 'BertTokenizer',
            'dir': 'model1'
        },
        'model2': {
            'name': 'microsoft/deberta-v3-small',
            'tokenizer': 'AutoTokenizer',
            'model_type': 'AutoModel',
            'dir': 'model2'
        },
        'model3': {
            'name': 'MilyaShams/SmolLM2-135M-Instruct-Reward',
            'tokenizer': 'AutoTokenizer',
            'dir': 'model3',
        },
    }


class BaseConfig:
    selected_model = 'model1'
    # 根据选择获取模型配置
    model_config = ModelConfig.MODELS[selected_model]
    model_name = model_config['name']
    model_dir = f"model/{model_config['dir']}"
    # 模型相关路径
    model_path = f"{model_dir}/best_model.pt"
    label_encoder_path = f"{model_dir}/label_encoder.pkl"
    # 基础配置
    max_length = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    confidence_threshold = 0.6


class TrainConfig(BaseConfig):
    """训练时的额外配置"""
    batch_size = 32
    epochs = 30
    learning_rate = 2e-5
    warmup_steps = 0
    train_ratio = 0.7
    valid_ratio = 0.15
    test_ratio = 0.15
    metrics_dir = f"{BaseConfig.model_dir}/metrics"

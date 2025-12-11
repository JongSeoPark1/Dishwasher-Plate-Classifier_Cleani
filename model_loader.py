import torch
import torch.nn as nn
from torchvision import models

def get_model(model_name, num_classes=2, device='cpu'):
    """
    모델 이름을 받아 껍데기(아키텍처)를 생성하여 반환합니다.
    학습된 가중치는 로드하지 않습니다 (weights=None).
    """
    model_name = model_name.lower()
    model = None

    if 'mobile' in model_name:
        model = models.mobilenet_v3_large(weights=None)
        # Classifier 구조 변경 (Dropout + Linear)
        model.classifier[3] = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(model.classifier[0].out_features, num_classes)
        )
    
    elif 'efficient' in model_name:
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )
        
    elif 'resnet' in model_name:
        model = models.resnet18(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(model.fc.in_features, num_classes)
        )
    
    if model:
        return model.to(device)
    else:
        raise ValueError(f"지원하지 않는 모델명입니다: {model_name}")
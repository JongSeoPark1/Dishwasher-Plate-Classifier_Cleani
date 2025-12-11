# train_base.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from dataset import get_dataloaders
from model_loader import get_model

# 1. 설정
DATA_DIR = './data/pretrain'      # 👈 1차 학습 데이터 경로
SAVE_DIR = './saved_models'       # 👈 1차 모델 저장 경로
MODEL_NAMES = ['mobilenet', 'efficientnet', 'resnet']
NUM_EPOCHS = 20  # 충분히 학습
LR = 1e-4        # 일반적인 학습률

def main():
    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")

    # 데이터셋 로드
    dataloaders, dataset_sizes, class_names = get_dataloaders(DATA_DIR)
    print(f"🚀 [1차 학습] 데이터: {DATA_DIR} | 클래스: {class_names}")

    os.makedirs(SAVE_DIR, exist_ok=True)

    for name in MODEL_NAMES:
        print(f"\n🔥 [{name}] 1차 학습 시작...")
        
        # 모델 생성 (ImageNet 가중치 사용하려면 model_loader 수정 필요, 지금은 깡통 or 기본)
        model = get_model(name, len(class_names), device) 
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=LR)

        best_acc = 0.0
        best_wts = copy.deepcopy(model.state_dict())

        for epoch in range(NUM_EPOCHS):
            for phase in ['train', 'val']:
                if phase == 'train': model.train()
                else: model.eval()

                running_corrects = 0
                for inputs, labels in dataloaders[phase]:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    running_corrects += torch.sum(preds == labels.data)
                
                epoch_acc = running_corrects.float() / dataset_sizes[phase]
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_wts = copy.deepcopy(model.state_dict())
            
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} Done.")

        # 저장
        torch.save(best_wts, os.path.join(SAVE_DIR, f"{name}_best.pth"))
        print(f"✅ [{name}] 1차 학습 완료 및 저장됨.")

if __name__ == '__main__':
    main()
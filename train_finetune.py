# train_finetune.py
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model_loader import get_model

# 1. 설정 (경로만 다릅니다!)
DATA_DIR = './data/finetune'      # 👈 파인튜닝 데이터 경로
LOAD_DIR = './saved_models'       # 👈 1차 학습된 모델 가져오는 곳
SAVE_DIR = './finetuned_models'   # 👈 최종 모델 저장하는 곳
MODEL_NAMES = ['mobilenet', 'efficientnet', 'resnet']
NUM_EPOCHS = 30
LR = 1e-5        # 👈 아주 작은 학습률 (지식 보존)

def main():
    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")

    # 데이터셋 로드
    if not os.path.exists(DATA_DIR):
        print(f"❌ 파인튜닝 데이터가 없습니다: {DATA_DIR}")
        return
    dataloaders, dataset_sizes, class_names = get_dataloaders(DATA_DIR)
    print(f"♻️ [파인 튜닝] 데이터: {DATA_DIR} | 클래스: {class_names}")

    os.makedirs(SAVE_DIR, exist_ok=True)

    for name in MODEL_NAMES:
        print(f"\n🔄 [{name}] 파인튜닝 시작...")
        
        # 모델 생성
        model = get_model(name, len(class_names), device)
        
        # ★ 1차 학습된 가중치 불러오기
        load_path = os.path.join(LOAD_DIR, f"{name}_best.pth")
        if os.path.exists(load_path):
            model.load_state_dict(torch.load(load_path, map_location=device))
            print(f"  ✅ 1차 학습 모델 로드 성공: {load_path}")
        else:
            print(f"  ⚠️ 1차 학습 파일이 없어 처음부터 학습합니다.")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)

        best_acc = 0.0
        best_wts = copy.deepcopy(model.state_dict())

        # (이하 학습 루프는 동일, 생략 가능하지만 복붙 편의를 위해 유지)
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
            
            if (epoch+1) % 5 == 0: print(f"Epoch {epoch+1}/{NUM_EPOCHS} 진행 중...")

        # 최종 저장
        torch.save(best_wts, os.path.join(SAVE_DIR, f"{name}_best.pth"))
        print(f"  💾 파인튜닝 완료 및 저장됨.")

if __name__ == '__main__':
    main()
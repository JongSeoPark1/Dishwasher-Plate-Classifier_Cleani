import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model_loader import get_model  # model_loader.py가 같은 폴더에 있어야 함

# -------------------------------------------------------------------------
# 1. 설정
# -------------------------------------------------------------------------
MODEL_DIR = './finetuned_models'   # 파인튜닝된 모델이 있는 곳
DATA_DIR = './data'                # 데이터 루트 (안에 test 폴더가 있어야 함)
CLASS_NAMES = ['cleaned', 'dirty'] # 클래스 이름 (0: cleaned, 1: dirty)

# 테스트용 이미지 변환 (검증 때와 동일)
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# -------------------------------------------------------------------------
# 2. 모델 로드 함수
# -------------------------------------------------------------------------
def load_ensemble_models(device):
    models_dict = {}
    
    # 모델 폴더 확인
    if not os.path.exists(MODEL_DIR):
        print(f"❌ 모델 폴더가 없습니다: {MODEL_DIR}")
        return {}

    # 폴더 내의 .pth 파일들을 찾아서 로드
    for f in sorted(os.listdir(MODEL_DIR)):
        if f.endswith('.pth') or f.endswith('.pt'):
            # 파일명에서 아키텍처 이름 추론
            if 'mobile' in f: arch = 'mobilenet'
            elif 'efficient' in f: arch = 'efficientnet'
            elif 'resnet' in f: arch = 'resnet'
            else: continue

            print(f"🔄 로딩 중: {f} ({arch})...")
            try:
                # 껍데기 생성
                model = get_model(arch, len(CLASS_NAMES), device)
                
                # 가중치 로드
                weight_path = os.path.join(MODEL_DIR, f)
                checkpoint = torch.load(weight_path, map_location=device)
                
                # state_dict 처리 (저장 방식에 따라 다를 수 있음)
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                elif isinstance(checkpoint, dict):
                    model.load_state_dict(checkpoint)
                else:
                    model.load_state_dict(checkpoint.state_dict())

                model.eval()
                models_dict[f] = model
                print(f"  ✅ 로드 성공!")
            except Exception as e:
                print(f"  ❌ 로드 실패 ({f}): {e}")

    return models_dict

# -------------------------------------------------------------------------
# 3. 예측 함수 (앙상블)
# -------------------------------------------------------------------------
def predict_image(image_path, models_dict, device):
    try:
        img = Image.open(image_path).convert('RGB')
    except:
        return None, 0, None # 이미지 못 읽음

    img_tensor = val_transform(img).unsqueeze(0).to(device)
    avg_probs = torch.zeros(1, len(CLASS_NAMES)).to(device)

    # 앙상블: 모든 모델의 예측 확률을 더함
    with torch.no_grad():
        for model in models_dict.values():
            outputs = model(img_tensor)
            probs = F.softmax(outputs, dim=1)
            avg_probs += probs
    
    # 평균 계산
    avg_probs /= len(models_dict)
    
    # 가장 높은 확률의 클래스 선택
    max_prob, idx = torch.max(avg_probs, 1)
    
    # Dirty(인덱스 1)일 확률 (0~100%)
    dirty_prob = avg_probs[0][1].item() * 100
    
    return CLASS_NAMES[idx.item()], dirty_prob

# -------------------------------------------------------------------------
# 4. 등급 판별 함수
# -------------------------------------------------------------------------
def get_grade(dirty_prob):
    if dirty_prob >= 80: return "A" # 매우 더러움
    elif dirty_prob >= 60: return "B"
    elif dirty_prob >= 40: return "C"
    elif dirty_prob >= 20: return "D"
    else: return "E" # 매우 깨끗함

# -------------------------------------------------------------------------
# 5. 메인 실행 함수 (여기가 빠져 있었습니다!)
# -------------------------------------------------------------------------
def main():
    # 디바이스 설정
    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    print(f"🚀 사용 디바이스: {device}")

    # 모델 로드
    models = load_ensemble_models(device)
    if not models:
        print("❌ 로드된 모델이 없습니다. 'finetuned_models' 폴더를 확인하세요.")
        return

    print(f"\n{'File Name':<30} | {'Pred':<10} | {'Dirty %':<10} | {'Grade':<5} | {'Result'}")
    print("-" * 85)

    # 테스트 데이터 경로 설정
    test_dir = os.path.join(DATA_DIR, 'test') # ./data/test

    if not os.path.exists(test_dir):
        print(f"❌ 테스트 폴더가 없습니다: {test_dir}")
        print("👉 'data/test/cleaned' 와 'data/test/dirty' 폴더를 만들고 사진을 넣어주세요.")
        return

    total, correct = 0, 0

    # cleaned, dirty 폴더를 각각 돌면서 테스트
    for label in CLASS_NAMES:
        folder_path = os.path.join(test_dir, label)
        if not os.path.exists(folder_path): continue
        
        # 이미지 파일만 골라내기
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for file in files:
            total += 1
            path = os.path.join(folder_path, file)
            
            # 예측 수행
            pred, dirty_prob = predict_image(path, models, device)
            
            if pred is None: continue # 이미지 로드 실패시 패스

            # 정답 여부 체크
            is_correct = (pred == label)
            if is_correct: correct += 1
            
            grade = get_grade(dirty_prob)
            
            # 결과 출력
            mark = '✅' if is_correct else '❌'
            print(f"{file:<30} | {pred:<10} | {dirty_prob:5.1f}%    | {grade:^5} | {mark} (Ans: {label})")

    print("-" * 85)
    if total > 0:
        print(f"📊 최종 정확도: {correct/total*100:.2f}% ({correct}/{total})")
    else:
        print("⚠️ 테스트할 이미지가 발견되지 않았습니다.")

if __name__ == '__main__':
    main()

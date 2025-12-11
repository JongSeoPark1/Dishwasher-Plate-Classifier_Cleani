# inference.py 수정 부분

# ... (위쪽 코드는 그대로)

def main():
    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")

    models = load_ensemble_models(device)
    if not models: return

    print(f"\n{'File Name':<30} | {'Pred':<10} | {'Dirty %':<10} | {'Grade':<5} | {'Answer'}")
    print("-" * 80)

    # 👇 [수정할 부분] 테스트 경로를 'data/test'로 지정
    test_dir = './data/test' 

    if not os.path.exists(test_dir):
        print(f"❌ 테스트 경로가 없습니다: {test_dir}")
        print("👉 'data' 폴더 안에 'test' 폴더를 만들고 사진을 넣어주세요!")
        return

    total, correct = 0, 0

    # cleaned 폴더와 dirty 폴더를 각각 돌면서 예측
    for label in CLASS_NAMES:
        folder_path = os.path.join(test_dir, label)
        if not os.path.exists(folder_path): continue
        
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                total += 1
                path = os.path.join(folder_path, file)
                pred, dirty_prob = predict_image(path, models, device)
                
                is_correct = (pred == label)
                if is_correct: correct += 1
                grade = get_grade(dirty_prob)
                
                print(f"{file:<30} | {pred:<10} | {dirty_prob:5.1f}%    | {grade:^5} | {label} {'✅' if is_correct else '❌'}")

    print("-" * 80)
    if total > 0:
        print(f"📊 Final Accuracy: {correct/total*100:.2f}% ({correct}/{total})")
    else:
        print("⚠️ 테스트할 이미지가 없습니다.")

if __name__ == '__main__':
    main()
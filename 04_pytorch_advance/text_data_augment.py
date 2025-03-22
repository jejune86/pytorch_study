# 텍스트 데이터 증강
# 증강 방법
## 삽입, 삭제, 교체, 대체, 생성, 반의어, 만춤법 교정, 역번역 ...
# NLPAUG 라이브러리 사용


# 삽입 및 삭제
import nlpaug.augmenter.word as naw

texts = [
    "Those who can imagine anything, can create the impossible.",
    "We can only see a short distance ahead, but we can see plenty there that needs to be done.",
    "If a machine is expected to be infallible, it cannot also be intelligent.",
]

aug = naw.ContextualWordEmbsAug(model_path="bert-base-uncased", action="insert")
augmented_texts = aug.augment(texts)

for text, augmented in zip(texts, augmented_texts):
    print(f"Original: {text}")
    print(f"Augmented: {augmented}")
    print()
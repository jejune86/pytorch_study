# 텍스트 데이터 증강
# 증강 방법
## 삽입, 삭제, 교체, 대체, 생성, 반의어, 만춤법 교정, 역번역 ...
# NLPAUG 라이브러리 사용


# 삽입
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
    
    
# 삭제
aug = naw.RandomWordAug(action="delete")
augmented_texts = aug.augment(texts)

for text, augmented in zip(texts, augmented_texts):
    print(f"Original: {text}")
    print(f"Augmented: {augmented}")
    print()
    
    
# 교체 (단어나 문자의 위치 교환)
aug = naw.RandomWordAug(action="swap")
augmented_texts = aug.augment(texts)

for text, augmented in zip(texts, augmented_texts):
    print(f"Original: {text}")
    print(f"Augmented: {augmented}")
    print()
    
    
# 대체 (단어나 문자를 임의의 단어나 문자로 바꾸거나, 동의어로 변경)
aug = naw.SynonymAug(aug_src="wordnet")
augmented_texts = aug.augment(texts)

for text, augmented in zip(texts, augmented_texts):
    print(f"Original: {text}")
    print(f"Augmented: {augmented}")
    print()

        
# 역번역 (Bake-translation), 특정 언어로 번역한 다음 다시 본래의 언어로 번역

back_translation_aug = naw.BackTranslationAug(
    from_model_name='facebook/wmt19-en-de', 
    to_model_name='facebook/wmt19-de-en'
)
augmented_texts = back_translation_aug.augment(texts)

for text, augmented in zip(texts, augmented_texts):
    print(f"Original: {text}")
    print(f"Augmented: {augmented}")
    print()
    

# 기타 증강
## 오타 오류 증강
aug = naw.KeyboardAug()

## 무작위 문자 증강
aug = naw.RandomCharAug(action) # action : 'insert', 'substitute', 'swap', 'delete'

## 무작위 단어 증강
aug = naw.RandomWordAug(action)

## 동의어 증강
aug = nac.RandomCharAug(action)

## 예약어 증강
naw.SynonymAug(aug_src="wordnet") # 대체

## 철자 오류 증강
naw.ReservedAug(reserved_tokens) # reserved_tokens : 예약어 리스트

## 상황별 단어 임베딩 증강
naw.SpellingAug()

## 역번역 증강
naw.BackTranslationAug(from_model_name, to_model_name)

## 문장 요약 증강
naw.AbstSummAug(model_path="t5-base")
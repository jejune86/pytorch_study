from konlpy.tag import Okt

okt = Okt()
sentence = "무엇이든 상상할 수 있는 사람은 무엇이든 만들어 낼 수 있다."

nouns = okt.nouns(sentence)
phrases = okt.phrases(sentence)
morphs = okt.morphs(sentence)
pos = okt.pos(sentence)

print("명사 추출 : ", nouns)
print("구 추출 : ", phrases)
print("형태소 추출 : ", morphs)
print("품사 태깅 : ", pos)


from konlpy.tag import Kkma # 꼬꼬마 형태소 분석기

kkma = Kkma()

nouns = kkma.nouns(sentence)
sentences = kkma.sentences(sentence)
morphs = kkma.morphs(sentence)
pos = kkma.pos(sentence)

print("명사 추출 : ", nouns)
print("문장 추출 : ", phrases)
print("형태소 추출 : ", morphs)
print("품사 태깅 : ", pos)

# Okt는 총 19개의 품사를 구분
# kkma는 총 56개의 품사를 구분
## 더 자세한 분석 가능하지만, 품사 태깅 소요시간 길고, 성능 저하 가능

import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from nltk import tokenize
sentence = "Those who can imagein anything, can create the impossible."

word_tokens = tokenize.word_tokenize(sentence)
sent_tokens = tokenize.sent_tokenize(sentence)
pos = nltk.pos_tag(word_tokens)

print(word_tokens)
print(sent_tokens)
print(pos)

import spacy

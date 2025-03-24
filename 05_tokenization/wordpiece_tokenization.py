# BPE와 유사한 방법이지만, 빈도 기반이 아닌 확률 기반으로 글자 쌍 병합
# 새로운 하위 단어를 생성할 때 이전 하위 단어와 함께 나타날 확률을 계산해 가장 높은 확률을 가진 하위 단어 선택

# score = f(x,y) / f(x) * f(y) : x, y가 합쳐졌을 때의 확률 / x, y가 각각 나타날 확률의 곱


# tokenizers 사용

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.normalizers import Sequence, NFD, Lowercase
from tokenizers.pre_tokenizers import Whitespace

# tokenizer = Tokenizer(WordPiece())

# tokenizer.normalizer = Sequence([NFD(), Lowercase()])
# tokenizer.pre_tokenizer = Whitespace() 

# tokenizer.train(["./datasets/corpus.txt"])
# tokenizer.save("./datasets/petition_wordpiece.json") 

from tokenizers import Tokenizer
from tokenizers.decoders import WordPiece as WordPieceDecoder

tokenizer = Tokenizer.from_file("./datasets/petition_wordpiece.json")
tokenizer.decoder = WordPieceDecoder()

sentence = "안녕하세요, 토크나이저가 잘 학습되었군요!"
sentences = ["이렇게 입력값을 리스트로 받아서,", "쉽게 토크나이저를 사용할 수 있답니다"]
encoded_sentence = tokenizer.encode(sentence)
encoded_sentences = tokenizer.encode_batch(sentences)

print("인코더 형식 : ", type(encoded_sentence))

print("단일 문장 토큰화 : ", encoded_sentence.tokens)
print("다중 문장 토큰화 : ", [enc.tokens for enc in encoded_sentences])

print("단일 문장 정수 인코딩 : ", encoded_sentence.ids)
print("다중 문장 정수 인코딩 : ", [enc.ids for enc in encoded_sentences])

print("정수 인코딩에서 문장 변환 : ", tokenizer.decode(encoded_sentence.ids))
from Korpora import Korpora

# 청와대 청원 데이터 다운로드드
corpus = Korpora.load("korean_petitions")
dataset = corpus.train
petition = dataset[0]

print("청원 시작일 : ", petition.begin)
print("청원 종료일 : ", petition.end)
print("청원 동의 수 : ", petition.num_agree)
print("청원 범주 : ", petition.category)
print("청원 제목 : ", petition.title)
print("청원 본문 : ", petition.text[:30])

# 학습 데이터세트 생성
petitions = corpus.get_all_texts()
with open("./datasets/corpus.txt", "w", encoding="utf-8") as f:
    for petition in petitions:
        f.write(petition + "\n")


# SentencePiece Tokenizer model 학습
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor

SentencePieceTrainer.Train(

    "--input=./datasets/corpus.txt\
    --model_prefix=petition_bpe\
    --vocab_size=8000 model_type=bpe"
    
    #input : 학습 데이터세트
    #model_prefix : 모델 파일 이름
    #vocab_size : 단어 집합 크기
)

tokenizer = SentencePieceProcessor()
tokenizer.load("petition_bpe.model")

sentence = "안녕하세요, 토크나이저가 잘 학습되었군요!"
sentences = ["이렇게 입력값을 리스트로 받아서,", "쉽게 토크나이저를 사용할 수 있답니다"]

tokenized_sentence = tokenizer.encode_as_pieces(sentence)
tokenized_sentences = tokenizer.encode_as_pieces(sentences)

print("단일 문장 토큰화 : ", tokenized_sentence)
print("다중 문장 토큰화 : ", tokenized_sentences)

encoded_sentence = tokenizer.encode_as_ids(sentence)
encoded_sentences = tokenizer.encode_as_ids(sentences)

print("단일 문장 정수 인코딩 : ", encoded_sentence)
print("다중 문장 정수 인코딩 : ", encoded_sentences)

decode_ids = tokenizer.decode_ids(encoded_sentences)
decode_pieces = tokenizer.decode_pieces(encoded_sentences)

print("정수 인코딩에서 문장 변환 : ", decode_ids)
print("하위 단어 토큰에서 문장 변환 : ", decode_pieces)

vocab = {idx: tokenizer.id_to_piece(idx) for idx in range(tokenizer.get_piece_size())}
print(list(vocab.items())[:5])
print("vocab size : ", len(vocab))
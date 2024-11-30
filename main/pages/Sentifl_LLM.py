import streamlit as st
import pandas as pd
import os
import requests
import json

st.set_page_config(page_title="Sentifl LLM", page_icon="☕", layout="wide")

st.title("Sentifl LLM")
st.divider()
st.subheader("소개")
st.markdown("""
    Sentifl LLM은 감정기반 BGM 생성 AI 입니다.\n
    이 AI는 한글로 작성된 글을 요약하고, 그 글의 감정을 파악한 후 그 감정에 맞는 BGM을 생성해줍니다.\n
    사용된 AI는 "gogamza/kobart-summarization"와 자체적으로 fine-tuning한 "meta/Llama-3.2-1B", "facebook/musicgen-melody"입니다.\n
    사용한 데이터셋은 "AI Hub"에서 제공하는 "한국어 감정 데이터셋"입니다.\n
    이 데이터셋은 감정을 공포, 놀람, 분노, 슬픔, 중립, 행복, 혐오로 총 7가지로 분류하고 있습니다.\n
    세 개의 AI 중 가장 공들이고 핵심이 되는 AI는 감정 분석 AI 입니다.\n
    그래서 감정 분석 AI를 만든 과정을 사용한 데이터를 전처리부터 학습까지 코드와 함께 진행하겠습니다.
""")
@st.cache_data
def get_data():
    return pd.read_csv("https://shyo2-public.s3.ap-northeast-2.amazonaws.com/emotion_data.csv", encoding="CP949")

rawdata = get_data()
emotion_data = rawdata[["Sentence", "Emotion"]]

st.sidebar.title("감정 데이터 선택")
filter = st.sidebar.radio("감정을 선택하세요", ["공포", "놀람", "분노", "슬픔", "중립", "행복", "혐오"])
filtered_data = emotion_data[emotion_data["Emotion"] == filter]

st.sidebar.download_button("Download Total Data", data=emotion_data.to_csv(), file_name="filtered_data.csv", mime="text/csv")

st.divider()
st.subheader("데이터")
st.write("{} 데이터의 갯수 : {}개".format(filter,filtered_data.shape[0]))
st.dataframe(filtered_data)

st.divider()

st.subheader("데이터 전처리")

st.code("""
# 토큰화
from konlpy.tag import Okt
tokenizer = Okt()
data['Tokenized'] = data['Sentence'].apply(lambda text: tokenizer.morphs(text))
""")
st.image("https://shyo2-public.s3.ap-northeast-2.amazonaws.com/static/Tokenized.png")
st.code("""
# 불용어 제거 및 단어 집합 생성
stopword = pd.read_csv(stopword_path, encoding="CP949")
vocab = {}
preprocessed_sentences = []

for tokenized_sentence in rawdata['Tokenized']:
    result = []
    for word in tokenized_sentence:
        if word not in stopword:
            result.append(word)
        if word not in vocab:
            vocab[word] = 0
        vocab[word] += 1
    preprocessed_sentences.append(result)
rawdata['Preprocessed'] = preprocessed_sentences
Data = rawdata[['Preprocessed','Emotion']]
""")
st.image("https://shyo2-public.s3.ap-northeast-2.amazonaws.com/static/Preprocessed.png")
st.code("""
# 단어의 빈도 파악
import numpy as np

vocab_value_list = list(vocab.values())

print(' mean:', np.mean(vocab_value_list),
'\n std:', np.std(vocab_value_list),
'\n min:', np.min(vocab_value_list),
'\n median:', np.median(vocab_value_list),
'\n max:', np.max(vocab_value_list),
'\n percentile:', np.percentile(vocab_value_list,[0,25,50,75,100]))
""")
st.image("https://shyo2-public.s3.ap-northeast-2.amazonaws.com/static/Vocab_Num.png")
st.code("""
# 단어의 빈도 분포 시각화
import matplotlib.pyplot as plt

plt.boxplot(vocab_value_list)
plt.show()
""")
st.image("https://shyo2-public.s3.ap-northeast-2.amazonaws.com/static/Vocab_Box.png")
st.code("""
# 임계값으로 희귀 단어 파악
threshold = 8
total_cnt = len(vocab)
rare_cnt = 0
total_freq = 0
rare_freq = 0

vocab_key_list = list(vocab.keys())
vocab_value_list = list(vocab.values())

for i in range(total_cnt):
    word = vocab_key_list[i]
    value = vocab_value_list[i]
    total_freq = total_freq + value

    if (value < threshold):
        rare_cnt += 1
        rare_freq = rare_freq + value

print('단어 집합의 크기 : ', total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수 : %s'%(threshold -1, rare_cnt))
print('단어 집합에서 희귀 단어를 제외시킬 경우의 단어 집합의 크기 %s'%(total_cnt - rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)
""")
st.image("https://shyo2-public.s3.ap-northeast-2.amazonaws.com/static/Del_Rare.png")
st.code("""
# 등장 빈도가 8번 이상인 단어들만 사용
vocab_df = pd.DataFrame({'keys': vocab_key_list, 'Values':vocab_value_list})
main_vocab_df = vocab_df[vocab_df['Values'] >= 8].reset_index()
main_vocab_df
""")
st.image("https://shyo2-public.s3.ap-northeast-2.amazonaws.com/static/Main_Vocab.png")
st.code("""
# 다시 딕셔너리 형태로 변환
mv_key = list(main_vocab_df['keys'])
mv_value = list(range(5091))
main_vocab = { key: value for key, value in zip(mv_key, mv_value)}
""")
st.code("""
# 인코딩
encodingdata = []
for sentence in Data['Preprocessed']:
    result = [main_vocab[word] for word in sentence if word in main_vocab]
    encodingdata.append(result)
Data['Encode'] = encodingdata
Data.head()
""")
st.image("https://shyo2-public.s3.ap-northeast-2.amazonaws.com/static/Encode.png")
st.code("""
# Label 인코딩
label_key = Data['Emotion'].unique()
label_value = [0,1,2,3,4,5,6]
label_dict = {key:value for key, value in zip(label_key,label_value)}
Data['LabelEncoding'] = Data['Emotion'].apply(lambda x: label_dict[x])
""")
st.image("https://shyo2-public.s3.ap-northeast-2.amazonaws.com/static/LabelOneHot.png")
st.code("""
# label의 차이를 없애기 위핸 원-핫 인코딩 진행
from keras.utils import to_categorical

Label = to_categorical(Data['LabelEncoding'])
""")
st.code("""
# 최종 데이터셋
X_Data = Data['Encode']
Y_Data = Label
""")
st.image("https://shyo2-public.s3.ap-northeast-2.amazonaws.com/static/FinalDataset.png")
st.code("""
# X_Data 원-핫 인코딩
X_Hot_Data = []
for data in X_Data:
    tmp = np.zeros(5091)
    set_data = set(data)
    list_data = list(data)
    tmp[list_data] = 1
    X_Hot_Data.append(list(tmp))
X_Hot_Data = np.array(X_Hot_Data)
""")

st.divider()

st.subheader("감정 분류 머신러닝 모델 학습")

st.code("""
# 학습, 테스트 데이터 분리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_Hot_Data, Y_Data, test_size=0.2, random_state=125)
""")
st.code("""
# epoch 6, acc 48.78%
from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt

model = Sequential()

model.add(Dense(12, activation='elu', input_shape=(5091,)))
model.add(Dense(7, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=100, epochs=15, validation_data=(x_test, y_test))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
""")
st.image("https://shyo2-public.s3.ap-northeast-2.amazonaws.com/static/Accuracy.png")
st.write("머신러닝으로 감정 분류를 진행한 결과 약 48%의 정확도를 얻었습니다. \n\nLLM으로는 어느정도 정확도가 나올지 궁금해 추가로 파인튜닝을 진행했습니다.\n")

st.divider()

st.subheader("Llama-3.2-1B 감정 분류 파인튜닝")
st.code("""
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# 데이터 불러오기 및 전처리
data = pd.read_csv(emotiondata_path)

# 라벨 인코딩
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['emotion'])

# 텍스트와 라벨 분리
texts = data['setence'].tolist()
labels = data['label'].tolist()

# 학습 데이터와 테스트 데이터 분리
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)
""")

st.code("""
#2
# 모델과 토크나이저 불러오기
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=11)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # eos_token을 pad_token으로 설정
model.config.pad_token_id = tokenizer.pad_token_id

# 데이터 토크나이징
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
""")

st.code("""
#2
# 데이터셋 클래스 정의
class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = EmotionDataset(train_encodings, train_labels)
val_dataset = EmotionDataset(val_encodings, val_labels)

# 학습 설정
training_args = TrainingArguments(
    output_dir='./results',            # 모델 결과 저장 경로
    eval_strategy="epoch",       # 매 epoch마다 평가
    per_device_train_batch_size=8,     # 학습 배치 사이즈
    per_device_eval_batch_size=8,      # 평가 배치 사이즈
    num_train_epochs=4,                # 에포크 수
    weight_decay=0.01,                 # 가중치 감소 (정규화)
    logging_dir='./logs',              # 로그 저장 경로
    logging_steps=10,
    load_best_model_at_end=True,       # 가장 좋은 모델을 마지막에 저장
    save_strategy="epoch"              # 에포크마다 모델 저장
)

# 정확도 계산 함수 정의
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")  # 가중치 평균 F1 스코어
    return {"accuracy": accuracy, "f1": f1}

# DataCollatorWithPadding을 사용해 패딩을 자동으로 추가
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

from transformers import Trainer

# Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator,  # 데이터 콜레이터 추가
)
""")

st.code("""
# 학습 후 평가
trainer.train()
eval_results = trainer.evaluate()

# 정확도 출력
print("Evaluation Results:", eval_results)
""")
st.image("https://shyo2-public.s3.ap-northeast-2.amazonaws.com/static/wandb.png")
st.write("파인튜닝 결과, epoch4에서 약 48%의 정확도를 얻었습니다. \n\n 머신러닝과 비교했을 때 정확도는 큰 차이가 없지만 파라미터의 차이가 약 5000개와 10억개로 큰 차이가 있고, 이를 통해 분류라는 작업에 있어서는 머신러닝이 더 적합하다는 생각이 들었습니다.\n")

st.divider()

# st.subheader("노래 생성 AI")
# st.write("노래 생성 AI는 musicgen-melody 모델을 거의 그대로 사용했습니다. \n\n 대신 프롬프트로 앞서 생성한 요약문과 감정을 입력해서 글에 맞는 BGM이 생성되도록 했습니다.\n\n 현재는 리소스 문제로 text로만 음악을 생성하고 있습니다!")
# st.link_button("musicgen-melody", "https://huggingface.co/facebook/musicgen-melody")
# st.code("""
# # high-level-pipeline을 활용한 노래 생성
# from transformers import pipeline

# musicgen = pipeline("text-to-audio", model="facebook/musicgen-melody")

# @st.cache_data
# def generate_song(emotion):
#     output = musicgen(emotion)
#     audio = np.array(output['audio'][0][0])  # 2차원 배열을 1차원으로 변환
#     sampling_rate = output['sampling_rate']
    
#     audio = (audio * 32767).astype(np.int16)

#     audio_buffer = BytesIO()
#     write(audio_buffer, sampling_rate, audio)
#     audio_buffer.seek(0)

#     return audio_buffer
# """)

# st.divider()

st.write("""facebook/musicgen-melody 모델을 활용한 노래 생성은 따로 작업한것 없이 연결만 했기에 따로 추가하지 않았습니다. \n\n 아래에는 노래 생성만이 제외된 Sentifl LLM을 테스트해보실 수 있습니다.""")


st.subheader("Sentifl LLM")
sentence = st.text_area("문장을 입력하세요", height=200)



url = os.getenv("SENTIFL_LLM_URL")
headers = {'accept': 'application/json','Content-Type': 'application/json'}
data = {"sentence": sentence}

if st.button("AI 실행"):
    if sentence.strip():
        with st.spinner("문장 요약 & 감정 추출 중..."):
            response = requests.post(url, json=data, headers=headers)
            response_data = response.json()
            summ_text = response_data.get("summary")
            emotion = response_data.get("emotion")
            st.success("요약 및 추출 성공!!")
            st.write("요약된 문장 : ", summ_text)
            st.write("감정 : ", emotion)
        # with st.spinner("노래 생성 중...\n\n- GPU가 비싸서 CPU로만 생성하고 있어 많이 느려요!!\n\n- 빠르면 5분 느리면 10분 정도 걸려요!!"):
        #     time.sleep(3)
        #     st.warning("현재는 리소스 문제로 노래 생성 AI가 동작하지 않아요!!")
            # audio_file = generate_song(emotion)
            # st.audio(audio_file, format="audio/wav")
            # st.success("노래 생성 성공!!")
    else:
        st.warning("문장을 입력하지 않았습니다. 문장을 입력하세요.")
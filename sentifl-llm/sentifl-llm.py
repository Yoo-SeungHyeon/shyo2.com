from transformers import BartForConditionalGeneration, AutoTokenizer, LlamaForSequenceClassification
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SentenceRequest(BaseModel):
    sentence: str

class SentenceResponse(BaseModel):
    summary: str
    emotion : str

kobart_summarizer_model = None
kobart_summarizer_tokenizer = None

emotion_classifier_model = None
emotion_classifier_tokenizer = None

model_setting = False

def get_model():
    
    global kobart_summarizer_model
    global kobart_summarizer_tokenizer
    global emotion_classifier_model
    global emotion_classifier_tokenizer

    global model_setting

    summ_path = 'gogamza/kobart-summarization'
    kobart_summarizer_model =  BartForConditionalGeneration.from_pretrained(summ_path)
    kobart_summarizer_tokenizer = AutoTokenizer.from_pretrained(summ_path)

    class_path = "shyo2/emotion_classification"
    emotion_classifier_model = LlamaForSequenceClassification.from_pretrained(class_path, num_labels=7)
    emotion_classifier_tokenizer = AutoTokenizer.from_pretrained(class_path)

    # pad_token 변경
    if emotion_classifier_tokenizer.pad_token is None:
        emotion_classifier_tokenizer.pad_token = emotion_classifier_tokenizer.eos_token

    model_setting = True

def sentifl_llm(text: str):

    emotion_list = ["공포","놀람","분노","슬픔","중립","행복","혐오"]

    text = text.replace("\n", "")
    raw_input_ids = kobart_summarizer_tokenizer.encode(text)
    input_ids = [kobart_summarizer_tokenizer.bos_token_id] + raw_input_ids + [kobart_summarizer_tokenizer.eos_token_id]
    # 모델 추론 (문장 요약)
    summary_ids = kobart_summarizer_model.generate(torch.tensor([input_ids]), max_length=128, early_stopping=True)
    # 요약 결과 반환
    summ_text = kobart_summarizer_tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
    # 입력 텍스트 토크나이징
    inputs = emotion_classifier_tokenizer(summ_text, return_tensors="pt", padding=True, truncation=True)
    # 모델 추론 (감정 분류)
    with torch.no_grad():
        outputs = emotion_classifier_model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1)


    # 결과 반환
    emotion = emotion_list[predicted_class.item()]

    return summ_text, emotion

@app.on_event("startup")
def startup_event():
    get_model()
    print("모델 로딩 완료")

@app.get("/")
def read_root():
    return {"message": "FastAPI 서버가 실행 중입니다!"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/ready")
def readiness_check():
    if model_setting:
        return {"status": "ready"}
    else:
        return {"status": "not_ready"}

@app.post("/sentifl-llm")
async def summarize_sentence(sentence_request: SentenceRequest):
    try:
        input = sentence_request.sentence
        print(input)
        print(type(input))
        summary, emotion = sentifl_llm(input)
    except Exception as e:
        import traceback
        traceback.print_exc()  # 전체 스택 트레이스를 출력
        raise HTTPException(status_code=400, detail=str(e))

    return SentenceResponse(summary=summary, emotion=emotion)
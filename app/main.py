from fastapi import FastAPI
from code import predict_language
import pickle

app = FastAPI()

m = pickle.load(open(r'..\model\cls_langauage_0.1.pkl', 'rb'))
cv = pickle.load(open(r'..\model\cv_feature.pkl', 'rb'))

@app.get("/")
def root():
    return {"message": "This is my api"}

@app.get("/api/predict{item_str}")
def read_str(item_str):
    lang = predict_language(m, cv, item_str)
    return {"language": lang}
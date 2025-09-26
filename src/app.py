import re
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Descargar recursos NLTK una vez
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_url(url):
    url = re.sub(r"https?://", "", url)
    url = re.sub(r"[^\w]", " ", url)
    url = url.lower()
    tokens = url.split()
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(clean_tokens)

# Cargar vectorizador y modelo entrenado
with open("modelo_nlp.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Página principal
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Endpoint de predicción
@app.post("/api/predict")
async def api_predict(url: str = Form(...)):
    try:
        url_clean = preprocess_url(url)
        url_vect = vectorizer.transform([url_clean])
        prediction = int(model.predict(url_vect)[0])
        return JSONResponse({"prediction": prediction})
    except Exception as e:
        return JSONResponse({"prediction": None, "error": str(e)})

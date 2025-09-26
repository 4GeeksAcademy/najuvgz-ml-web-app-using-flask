import pickle
from fastapi import FastAPI
from pydantic import BaseModel

# Cargar el modelo desde el archivo .pkl
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

# Definimos el formato de entrada
class LinkInput(BaseModel):
    url: str

@app.get("/")
def home():
    return {"message": "API funcionando. Usa /predict para clasificar."}

@app.post("/predict")
def predict(data: LinkInput):
    # Preprocesa el enlace si tu modelo lo requiere
    url = data.url
    
    # Renderizar predicción (asumiendo que tu modelo tiene .predict)
    prediction = model.predict([url])[0]
    
    return {"url": url, "prediction": str(prediction)}

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import time
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(
    title="API de Previsão de Ações - ITSA4",
    description="Previsão de preço de fechamento com LSTM",
    version="3.0"
)

Instrumentator().instrument(app).expose(app)

# Carrega o modelo
model = tf.keras.models.load_model("itausa_model.h5")
scaler = MinMaxScaler()

# Modelo da requisição
class PriceRequest(BaseModel):
    closes: list[float]  # Últimos 60 preços
    n_dias: int = 7       # Dias que o usuário quer prever (padrão = 7)

@app.post("/predict")
def predict_price(request: PriceRequest):
    start_time = time.time()

    closes = request.closes
    n_dias = request.n_dias

    if len(closes) < 60:
        raise HTTPException(status_code=400, detail="São necessários 60 preços para previsão.")

    if n_dias < 1 or n_dias > 30:
        raise HTTPException(status_code=400, detail="O número de dias deve estar entre 1 e 30.")

    sequence = closes.copy()
    preds = []

    # Fit scaler na sequência original
    data = np.array(sequence).reshape(-1, 1)
    scaler.fit(data)

    for _ in range(n_dias):
        scaled_input = scaler.transform(np.array(sequence[-60:]).reshape(-1, 1))
        X = np.reshape(scaled_input, (1, 60, 1))
        pred_scaled = model.predict(X, verbose=0)
        pred_real = scaler.inverse_transform(pred_scaled)[0][0]
        preds.append(round(float(pred_real), 2))
        sequence.append(pred_real)

    tempo_resposta = round(time.time() - start_time, 4)

    return {
        "dias_previstos": n_dias,
        "previsao": preds,
        "tempo_resposta_segundos": tempo_resposta
    }

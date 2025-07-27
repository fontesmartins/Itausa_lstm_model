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
    version="1.0"
)


## Adiciona monitoramento Prometheus ANTES da inicialização
Instrumentator().instrument(app).expose(app)

## Carregando modelo e pré-processamento
model = tf.keras.models.load_model("itausa_model.h5")
scaler = MinMaxScaler()

## Classe de entrada com validação
class PriceRequest(BaseModel):
    closes: list[float]  # Últimos 60 preços de fechamento

# endpoint de previsão
@app.post("/predict")
def predict_price(request: PriceRequest):
    start_time = time.time()
    closes = request.closes

    if len(closes) < 60:
        raise HTTPException(status_code=400, detail="São necessários pelo menos 60 preços.")

    data = np.array(closes).reshape(-1, 1)
    data_scaled = scaler.fit_transform(data)

    X = np.reshape(data_scaled[-60:], (1, 60, 1))

    # 🔮 Previsão
    pred_scaled = model.predict(X)
    pred = scaler.inverse_transform(pred_scaled)

    tempo_resposta = round(time.time() - start_time, 4)

    return {
        "preco_previsto": round(float(pred[0][0]), 2),
        "tempo_resposta_segundos": tempo_resposta
    }

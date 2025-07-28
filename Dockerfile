FROM python:3.10

WORKDIR /app

# Copia arquivos
COPY . /app

# Instala dependências
RUN pip install --no-cache-dir -r requirements.txt

# Expõe a porta padrão do Railway
EXPOSE 8000

# Inicia o app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

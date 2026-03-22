FROM python:3.12-slim

WORKDIR /app

# install CPU-only PyTorch first (avoids 2GB+ CUDA libs)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# install remaining deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy app code
COPY app/ ./app/

# copy model weights
COPY model_output/ ./model_output/

# create logs dir
RUN mkdir -p logs

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
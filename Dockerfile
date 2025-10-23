FROM python:3.11-slim

# System deps (nice-to-have for pandas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

# Python deps
RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install streamlit==1.37.1 pandas==2.2.2 numpy==1.26.4 python-dotenv==1.0.1

# Copy app
WORKDIR /app
COPY app/ app/
COPY data/ data/

EXPOSE 8501
CMD ["python","-m","streamlit","run","app/streamlit_app.py","--server.address=0.0.0.0","--server.port=8501"]

"""
=========================================================
⚡ FastAPI Master Class
---------------------------------------------------------
Model: Iris Species Classifier (RandomForest)
Pydantic models defined in app/schemas.py
---------------------------------------------------------
This app demonstrates:
In-memory caching with lru_cache
Redis caching for distributed systems
Simulated heavy computation API
ML model inference caching
Readiness for Load Balancing (stateless APIs)
=========================================================
"""

from fastapi import FastAPI
from functools import lru_cache
from transformers import pipeline
import time, redis, json, hashlib
from app.schemas import IrisInput, PredictionOutput
import joblib
import numpy as np


app = FastAPI()


# Load the model once at startup
model = joblib.load("model/iris_model.joblib")

# Iris class names for readability
class_names = ['setosa', 'versicolor', 'virginica']


@app.get("/info")
def model_info():
    """Basic model info"""
    return {
        "model_type": "RandomForestClassifier",
        "classes": class_names
    }

@app.post("/predict", response_model=PredictionOutput)
def predict_species(data: IrisInput):
    """Make prediction from input features"""
    features = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    prediction = model.predict(features)[0]
    predicted_class = class_names[prediction]
    return {"predicted_class": predicted_class}


# ----------------------------------------------------------------
# 🧠 Redis Setup (for distributed caching across multiple servers)
# ----------------------------------------------------------------
try:
    r = redis.Redis(host="localhost", port=6379, db=0)
    r.ping()  # check if redis is running
    print("✅ Connected to Redis")
except Exception as e:
    r = None
    print("⚠️ Redis not available:", e)

# ----------------------------------------------------------------
# ⚙️ 1. In-Memory Cache Example using functools.lru_cache
# ----------------------------------------------------------------
@lru_cache(maxsize=50)
def heavy_computation(n: int):
    """
    Simulates a slow computation (e.g., model training or data aggregation)
    The result is cached in memory for faster repeat access.
    """
    print(f"🧮 Computing heavy task for {n}...")
    time.sleep(3)
    return {"number": n, "result": n * n}


@app.get("/inmemory-cache")
def get_inmemory_cache(n: int):
    """
    Example API:
    - Uses in-memory caching.
    - Fast for repeated requests on the same input.
    ⚙️ Use when: You have single instance / local caching needs.
    """
    return heavy_computation(n)


# ----------------------------------------------------------------
# ⚙️ 2. Redis Caching Example (Distributed Cache)
# ----------------------------------------------------------------
@app.get("/redis-cache")
def get_redis_cache(query: str):
    """
    Example API:
    - Uses Redis for shared caching across multiple app servers.
    - Simulates an expensive operation (3s delay).
    ⚙️ Use when: You deploy multiple FastAPI instances behind a load balancer.
    """
    if not r:
        return {"error": "Redis not connected"}

    cache_key = f"data:{query}"
    cached_data = r.get(cache_key)  

    # Return cached result if available
    if cached_data:
        print("⚡ Cache hit!")
        return json.loads(cached_data)

    print("🕒 Cache miss, performing heavy computation...")
    time.sleep(3)
    result = {"query": query, "response": f"Processed data for '{query}'"}

    # Cache result for 1 minute
    r.setex(cache_key, 60, json.dumps(result))
    return result


# ----------------------------------------------------------------
# ⚙️ 3. AI/ML Model Caching Example (Sentiment Analysis)
# ----------------------------------------------------------------
# Load model once globally (avoid reloading every request)
sentiment_model = pipeline("sentiment-analysis")


def generate_hash_key(text: str):
    """Create a unique hash key for each input text"""
    return "sentiment:" + hashlib.md5(text.encode()).hexdigest()


@app.post("/analyze-text")
async def analyze_text(text: str):
    """
    Example API:
    - Performs sentiment analysis on a given text.
    - Results cached in Redis for faster repeat calls.
    ⚙️ Use when: You serve AI/ML inference APIs that receive repeat queries.
    """
    if not r:
        return {"error": "Redis not connected"}

    key = generate_hash_key(text)
    cached_result = r.get(key)

    if cached_result:
        print("⚡ Returning cached model output")
        return json.loads(cached_result)

    print("🧠 Running model inference...")
    result = sentiment_model(text)
    r.setex(key, 300, json.dumps(result))  # cache for 5 mins
    return result


# ----------------------------------------------------------------
# ⚙️ 4. Health Check API (for Load Balancers)
# ----------------------------------------------------------------
@app.get("/health")
def health_check():
    """
    Example API:
    - Simple health check endpoint.
    - Load balancers like Nginx, AWS ELB, or Kubernetes call this to check
      if the instance is alive and healthy.
    ⚙️ Use when: You deploy multiple instances and need auto-failover.
    """
    return {"status": "healthy"}


# ----------------------------------------------------------------
# ⚙️ 5. Simulated Slow API (to visualize balancing)
# ----------------------------------------------------------------
@app.get("/simulate-load")
def simulate_load():
    """
    Example API:
    - Simulates a slow endpoint to test how a load balancer
      distributes load between multiple FastAPI servers.
    ⚙️ Use when: You want to test Nginx or Docker load balancing.
    """
    time.sleep(2)
    return {"message": "This request was served successfully after a short delay!"}


# ----------------------------------------------------------------
# ✅ App ready for Load Balancing
# ----------------------------------------------------------------
"""
To simulate load balancing locally:
1. Run two instances:
   uvicorn app:app --port 8000
   uvicorn app:app --port 8001

2. Use Nginx with upstream config:
   upstream fastapi_servers {
       server 127.0.0.1:8000;
       server 127.0.0.1:8001;
   }
   server {
       listen 8080;
       location / {
           proxy_pass http://fastapi_servers;
       }
   }

3. Access http://localhost:8080/
   -> Nginx will distribute requests between the two FastAPI instances.
"""

import asyncio
import json
import pickle
import uuid
from datetime import datetime

import httpx
from hypercorn.asyncio import serve
from hypercorn.config import Config
from quart import Quart, jsonify, request

app = Quart("Churn")

# Globals
C = 1.0
input_file = f"model_C={C}.bin"
G = 0.5

with open(input_file, "rb") as f_in:
    dv, model = pickle.load(f_in)

@app.post("/predict")
async def predict():
    try:
        customer = await request.get_json(force=True)
        await logger.info(f"Predicting customer:\n {'\n '.join(f'{k}: {v}' for k, v in customer.items())}")
        X = dv.transform([customer])
        y_pred = model.predict_proba(X)[0, 1]
        await logger.info(f"Churn rate for prediction is {y_pred:.2%}")
        churn = y_pred >= G
        result = {
            "churn_rate": f"{y_pred:.2%}",
            "churn": bool(churn),
            "timestamp": datetime.now().isoformat(),
            "customer-id": uuid.uuid4(),
        }
        return jsonify(result)
    except Exception as e:
        await logger.error(e)


@app.get("/predict")
async def predict_proxy():
    with open("request.json", "r") as json_file:
        data = json.load(json_file)
    async with httpx.AsyncClient() as client:
        r = await client.post(url="http://127.0.0.1:8000/predict", json=data)
        result = r.json()
        return result


config = Config()
config.accesslog = "-"
logger = config.logger_class(config)

if __name__ == "__main__":
    with open(input_file, "rb") as f_in:
        dv, model = pickle.load(f_in)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.set_debug(True)
    loop.run_until_complete(serve(app, config))

import asyncio
from asgiref.wsgi import WsgiToAsgi
from flask import Flask
from hypercorn.asyncio import Config, serve

app = Flask(__name__)


@app.route("/ping", methods=["GET"])
async def ping():
    await logger.info(message="pong")
    return "pong"


asgi_app = WsgiToAsgi(app)
asgi_config = Config
asgi_config.accesslog = "-"
logger = asgi_config.logger_class(asgi_config())

if __name__ == "__main__":
    asyncio.run(serve(asgi_app, asgi_config()))

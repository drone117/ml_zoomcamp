import asyncio
from quart import Quart
from hypercorn.asyncio import serve
from hypercorn.config import Config

app = Quart(__name__)


@app.get("/ping")
async def ping():
    await logger.info("Received ping request - pong")
    return "pong"


config = Config()
config.accesslog = "-"
logger = config.logger_class(config)

if __name__ == "__main__":
    asyncio.run(serve(app, config))

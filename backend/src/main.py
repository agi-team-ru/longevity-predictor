from contextlib import asynccontextmanager
import logging
import os
from fastapi import FastAPI

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "WARNING").upper())

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):

    # classifier_models.create_all_tables()

    yield {
    }


app = FastAPI(lifespan=lifespan)


@app.get("/")
def version():
    return {"version": "1.0.0"}

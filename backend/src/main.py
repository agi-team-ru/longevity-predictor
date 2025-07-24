from contextlib import asynccontextmanager
import logging
import os
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import spacy

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "WARNING").upper())

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):

    # classifier_models.create_all_tables()

    yield {}


# Загрузка spaCy моделей напрямую с указанием cache_dir
ner_bc5cdr = None
ner_bionlp = None

# linker = EntityLinker(
#     resolve_abbreviations=True,
#     linker_name='umls',
#     threshold=0.85,
#     filter_for_definitions=True,
# )

# SentenceTransformer использует /app/models как cache_folder
embedder: Optional[SentenceTransformer] = None


class NERRequest(BaseModel):
    text: str


class EmbedRequest(BaseModel):
    text: str


app = FastAPI()


@app.get("/")
def version():
    return {"version": "1.0.0"}


@app.post("/ner")
def ner_entities(req: NERRequest):
    global ner_bc5cdr, ner_bionlp
    if not ner_bc5cdr:
        ner_bc5cdr = spacy.load("/app/models/en_ner_bc5cdr_md")
        ner_bionlp = spacy.load("/app/models/en_ner_bionlp13cg_md")

    entities = []
    for ner in [ner_bionlp, ner_bc5cdr]:
        doc = ner(req.text)
        for ent in doc.ents:
            entities.append({"text": ent.text, "label": ent.label_})
    return {"entities": entities}


@app.post("/embed")
def embed_text(req: EmbedRequest):
    global embedder
    if not embedder:
        embedder = SentenceTransformer(
            "BAAI/bge-large-en-v1.5", cache_folder="/app/models"
        )

    emb = embedder.encode(req.text)
    return {"embedding": emb.tolist() if hasattr(emb, "tolist") else list(emb)}



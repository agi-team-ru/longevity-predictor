from contextlib import asynccontextmanager
import logging
import os
from fastapi import FastAPI
from pydantic import BaseModel
from scispacy.linking import EntityLinker
from sentence_transformers import SentenceTransformer
import pickle

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "WARNING").upper())

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):

    # classifier_models.create_all_tables()

    yield {
    }


app = FastAPI(lifespan=lifespan)


# Загрузка моделей из pickle из /models
with open("/app/models/en_ner_bc5cdr_md.pkl", "rb") as f:
    ner_bc5cdr = pickle.load(f)
with open("/app/models/en_ner_bionlp13cg_md.pkl", "rb") as f:
    ner_bionlp = pickle.load(f)
linker = EntityLinker(
    resolve_abbreviations=True,
    linker_name='umls',
    threshold=0.85,
    filter_for_definitions=True
)
embedder = SentenceTransformer("BAAI/bge-large-en-v1.5")

class NERRequest(BaseModel):
    text: str

class EmbedRequest(BaseModel):
    text: str

@app.get("/")
def version():
    return {"version": "1.0.0"}

@app.post("/ner")
def ner_entities(req: NERRequest):
    entities = []
    for ner in [ner_bionlp, ner_bc5cdr]:
        doc = ner(req.text)
        doc = linker(doc)
        for ent in doc.ents:
            cui, score = ent._.kb_ents[0] if ent._.kb_ents else (None, None)
            canon = linker.kb.cui_to_entity[cui].canonical_name if cui else None
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "cui": cui,
                "canonical": canon,
                "score": score
            })
    return {"entities": entities}

@app.post("/embed")
def embed_text(req: EmbedRequest):
    emb = embedder.encode(req.text, normalize_embeddings=True, show_progress_bar=False)
    return {"embedding": emb}

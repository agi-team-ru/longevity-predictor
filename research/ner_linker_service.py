from fastapi import FastAPI
from pydantic import BaseModel
import spacy
from scispacy.linking import EntityLinker
from langchain.embeddings import HuggingFaceEmbeddings

app = FastAPI()

# Загружаем модели один раз при старте
ner_bc5cdr = spacy.load("en_ner_bc5cdr_md")
ner_bionlp = spacy.load("en_ner_bionlp13cg_md")
linker = EntityLinker(
    resolve_abbreviations=True,
    linker_name='umls',
    threshold=0.85,
    filter_for_definitions=True
)
embedder = HuggingFaceEmbeddings(
    model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
)

class NERRequest(BaseModel):
    text: str

class EmbedRequest(BaseModel):
    text: str

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
    # Возвращаем список float (эмбеддинг)
    emb = embedder.embed_query(req.text)
    return {"embedding": emb}
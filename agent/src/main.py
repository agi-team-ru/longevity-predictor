from fastapi import FastAPI
from pydantic import BaseModel
import logging
import os
from contextlib import asynccontextmanager
import json
from src.graph_builder.neo4j_graph_maker import process_articles, client, LLM_MODEL
from src.graph_builder.graph_clusterizer import cluster_tasks_in_neo4j
from src.scorer.scorer_acticles import update_confidence_scores_in_neo4j

# Импорт парсеров напрямую
from src.parsers.parcer_pudmed import PubMedParser
from src.parsers.parcer_biorxiv import fast_multiword_biorxiv_search
# from src.parsers.parser_X import scrape_x_hashtag

from src.scorer.score_name_clusters import name_and_score_clusters
from src.reporter.report_creating import generate_reports_from_neo4j

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "WARNING").upper())

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # classifier_models.create_all_tables()
    yield {}

app = FastAPI()

@app.get("/")
def version():
    return {"version": "1.0.0"}

@app.post("/run_all_parsers")
def run_all_parsers():
    data_dir = "/app/data"
    os.makedirs(data_dir, exist_ok=True)
    # PubMed
    parser = PubMedParser()
    uids = parser.get_uids()
    pubmed_data = parser.get_json(uids)
    with open(os.path.join(data_dir, "pubmed_articles.json"), "w", encoding="utf-8") as f:
        json.dump(pubmed_data, f, ensure_ascii=False, indent=2)

    # BioRxiv
    biorxiv_data = fast_multiword_biorxiv_search()
    with open(os.path.join(data_dir, "biorxiv_multiword.json"), "w", encoding="utf-8") as f:
        json.dump(biorxiv_data, f, ensure_ascii=False, indent=2)

    # parser_X
    # try:
    #     hashtag = "longevity"
    #     X_USERNAME = os.environ.get('X_USERNAME', 'X_USERNAME')
    #     X_PHONE_OR_USERNAME = os.environ.get('X_PHONE_OR_USERNAME', 'X_PHONE_OR_USERNAME')
    #     X_PASSWORD = os.environ.get('X_PASSWORD', 'X_PASSWORD')
    #     # tweets = scrape_x_hashtag(
    #     #     hashtag,
    #     #     scroll_count=2,
    #     #     username=X_USERNAME,
    #     #     password=X_PASSWORD,
    #     #     phone_or_username=X_PHONE_OR_USERNAME
    #     # )
    #     with open(os.path.join(data_dir, "tweets.json"), "w", encoding="utf-8") as f:
    #         json.dump(tweets, f, ensure_ascii=False, indent=2)
    # except Exception as e:
    #     with open(os.path.join(data_dir, "x_status.txt"), "w", encoding="utf-8") as f:
    #         f.write(f"Error: {e}")
    return {"status": "ok"}

@app.post("/build_graph_from_file")
def build_graph_from_file():
    process_articles("/app/data/pubmed_articles.json")
    return {"status": "ok"}

@app.post("/cluster_tasks")
def cluster_tasks():
    cluster_tasks_in_neo4j()
    return {"status": "ok"}

# @app.post("/test_llm")
# def test_llm():
#     response = client.chat.completions.create(
#         model=LLM_MODEL,
#         messages=[{"role": "user", "content": "text"}],
#         max_tokens=200,
#         temperature=1
#     )
#     return {"status": "ok"}

@app.post("/score_graph")
def score_graph():
    update_confidence_scores_in_neo4j()
    return {"status": "ok"}

@app.post("/score_graph_name_clusters")
def score_name_clusters():
    name_and_score_clusters()
    return {"status": "ok"}

@app.post("/generate_report")
def generate_report():
    generate_reports_from_neo4j(top_k=1)
    return {"status": "ok"}
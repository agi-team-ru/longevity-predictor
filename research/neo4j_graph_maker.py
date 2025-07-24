import json
import os
from openai import OpenAI
import requests
from dotenv import load_dotenv
from neo4j import GraphDatabase
from datetime import datetime
from tqdm import tqdm

load_dotenv()

# --- Настройки ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

client = OpenAI(
    base_url=os.getenv("OPENAI_API_BASE"),
    api_key=os.getenv("OPENAI_API_KEY")
)

# --- LLM PROMPT (без изменений) ---
def verif_article_and_extract_task(abstract: str, temperature: float = 0.2) -> dict:
    maturity_options = [
        "high", "medium", "low", "unknown",
        "preclinical", "animal", "clinical", "approved", "review", "other"
    ]
    maturity_options_str = ', '.join(f'"{m}"' for m in maturity_options)
    prompt = f'''
You are a biomedical research assistant building a knowledge graph of *research priorities in the biology of ageing and longevity*.

**Task**
Given the ABSTRACT below, decide whether the paper is relevant to our project. A *relevant* paper:
- discusses **open questions / future directions / challenges / knowledge gaps / research priorities**
- addresses **molecular or cellular mechanisms of ageing** (telomere attrition, mitochondrial dysfunction, DNA damage, epigenetic changes, cellular senescence, etc.) **or interventions targeting those mechanisms**
- is **not** purely geriatric clinical work, sociology, health‑policy, resource allocation or economics.

**Output format (JSON only)**
If the article is relevant, return:
{{ "relevant": true, "task": "<one sentence>", "category": "<main research area, e.g. epigenetics/metabolism/immunology/senescence/etc.>", "maturity": "<one of: {maturity_options_str}>", "explanation": "<brief about 25 words>" }}
If the article is not relevant, return:
{{ "relevant": false, "explanation": "<brief reason>" }}

Return only the JSON object with no additional text or markdown.

ABSTRACT:
"""{abstract}"""
'''.strip()

    response = client.chat.completions.create(
        model="llama-3.3-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=temperature
    )
    raw = response.choices[0].message.content.strip()
    data = json.loads(raw)
    if data.get('maturity') not in set(maturity_options):
        data['maturity'] = 'unknown'
    return data

# --- NER API ---
def get_tags_via_api(abstract: str, url="http://localhost:8001/ner"):
    resp = requests.post(url, json={"text": abstract})
    resp.raise_for_status()
    entities = resp.json()["entities"]
    tags = {}
    for ent in entities:
        canon = ent["canonical"]
        label = ent["label"]
        if canon and canon not in tags:
            tags[canon] = label
    return tags

# --- Neo4j ---
class Neo4jGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    def close(self):
        self.driver.close()

    def add_task_with_article_and_tags(self, task_data, article, tags):
        with self.driver.session() as session:
            session.write_transaction(self._add_task_tx, task_data, article, tags)

    @staticmethod
    def _add_task_tx(tx, task_data, article, tags):
        # Добавляем задачу
        tx.run(
            """
            MERGE (t:Task {text: $task, category: $category, maturity: $maturity})
            SET t.explanation = $explanation
            """,
            task=task_data['task'],
            category=task_data.get('category', 'unknown'),
            maturity=task_data.get('maturity', 'unknown'),
            explanation=task_data.get('explanation', '')
        )
        # Добавляем статью
        tx.run(
            """
            MERGE (a:Article {title: $title})
            SET a += $props
            """,
            title=article['title'],
            props={k: v for k, v in article.items() if k != 'title'}
        )
        # Связь задача-статья
        tx.run(
            """
            MATCH (t:Task {text: $task}), (a:Article {title: $title})
            MERGE (t)-[:MENTIONED_IN]->(a)
            """,
            task=task_data['task'],
            title=article['title']
        )
        # Добавляем теги и связи
        for ent, label in tags.items():
            tx.run(
                """
                MERGE (e:Tag {name: $ent})
                SET e.label = $label
                WITH e
                MATCH (t:Task {text: $task})
                MERGE (t)-[:HAS_TAG]->(e)
                """,
                ent=ent, label=label, task=task_data['task']
            )

# --- Основной процесс ---
def process_articles(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)[:10]
    graph = Neo4jGraph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    for idx, art in enumerate(tqdm(articles)):
        print(f"\n[INFO] Processing article {idx+1}/{len(articles)}: {art.get('title', '')[:80]}")
        abstract = art.get('abstract', '')
        if not abstract:
            print("[WARN] No abstract, skipping.")
            continue
        try:
            task_data = verif_article_and_extract_task(abstract)
            print(f"[INFO] LLM task extraction result: {task_data}")
        except Exception as e:
            print(f"[ERROR] LLM extraction failed: {e}")
            continue
        if not task_data.get('relevant'):
            print("[INFO] Article not relevant, skipping.")
            continue
        try:
            tags = get_tags_via_api(abstract)
            print(f"[INFO] NER tags: {tags}")
        except Exception as e:
            print(f"[ERROR] NER API failed: {e}")
            tags = {}
        try:
            graph.add_task_with_article_and_tags(task_data, art, tags)
            print("[INFO] Written to Neo4j.")
        except Exception as e:
            print(f"[ERROR] Neo4j write failed: {e}")
    graph.close()
    print("[INFO] Processing complete.")

if __name__ == "__main__":
    process_articles("parsed_articles/pubmed_articles.json") 
import os
import json
from openai import OpenAI
from datetime import datetime
from neo4j import GraphDatabase

NEO4J_HOST = os.getenv("NEO4J_HOST", "localhost")
NEO4J_URI = os.getenv("NEO4J_URI", f"bolt://{NEO4J_HOST}:7687")
LLM_MODEL = os.getenv("LLM_MODEL")
REPORTS_DIR = "/app/data/reports/"

client = OpenAI()

def generate_reports_from_neo4j(top_k=1):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    driver = GraphDatabase.driver(NEO4J_URI, auth=("", ""))
    with driver.session() as session:
        # Получить все кластеры с именем и score, отсортировать по score
        clusters = session.run(
            """
            MATCH (c:Cluster)
            WHERE c.name IS NOT NULL AND c.score IS NOT NULL
            RETURN c.cid as cid, c.name as name, c.score as score
            ORDER BY c.score DESC
            """
        ).data()
        clusters = clusters[:top_k]
        for cluster in clusters:
            cid = cluster["cid"]
            name = cluster["name"]
            score = cluster["score"]
            # Получить все статьи, связанные с этим кластером через задачи
            articles_result = session.run(
                """
                MATCH (c:Cluster {cid: $cid})-[:HAS_TASK]->(:Task)-[:MENTIONED_IN]->(a:Article)
                RETURN a.title as title, a.abstract as abstract, a.journal as journal, a.maturity as maturity, a.pubdate as pubdate, a.confidence_score as confidence_score
                """,
                cid=cid
            )
            articles = []
            for row in articles_result:
                pubdate = row["pubdate"]
                try:
                    year = int(pubdate.split("-")[0]) if pubdate and pubdate != "0000-00-00" else "?"
                except Exception:
                    year = "?"
                articles.append({
                    "title": row["title"],
                    "abstract": row["abstract"],
                    "journal": row["journal"],
                    "maturity": row["maturity"],
                    "publication_year": year,
                    "confidence_score": row["confidence_score"]
                })
            # Формируем промпт для LLM
            articles_str = "\n".join([
                f"- Title: {a['title']}\n  Journal: {a['journal']}\n  Maturity: {a['maturity']}\n  Year: {a['publication_year']}\n  Confidence: {a['confidence_score']}\n  Abstract: {a['abstract']}" for a in articles
            ])
            prompt = f'''
You are a biomedical research reporting agent. Generate a detailed, explainable report for the following research priority in aging/longevity. Use only the information from the articles provided. Do not add external facts.

Cluster: {name}
Score: {score}

Articles:
{articles_str}

Instructions:
- Summarize the cluster's research priority and its importance.
- Justify why it is a top priority (use supporting facts from the articles).
- Mention the number and titles of supporting articles.
- List key journals and maturities.
- Explain the score and confidence.
- Output in readable, scientific style.
'''.strip()
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.2
            )
            report = response.choices[0].message.content.strip()
            fname = f"{REPORTS_DIR}report_{cid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(fname, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Report saved to {fname}")
    driver.close()

def main():
    generate_reports_from_neo4j(top_k=3)

if __name__ == "__main__":
    main()
import os
import json
from openai import OpenAI
from neo4j import GraphDatabase

NEO4J_HOST = os.getenv("NEO4J_HOST", "localhost")
NEO4J_URI = os.getenv("NEO4J_URI", f"bolt://{NEO4J_HOST}:7687")
LLM_MODEL = os.getenv("LLM_MODEL")

client = OpenAI()

def name_and_score_clusters():
    driver = GraphDatabase.driver(NEO4J_URI, auth=("", ""))
    with driver.session() as session:
        # Удалить все кластеры и их связи
        session.run("MATCH (c:Cluster) DETACH DELETE c")
        clusters = session.run("MATCH (c:Cluster) RETURN c.cid as cid").data()
        for cluster in clusters:
            cid = cluster["cid"]
            # Получить все задачи кластера
            tasks_result = session.run(
                """
                MATCH (c:Cluster {cid: $cid})-[:HAS_TASK]->(t:Task)
                RETURN t.text as text
                """,
                cid=cid
            )
            task_names = [row["text"] for row in tasks_result]
            if not task_names:
                continue
            # Сгенерировать название кластера через LLM
            system_prompt = (
                "You are an expert in biomedical research task clustering. "
                "Given a list of research tasks, generate a single, averaged canonical task formulation that best represents the typical wording and content of the tasks in the cluster. "
                "Do not add explanations or details not present in the tasks. Output only the canonical task."
            )
            user_prompt = (
                "Tasks:\n" + "\n".join(f"- {t}" for t in task_names) + "\nCanonical task:"
            )
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=50,
                temperature=0.2
            )
            cluster_name = response.choices[0].message.content.strip()
            # Обновить имя кластера
            session.run(
                """
                MATCH (c:Cluster {cid: $cid})
                SET c.name = $name
                """,
                cid=cid,
                name=cluster_name
            )
            # Получить confidence_score всех статей, связанных с этим кластером через задачи
            scores_result = session.run(
                """
                MATCH (c:Cluster {cid: $cid})-[:HAS_TASK]->(:Task)-[:MENTIONED_IN]->(a:Article)
                WHERE a.confidence_score IS NOT NULL
                RETURN a.confidence_score as score
                """,
                cid=cid
            )
            scores = [row["score"] for row in scores_result if row["score"] is not None]
            if scores:
                try:
                    scores = [float(s) for s in scores]
                    avg_score = sum(scores) / len(scores)
                except Exception:
                    avg_score = None
            else:
                avg_score = None
            # Обновить score кластера
            session.run(
                """
                MATCH (c:Cluster {cid: $cid})
                SET c.score = $score
                """,
                cid=cid,
                score=avg_score
            )
            print(f"[INFO] Cluster {cid}: name set to '{cluster_name}', score set to {avg_score}")
    driver.close()

if __name__ == "__main__":
    name_and_score_clusters()

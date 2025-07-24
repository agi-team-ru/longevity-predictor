from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np
from collections import defaultdict
from neo4j import GraphDatabase
import json
import os

NEO4J_HOST = os.getenv("NEO4J_HOST", "localhost")
NEO4J_URI = os.getenv("NEO4J_URI", f"bolt://{NEO4J_HOST}:7687")


def cluster_tasks_in_neo4j(threshold=0.5):
    driver = GraphDatabase.driver(NEO4J_URI, auth=("", ""))
    with driver.session() as session:
        # 1. Выгрузить задачи и эмбеддинги
        result = session.run(
            """
            MATCH (t:Task) WHERE t.embedding IS NOT NULL
            RETURN id(t) as id, t.text as text, t.embedding as embedding
            """
        )
        tasks = []
        embeddings = []
        node_ids = []
        for row in result:
            node_ids.append(row["id"])
            tasks.append(row["text"])
            emb = json.loads(row["embedding"]) if isinstance(row["embedding"], str) else row["embedding"]
            embeddings.append(emb)
        if not embeddings:
            print("[Clusterizer] No tasks with embeddings found.")
            return
        embeddings = np.array(embeddings)
        # 2. Кластеризация
        dist = 1 - cosine_similarity(embeddings)
        Z = linkage(dist, method='average')
        labels = fcluster(Z, t=threshold, criterion='distance')
        # 3. Записать кластеры в Neo4j
        cluster_map = defaultdict(list)
        for lbl, node_id in zip(labels, node_ids):
            cluster_map[lbl].append(node_id)
        for cluster_id, task_ids in cluster_map.items():
            # Создать ноду Cluster
            session.run(
                """
                MERGE (c:Cluster {cid: $cid})
                SET c.size = $size
                """,
                cid=int(cluster_id), size=len(task_ids)
            )
            # Связать Cluster с задачами
            for tid in task_ids:
                session.run(
                    """
                    MATCH (t:Task) WHERE id(t) = $tid
                    MATCH (c:Cluster {cid: $cid})
                    MERGE (c)-[:HAS_TASK]->(t)
                    """,
                    tid=tid, cid=int(cluster_id)
                )
        print(f"[Clusterizer] Created {len(cluster_map)} clusters in Neo4j.")
    driver.close()

if __name__ == "__main__":
    cluster_tasks_in_neo4j()

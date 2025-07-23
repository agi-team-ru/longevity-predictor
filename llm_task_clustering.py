import json
from openai import OpenAI
import os
from tqdm import tqdm
import time

os.environ["OPENAI_BASE_URL"]="https://localhost/api/v1"
os.environ["OPENAI_API_BASE"]="https://localhost/api/v1"
os.environ["OPENAI_API_KEY"]="sk-dummy-key"

CLUSTERS_PATH = 'llm_task_clusters.json'

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key"
)

def load_clusters():
    if os.path.exists(CLUSTERS_PATH):
        with open(CLUSTERS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_clusters(clusters):
    with open(CLUSTERS_PATH, 'w', encoding='utf-8') as f:
        json.dump(clusters, f, ensure_ascii=False, indent=2)

def llm_assign_cluster(new_task, canonical_tasks, max_retries=3):
    if not canonical_tasks:
        return None
    system_prompt = (
        'You are a biomedical research task clustering assistant. Your goal is to assign new research tasks to existing clusters based on semantic similarity of the research goal. If the new task is unrelated to all canonical tasks, say <answer>new cluster</answer>.\nIMPORTANT: Take into account both topic and level of specificity. Only assign to a cluster if the new task is aligned both in content and in level of generality/specificity.'
    )
    prompt = "Given the following canonical tasks (each represents a cluster):\n"
    for i, ct in enumerate(canonical_tasks, 1):
        prompt += f"{i}. {ct}\n"
    prompt += (
        '\nWhich cluster number (1..N) is the most semantically similar to the <input_task>? If none is similar, answer <answer>new cluster</answer>.\n'
        "\n<input_task>\n"
        f"{new_task}\n"
        "</input_task>\n\n"
        "Answer format:\n"
        '1. If the new task is unrelated to all canonical tasks, say <answer>new cluster</answer>\n'
        '2. Otherwise: <answer>NUMBER_OF_CLUSTER</answer>\n'
    )
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            answer = response.choices[0].message.content.strip().lower()
            if "<answer>new cluster</answer>" in answer:
                return None
            for i in range(1, len(canonical_tasks)+1):
                if f"<answer>{i}</answer>" in answer:
                    return i-1  # индекс кластера
            raise RuntimeError(f"Failed to assign cluster to task: {new_task}")
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 * attempt)
            else:
                raise e

def llm_generate_canonical_task(tasks, max_retries=3):
    system_prompt = (
        "You are an expert in biomedical research task clustering. "
        "Given a list of research tasks, generate a single, averaged canonical task formulation that best represents the typical wording and content of the tasks in the cluster. "
        "Do not add explanations or details not present in the tasks. Output only the canonical task."
    )
    prompt = "Tasks:\n"
    for t in tasks:
        prompt += f"- {t}\n"
    prompt += "Canonical task:"
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 * attempt)
            else:
                raise e

def process_new_task(new_task):
    clusters = load_clusters()
    canonical_tasks = [c['canonical_task'] for c in clusters]
    idx = llm_assign_cluster(new_task, canonical_tasks)
    if idx is None:
        # Новый кластер: canonical_task = первая задача
        clusters.append({'canonical_task': new_task, 'tasks': [new_task]})
        print(f"Created new cluster: {new_task}")
    else:
        clusters[idx]['tasks'].append(new_task)
        # canonical_task = среднее по всем задачам
        clusters[idx]['canonical_task'] = llm_generate_canonical_task(clusters[idx]['tasks'])
        print(f"Added to cluster {idx+1}: {clusters[idx]['canonical_task']}")
    save_clusters(clusters)

# Пример интеграции:
if __name__ == "__main__":
    # Пример: обработка списка задач из файла
    with open('example_new_tasks.json', 'r', encoding='utf-8') as f:
        new_tasks = json.load(f)
    for t in tqdm(new_tasks):
        process_new_task(t) 
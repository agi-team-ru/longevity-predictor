import json
from openai import OpenAI
from datetime import datetime

# Параметры
TASKS_DB_PATH = 'tasks_db.json'
GRAPH_PATH = 'full_longevity_graph.pkl'
REPORTS_DIR = 'reports/'

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key"
)

# 1. Загрузка задач и графа
def load_tasks():
    with open(TASKS_DB_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_graph():
    import pickle
    with open(GRAPH_PATH, 'rb') as f:
        return pickle.load(f)

def get_top_task(tasks_db, top_n=1):
    # Сортировка по score
    sorted_tasks = sorted(tasks_db.items(), key=lambda x: x[1].get('score', 0), reverse=True)
    return sorted_tasks[:top_n]

def gather_supporting_facts(graph, task_id):
    # Собираем supporting facts: статьи, теги, found_tasks
    node = graph.nodes[task_id]
    articles = [n for n in graph.neighbors(task_id) if graph.nodes[n].get('type') == 'article']
    tags = [n for n in graph.neighbors(task_id) if graph.nodes[n].get('type') == 'tag']
    return {
        'found_tasks': node.get('found_tasks', []),
        'articles': articles,
        'tags': tags
    }

def generate_report(task, facts, tasks_db, graph):
    # Формируем промпт для LLM
    task_text = task['found_tasks'][0] if task['found_tasks'] else ''
    meta = f"Category: {task.get('category','?')}, Maturity: {task.get('maturity','?')}, Impact: {task.get('impact','?')}, Score: {task.get('score','?')}, Confidence Score: {task.get('confidence_score','?')}"
    article_titles = [a for a in facts['articles']]
    tag_names = [t for t in facts['tags']]
    prompt = f'''
You are a biomedical research reporting agent. Generate a detailed, explainable report for the following research priority in aging/longevity.

Task: {task_text}
{meta}

Supporting articles:
{json.dumps(article_titles, ensure_ascii=False, indent=2)}

Key tags/entities:
{json.dumps(tag_names, ensure_ascii=False, indent=2)}

Instructions:
- Summarize the task and its importance.
- Justify why it is a top priority (use supporting facts).
- Mention the number and titles of supporting articles.
- List key tags/entities.
- Explain the maturity, impact, and score.
- Include the confidence score ({task.get('confidence_score', 'N/A')}) and explain what it means (higher = more reliable data from diverse sources).
- Output in readable, scientific style.
'''.strip()
    response = client.chat.completions.create(
        model="llama-3.3-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

def save_report(report, task_id):
    import os
    os.makedirs(REPORTS_DIR, exist_ok=True)
    fname = f"{REPORTS_DIR}report_{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(fname, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Report saved to {fname}")

def main():
    tasks_db = load_tasks()
    graph = load_graph()
    top_tasks = get_top_task(tasks_db, top_n=1)
    for task_id, task in top_tasks:
        facts = gather_supporting_facts(graph, task_id)
        report = generate_report(task, facts, tasks_db, graph)
        save_report(report, task_id)

if __name__ == "__main__":
    main()
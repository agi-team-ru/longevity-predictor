import pickle
import networkx as nx
import matplotlib.pyplot as plt
import textwrap
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder

with open('full_longevity_graph.pkl', 'rb') as f:
    graph = pickle.load(f)
with open('tasks_db.json', 'r', encoding='utf-8') as f:
    tasks_db = json.load(f)

task_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'task']
article_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'article']
tag_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'tag']
sub_nodes = set()
for t in task_nodes:
    for a in graph.neighbors(t):
        if graph.nodes[a].get('type') in ('article', 'tag'):
            sub_nodes.add(t)
            sub_nodes.add(a)
subg = graph.subgraph(sub_nodes)

plt.figure(figsize=(18, 14))
# Кластеризация по category (цвет)
categories = [tasks_db[n]['category'] if n in tasks_db else 'other' for n in task_nodes]
le = LabelEncoder()
cat_labels = le.fit_transform(categories)
cat2color = {cat: plt.cm.tab20(i % 20) for i, cat in enumerate(le.classes_)}

# UMAP-проекция, если есть эмбеддинги
try:
    import umap
    task_embeds = np.array([tasks_db[n]['embedding'] for n in task_nodes if n in tasks_db])
    reducer = umap.UMAP(n_components=2, random_state=42)
    task_pos = reducer.fit_transform(task_embeds)
    pos = {}
    for i, n in enumerate(task_nodes):
        if n in tasks_db:
            pos[n] = task_pos[i]
    # Для остальных узлов — spring_layout
    other_nodes = [n for n in subg.nodes if n not in pos]
    pos.update(nx.spring_layout(subg.subgraph(other_nodes), k=1.2, seed=42))
except Exception:
    pos = nx.spring_layout(subg, k=1.2, seed=42)

# Размер и цвет задач
task_sizes = []
task_colors = []
for n in task_nodes:
    d = tasks_db.get(n)
    if d:
        score = d.get('score', 0.1)
        task_sizes.append(600 + 2000 * score)
        task_colors.append(cat2color.get(d.get('category', 'other'), (0.5, 0.5, 0.5, 1)))
    else:
        task_sizes.append(400)
        task_colors.append((0.5, 0.5, 0.5, 1))

nx.draw_networkx_nodes(subg, pos, nodelist=task_nodes, node_color=task_colors, node_size=task_sizes, label='Tasks')
nx.draw_networkx_nodes(subg, pos, nodelist=article_nodes, node_color='blue', node_size=300, label='Articles')
nx.draw_networkx_nodes(subg, pos, nodelist=tag_nodes, node_color='green', node_size=200, label='Tags')
nx.draw_networkx_edges(subg, pos, alpha=0.3)

# Подписи
labels = {}
for n in subg.nodes:
    if n in task_nodes:
        d = tasks_db.get(n)
        if d:
            meta = f"cat: {d.get('category','?')}, mat: {d.get('maturity','?')}, imp: {d.get('impact','?')}, score: {d.get('score','?')}, n_articles: {len([x for x in graph.neighbors(n) if graph.nodes[x].get('type') == 'article'])}"
            task_text = d['found_tasks'][0] if d['found_tasks'] else n
            label = f"{task_text}\n{meta}"
        else:
            # Попробуем взять название первой соседней статьи
            neighbors = [x for x in subg.neighbors(n) if subg.nodes[x].get('type') == 'article']
            if neighbors:
                label = f"{neighbors[0]}\n(id: {n})"
            else:
                label = n
        labels[n] = label
    elif n in article_nodes:
        wrapped = '\n'.join(textwrap.wrap(n, width=60))
        labels[n] = wrapped
    elif n in tag_nodes:
        labels[n] = n
nx.draw_networkx_labels(subg, pos, labels=labels, font_size=6, font_color='black')

# Легенда
for i, cat in enumerate(le.classes_):
    plt.scatter([], [], color=cat2color[cat], label=cat)
plt.scatter([], [], color='blue', label='Article')
plt.scatter([], [], color='green', label='Tag')
plt.legend(scatterpoints=1, fontsize=10, loc='upper right', frameon=True)

plt.title('Clustered Task-Article-Tag Graph Visualization')
plt.tight_layout()
plt.show()

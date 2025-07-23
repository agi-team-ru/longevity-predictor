import copy
import json
import networkx as nx
import numpy as np
import os
from datetime import datetime
import uuid
import pickle
import requests
from collections import Counter
from tqdm import tqdm
from openai import OpenAI
from scorer_v2 import TopicScorer

LLM_LOG_PATH = "llm_log.jsonl"
LLM_CACHE_PATH = "llm_cache.json"

# Загрузка кеша LLM
if os.path.exists(LLM_CACHE_PATH):
    with open(LLM_CACHE_PATH, "r", encoding="utf-8") as f:
        llm_cache = json.load(f)
else:
    llm_cache = {}

def log_llm_interaction(prompt, response):
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'prompt': prompt,
        'response': response
    }
    with open(LLM_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

def save_llm_cache():
    with open(LLM_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(llm_cache, f, ensure_ascii=False, indent=2)

def verif_article_and_extract_task(abstract:str, temperature:float = 0.2) -> dict:
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

    if prompt in llm_cache:
        raw = llm_cache[prompt]
    else:
        response = client.chat.completions.create(
            model="llama-3.3-70b-instruct",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=temperature
        )
        raw = response.choices[0].message.content.strip()
        llm_cache[prompt] = raw
        save_llm_cache()

    data = json.loads(raw)
    log_llm_interaction(prompt, raw)

    maturity_map = {"high", "medium", "low", "unknown", "preclinical", "animal", "clinical", "approved", "review", "other"}
    if data.get('maturity') not in maturity_map:
        data['maturity'] = 'unknown'

    return data

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

JOURNAL_WEIGHT = {
    "Nature":                        1.00,
    "Science":                       1.00,
    "Cell":                          1.00,
    "Nature Medicine":               0.95,
    "Nature Biotechnology":          0.94,
    "Nature Metabolism":             0.92,
    "Nature Aging":                  0.90,
    "Science Translational Medicine":0.90,
    "Cell Metabolism":               0.88,
    "Cell Stem Cell":                0.88,
    "Aging Cell":                    0.80,
    "Aging":                         0.70,
    "Geroscience":                   0.68,
    "Mechanisms of Ageing and Development": 0.60,
    "npj Aging":                     0.60,
    "Genome Research":               0.74,
    "Genome Biology":                0.74,
    "Epigenetics & Chromatin":       0.65,
    "EMBO Molecular Medicine":       0.78,
    "Autophagy":                     0.82,
    "Redox Biology":                 0.75,
    "Free Radical Biology & Medicine":0.70,
    "Cell Death & Disease":          0.72,
    "Journals of Gerontology Series A": 0.65,
    "Experimental Gerontology":      0.55,
    "Age and Ageing":                0.55,
    "eLife":                         0.76,
    "EMBO Journal":                  0.82,
    "PLOS Biology":                  0.70,
    "iScience":                      0.55,
    "bioRxiv":                       0.30,
    "Research Square":               0.25
}

def estimate_impact(task: str, graph: nx.Graph) -> float:
    # Статьи, связанные с задачей (соседи типа article)
    articles = [n for n in graph.neighbors(task) if graph.nodes[n].get('type') == 'article']
    if not articles:
        return 0.0
    # Максимальный вес журнала среди связанных статей (только по полному совпадению)
    max_weight = 0.0
    for art in articles:
        journal = graph.nodes[art].get('journal', '').strip()
        for j, w in JOURNAL_WEIGHT.items():
            if j.strip() == journal:
                max_weight = max(max_weight, w)
    return max_weight

def collect_data_for_confidence_scoring(task: str, graph: nx.Graph) -> dict:
    """
    Собирает данные для вычисления confidence score:
    - абстракты статей
    - источники статей
    - типы источников
    - годы публикации
    """
    articles = [n for n in graph.neighbors(task) if graph.nodes[n].get('type') == 'article']
    
    if not articles:
        return {
            'abstracts': [],
            'sources': [],
            'source_types': [],
            'years': []
        }
    
    abstracts = []
    sources = []
    source_types = []
    years = []
    
    for article in articles:
        # Абстракт
        abstract = graph.nodes[article].get('abstract', '')
        if abstract:
            abstracts.append(abstract)
        
        # Источник (журнал)
        journal = graph.nodes[article].get('journal', '')
        if journal:
            sources.append(journal)
        
        # Тип источника (определяем по названию журнала)
        source_type = 'research_paper'  # по умолчанию
        journal_lower = journal.lower()
        if 'preprint' in journal_lower or 'arxiv' in journal_lower or 'biorxiv' in journal_lower:
            source_type = 'preprint'
        elif 'review' in journal_lower:
            source_type = 'systematic_review'
        elif 'news' in journal_lower or 'press' in journal_lower:
            source_type = 'news_article'
        source_types.append(source_type)
        
        # Год публикации
        pubdate = graph.nodes[article].get('pubdate', '')
        if pubdate and pubdate != '0000-00-00':
            try:
                year = int(pubdate.split('-')[0])
                years.append(year)
            except (ValueError, IndexError):
                years.append(2024)  # по умолчанию
        else:
            years.append(2024)  # по умолчанию
    
    return {
        'abstracts': abstracts,
        'sources': sources,
        'source_types': source_types,
        'years': years
    }

def compute_confidence_score(task: str, graph: nx.Graph, task_text: str) -> float:
    """
    Вычисляет confidence score используя TopicScorer
    """
    try:
        # Собираем данные для скоринга
        data = collect_data_for_confidence_scoring(task, graph)
        
        if not data['abstracts']:
            return 0.0
        
        # Берем первый абстракт как основной
        main_abstract = data['abstracts'][0]
        
        # Остальные абстракты как supporting articles
        other_abstracts = data['abstracts'][1:] if len(data['abstracts']) > 1 else []
        
        # Средний год публикации
        avg_year = int(np.mean(data['years'])) if data['years'] else 2024
        
        # Создаем TopicScorer и вычисляем score
        scorer = TopicScorer()
        confidence_score = scorer.calculate_final_score(
            article_abstract=main_abstract,
            other_articles=other_abstracts,
            article_sources_types=data['source_types'],
            article_sources=data['sources'],
            publication_year=avg_year
        )
        
        return round(confidence_score, 3)
        
    except Exception as e:
        print(f"Error computing confidence score for task {task}: {e}")
        return 0.0

def compute_score(maturity: str, impact: float, freq: int) -> float:
    maturity_map = {"high": 1.0, "medium": 0.6, "low": 0.3, "unknown": 0.1,
                   "preclinical": 0.3, "animal": 0.4, "clinical": 0.7,
                   "approved": 1.0, "review": 0.2, "other": 0.1}
    score = 0.5 * maturity_map.get(maturity, 0.1) + 0.3 * impact + 0.2 * min(freq/5, 1.0)
    return round(score, 3)

def update_graph(graph:nx.Graph, task:str, found_task:str, article:dict, tags:dict, category:str, maturity:str, impact:float, score:float):
    if task not in graph:
        graph.add_node(task, type='task')
        graph.nodes[task]['found_tasks'] = [found_task]
        graph.nodes[task]['category'] = category
        graph.nodes[task]['maturity'] = maturity
        graph.nodes[task]['impact'] = impact
        graph.nodes[task]['score'] = score
    else:
        graph.nodes[task]['found_tasks'].append(found_task)
    title = article['title']
    graph.add_node(title, type='article')
    for k, v in article.items():
        if k != 'title':
            graph.nodes[title][k] = v
    graph.add_edge(task, title)
    for ent, label in tags.items():
        if ent not in graph:
            graph.add_node(ent, type='tag')
            graph.nodes[ent]['label'] = label
        graph.add_edge(task, ent)

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key"
)

# embedder = HuggingFaceEmbeddings(
#     model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
# )

graph = nx.Graph()

tasks_db = {}

def add_task(task_text, embedding, meta):
    # Поиск похожей задачи по эмбеддингу (через FAISS)
    # Если похожая найдена:
    #   task_id = найденный
    #   tasks_db[task_id]['found_tasks'].append(task_text)
    #   ... (обновить метаданные)
    # Если нет:
    task_id = f'task_{uuid.uuid4().hex[:8]}'
    tasks_db[task_id] = {
        'found_tasks': [task_text],
        'embedding': embedding,
        **meta
    }
    graph.add_node(task_id, type='task')
    return task_id

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_existing_task(embedding, tasks_db, threshold=0.85):
    best_id, best_sim = None, -1
    for tid, tdata in tasks_db.items():
        sim = cosine_similarity(embedding, tdata['embedding'])
        if sim > best_sim:
            best_sim = sim
            best_id = tid
    if best_sim >= threshold:
        return best_id
    return None

def get_embedding_via_api(text, url="http://localhost:8001/embed"):
    resp = requests.post(url, json={"text": text})
    resp.raise_for_status()
    return resp.json()["embedding"]

with open(r'pubmed_articles.json', 'r', encoding='utf-8') as f:
    articles = json.load(f)

for art in tqdm(articles[:100]):
    data = verif_article_and_extract_task(art['abstract'])
    if data['relevant']:
        tags = get_tags_via_api(art['abstract'])
        embedding = get_embedding_via_api(data['task'])
        category = data.get('category', 'unknown')
        maturity = data.get('maturity', 'unknown')
        meta = {'category': category, 'maturity': maturity, 'impact': 0.0, 'score': 0.0}
        existing_id = find_existing_task(embedding, tasks_db)
        if existing_id:
            task_id = existing_id
            # Всегда добавлять новую формулировку (допускаются дубли)
            tasks_db[task_id]['found_tasks'].append(data['task'])
        else:
            task_id = add_task(data['task'], embedding, meta)
        title = art['title']
        graph.add_node(title, type='article')
        for k, v in art.items():
            if k != 'title':
                graph.nodes[title][k] = v
        graph.add_edge(task_id, title)
        for ent, label in tags.items():
            if ent not in graph:
                graph.add_node(ent, type='tag')
                graph.nodes[ent]['label'] = label
            graph.add_edge(task_id, ent)

# После построения графа — вычисляем impact, score и confidence_score для всех задач
for node, attrs in graph.nodes(data=True):
    if attrs.get('type') == 'task':
        freq = len([n for n in graph.neighbors(node) if graph.nodes[n].get('type') == 'article'])
        impact = estimate_impact(node, graph)
        maturity = tasks_db[node].get('maturity', 'unknown')
        
        # Вычисляем старый score для совместимости
        score = compute_score(maturity, impact, freq)
        
        # Вычисляем новый confidence_score
        task_text = tasks_db[node]['found_tasks'][0] if tasks_db[node]['found_tasks'] else ''
        confidence_score = compute_confidence_score(node, graph, task_text)
        
        tasks_db[node]['impact'] = impact
        tasks_db[node]['score'] = score
        tasks_db[node]['confidence_score'] = confidence_score

# После построения графа и расчёта метрик
with open("full_longevity_graph.pkl", "wb") as f:
    import pickle
    pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)

# Сохраняем tasks_db в json (без эмбеддингов)
tasks_db_json = copy.deepcopy(tasks_db)
for v in tasks_db_json.values():
    if 'embedding' in v:
        v.pop('embedding')
with open("tasks_db.json", "w", encoding="utf-8") as f:
    json.dump(tasks_db_json, f, ensure_ascii=False, indent=2)

# Эмбеддинги отдельно (если нужно)
embeds = {k: v['embedding'] for k, v in tasks_db.items() if 'embedding' in v}
with open("tasks_db_embeds.pkl", "wb") as f:
    import pickle
    pickle.dump(embeds, f, protocol=pickle.HIGHEST_PROTOCOL)
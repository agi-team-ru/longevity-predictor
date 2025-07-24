import numpy as np
import networkx as nx


import math
from typing import List
from datetime import datetime

from pydantic import BaseModel, Field
import json

import openai

import os

LLM_MODEL = os.getenv("LLM_MODEL")

def get_model_response(
    prompt: str,
    model_name: str = "model_base"
) -> str:
    client = openai.OpenAI()
    # Формируем сообщение для модели
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages
    )
    # Извлекаем текст ответа
    if hasattr(response, "choices") and len(response.choices) > 0:
        message = response.choices[0].message
        if hasattr(message, "content"):
            return message.content
    return ""

def extract_json_from_text(text: str) -> List[str]:
    start_json_char = "{"
    decoder = json.JSONDecoder(strict=False)
    pos = 0
    ret: List[str] = []
    while True:
        start_char_pos = text.find(start_json_char, pos)
        if start_char_pos < 0:
            break
        try:
            result, index = decoder.raw_decode(text[start_char_pos:])
            pos = start_char_pos + index
            ret.append(json.dumps(result, ensure_ascii=False))
        except ValueError:
            pos = start_char_pos + 1
    return ret

class ArticleToScore(BaseModel):
    abstract: str = Field(..., description="Краткое описание")

    other_articles: List[str] = Field(..., description="Статьи, которые смежные с тем, что есть в графе")
    article_sources_types: List[str] = Field(..., description="Типы источников, например СМИ, Research, X, пр.")
    article_sources: List[str] = Field(..., description="Список источников, например Arxiv")
    publication_year: int = Field(..., description="Год публикации")


PROMPT = """
You are an expert in analyzing scientific articles.

Your task:
1. Carefully read the content of the main article and the other articles.
2. Analyze which of them contradict the main article, and which ones agree with it.
3. List the numbers of the articles that CONTRADICT the main article, and those that HAVE THE SAME CONCLUSION as the main article.

NOTE: Articles CONTRADICT each other if they ASSERT DIFFERENT things about the same phenomenon/experiment/concept/result/etc.
NOTE: Articles AGREE with each other if they INTERPRET SIMILAR DEFINITIONS about the phenomenon/experiment/concept/result/etc.

You can add a short explanation for why you think the articles agree or contradict each other.

Reply in the following format:
<answer>
{{
    "agree": [1, 2, 3, 4, ...],
    "disagree": [6, 7, 8, 9, ...]
}}
</answer>

Главная статья:
<main_article>
{main_article}
</main_article>

Статьи:
<articles>
{article_analysis}
</articles>
"""


class TopicScorer:
    def __init__(self):
        # Маппинг типов источников по приоритетности
        self.source_weights = {
            'research_paper': 1.0,
            'patent': 1.0,
            'systematic_review': 1.2,
            'industry_report': 0.8,
            'news_article': 0.4,
            'preprint': 0.3,
            'blog_post': 0.1
        }
        
        # Маппинг качества журналов/источников
        self.journal_weights = {
            'nature': 1.0,
            'lancet': 1.0,
            'science': 1.0,
            'cell': 0.9,
            'pnas': 0.8,
            'nejm': 0.9,
            'arxiv': 0.5,
            'biorxiv': 0.4,
            'unknown': 0.2
        }
        
        # Коэффициент затухания для актуальности (λ)
        self.decay_lambda = 0.3
        
        # Веса для финальной агрегации
        self.aggregation_weights = {
            'sources': 0.25,
            'diversity': 0.20,
            'agreement': 0.15,
            'quality': 0.25,
            'recency': 0.15
        }

    def calculate_confirm_score(self, confirm_articles: List[str]) -> float:
        # Количество подтверждающих источников
        sources_count = len(confirm_articles)
        
        if sources_count == 0:
            return 0.0
            
        score = math.log(1 + sources_count)
        return score

    def calculate_source_score(self, source_types: List[str]) -> float:
        # Разнообразие источников с учетом весов
        if not source_types:
            return 0.0
            
        unique_types = set(source_types)
        total_weight = 0.0
        
        for source_type in unique_types:
            weight = self.source_weights.get(source_type.lower(), 0.1)
            total_weight += weight
            
        # Бонус за разнообразие типов
        diversity_bonus = len(unique_types) * 0.1
        
        final_score = total_weight + diversity_bonus
        return final_score

    def calculate_acceptance_score(self, article_abstract: str, other_articles: List[str]) -> float:
        # Источники (краткое содержание)
        same_themed_articles: List[str] = other_articles

        prompt = PROMPT.format(
            main_article=article_abstract,
            article_analysis="\n\n".join([f"{str(idx)}. " + art for idx, art in enumerate(same_themed_articles)])
        )

        response = get_model_response(prompt)
        json_extract = extract_json_from_text(response)
        if not json:
            return 0.0
        
        answer = json.loads(json_extract[-1])
        agree = answer.get("agree", [])
        disagree = answer.get("disagree", [])

        agree_count = len(agree)
        disagree_count = len(disagree)

        total = agree_count + disagree_count
        if total == 0:
            agreement_score = 0.0
        else:
            # Балансируем: если agree = disagree, то 0; если disagree больше, то от -1 до 0; если agree больше, то от 0 до 0.2 (максимум)
            balance = (agree_count - (5 * disagree_count)) / total
            if balance >= 0:
                agreement_score = min(balance * 0.2, 1)
            else:
                agreement_score = max(balance, -10.0)

        print("analysis |", f"agrees: {len(agree)} | disagrees: {len(disagree)} | agreement score {agreement_score}")

        return agreement_score


    def calculate_quality_score(self, source_articles: List[str]) -> float:
        # Качество данных на основе журналов/источников
        if not source_articles:
            return 0.0
            
        total_weight = 0.0
        
        for article in source_articles:
            article_lower = article.lower()
            weight = self.journal_weights.get('unknown', 0.2)
            
            # Поиск известных журналов в названии статьи
            for journal, journal_weight in self.journal_weights.items():
                if journal in article_lower:
                    weight = journal_weight
                    break
                    
            total_weight += weight
            
        average_quality = total_weight / len(source_articles)
        return average_quality

    def calculate_actuality_score(self, year: int) -> float:
        # Актуальность данных с коэффициентом затухания
        current_year = datetime.now().year
        
        if year > current_year:
            year = current_year
            
        years_difference = current_year - year
        
        # Применяем экспоненциальное затухание
        decay_factor = math.exp(-self.decay_lambda * years_difference)
        
        return decay_factor

    def calculate_sigmoid(self, x: float) -> float:
        # Сигмоидальная функция для нормализации
        try:
            result = 1 / (1 + math.exp(-x + 0.25))
        except OverflowError:
            result = 0.0 if x < 0 else 1.0
            
        return result

    def calculate_final_score(
        self,
        article_abstract: str, # Сама статья
        other_articles: List[str], # Статьи, которые подтверждают (неважно что внутри)
        article_sources_types: List[str], # Типы источников (газета, СМИ)
        article_sources: List[str], # Источники статьи (arxiv, biorxiv)
        publication_year: int # Год публикации
    ) -> float:
        # Вычисляем все компоненты
        sources_score = self.calculate_confirm_score(other_articles)
        diversity_score = self.calculate_source_score(article_sources_types)
        agreement_score = self.calculate_acceptance_score(article_abstract, other_articles)
        quality_score = self.calculate_quality_score(article_sources)
        recency_score = self.calculate_actuality_score(publication_year)
        
        # Взвешенная сумма всех компонентов
        weighted_sum = (
            self.aggregation_weights['sources'] * sources_score +
            self.aggregation_weights['diversity'] * diversity_score +
            self.aggregation_weights['agreement'] * agreement_score +
            self.aggregation_weights['quality'] * quality_score +
            self.aggregation_weights['recency'] * recency_score
        )
        
        # Нормализуем через сигмоид
        final_confidence = self.calculate_sigmoid(weighted_sum)

        
        verbose = True
        if verbose:
            # Формируем таблицу для красивого вывода
            components = [
                ("sources_score", sources_score, self.aggregation_weights['sources']),
                ("diversity_score", diversity_score, self.aggregation_weights['diversity']),
                ("agreement_score", agreement_score, self.aggregation_weights['agreement']),
                ("quality_score", quality_score, self.aggregation_weights['quality']),
                ("recency_score", recency_score, self.aggregation_weights['recency']),
            ]


            # Заголовок таблицы
            print(f"{'name':<16} | {'value':<7} | {'impact'}")
            print("-" * 36)
            # Выводим каждую строку
            for name, value, impact in components:
                # impact в процентах
                impact_percent = f"{impact * 100:.0f}%"
                print(f"{name:<16} | {value:<7.3f} | {impact_percent}")

            print("-" * 36)
            print(f"{'weighted_sum':<16} | {weighted_sum:<7.3f} |")
            print(f"{'final_confidence':<16} | {final_confidence:<7.3f} |")
        
        return final_confidence


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
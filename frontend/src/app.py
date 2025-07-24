import streamlit as st
import pickle
import json
import networkx as nx
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
import os
from neo4j import GraphDatabase

# Конфигурация страницы
st.set_page_config(
    page_title="Longevity Research Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Кастомные стили
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data_from_neo4j():
    NEO4J_HOST = os.getenv("NEO4J_HOST", "localhost")
    NEO4J_URI = os.getenv("NEO4J_URI", f"bolt://{NEO4J_HOST}:7687")
    driver = GraphDatabase.driver(NEO4J_URI, auth=("", ""))
    with driver.session() as session:
        # Загрузка задач
        tasks = session.run(
            """
            MATCH (t:Task)
            OPTIONAL MATCH (t)-[:MENTIONED_IN]->(a:Article)
            RETURN id(t) as id, t.text as text, t.category as category, t.maturity as maturity, 
                   t.score as score, t.impact as impact, t.found_tasks as found_tasks, collect(id(a)) as articles
            """
        ).data()
        tasks_db = {}
        for row in tasks:
            tasks_db[row["id"]] = {
                "text": row["text"],
                "category": row.get("category", "unknown"),
                "maturity": row.get("maturity", "unknown"),
                "score": row.get("score", 0),
                "impact": row.get("impact", 0),
                "found_tasks": row.get("found_tasks", []),
                "articles": row.get("articles", [])
            }
        # Загрузка статей
        articles = session.run(
            """
            MATCH (a:Article)
            RETURN id(a) as id, a.title as title, a.abstract as abstract, a.journal as journal, 
                   a.maturity as maturity, a.pubdate as pubdate, a.confidence_score as confidence_score
            """
        ).data()
        articles_dict = []
        for row in articles:
            articles_dict.append({
                "id": row["id"],
                "title": row["title"],
                "abstract": row["abstract"],
                "journal": row["journal"],
                "maturity": row["maturity"],
                "pubdate": row["pubdate"],
                "confidence_score": row["confidence_score"]
            })
        # Загрузка графа (nodes/edges)
        nodes = session.run(
            """
            MATCH (n)
            RETURN id(n) as id, labels(n) as labels, n
            """
        ).data()
        edges = session.run(
            """
            MATCH (n)-[r]->(m)
            RETURN id(n) as source, type(r) as type, id(m) as target
            """
        ).data()
    driver.close()
    return nodes, edges, tasks_db, articles_dict

def create_network_visualization(graph, tasks_db, task_filter=None, category_filter=None):
    """Создает интерактивную визуализацию сети"""
    # Фильтрация узлов
    nodes_to_include = set()
    
    task_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'task']
    
    for task in task_nodes:
        if task in tasks_db:
            task_data = tasks_db[task]
            
            # Фильтр по категории
            if category_filter and task_data.get('category') != category_filter:
                continue
                
            # Фильтр по тексту задачи
            if task_filter:
                found_tasks = task_data.get('found_tasks', [])
                if not any(task_filter.lower() in task_text.lower() for task_text in found_tasks):
                    continue
            
            nodes_to_include.add(task)
            # Добавляем соседей
            for neighbor in graph.neighbors(task):
                nodes_to_include.add(neighbor)
    
    if not nodes_to_include:
        return None
    
    subgraph = graph.subgraph(nodes_to_include)
    
    # Получение позиций узлов
    try:
        pos = nx.spring_layout(subgraph, k=3, iterations=50, seed=42)
    except:
        pos = {node: (0, 0) for node in subgraph.nodes()}
    
    # Подготовка данных для Plotly
    edge_x, edge_y = [], []
    for edge in subgraph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Создание следов для узлов по типам
    node_traces = {}
    
    for node in subgraph.nodes():
        node_type = graph.nodes[node].get('type', 'unknown')
        x, y = pos[node]
        
        if node_type not in node_traces:
            node_traces[node_type] = {'x': [], 'y': [], 'text': [], 'size': [], 'color': []}
        
        node_traces[node_type]['x'].append(x)
        node_traces[node_type]['y'].append(y)
        
        # Размер и цвет узла
        if node_type == 'task' and node in tasks_db:
            task_data = tasks_db[node]
            size = 20 + task_data.get('score', 0) * 30
            color_map = {
                'senescence': '#FF6B6B',
                'metabolism': '#4ECDC4', 
                'telomere biology': '#45B7D1',
                'mitochondria': '#96CEB4',
                'epigenetics': '#FFEAA7',
                'immunology': '#DDA0DD',
                'cellular aging': '#98D8C8'
            }
            color = color_map.get(task_data.get('category', 'unknown'), '#BDC3C7')
            
            # Информация для всплывающих подсказок
            found_tasks = task_data.get('found_tasks', [])
            task_text = found_tasks[0] if found_tasks else node
            tooltip = f"Task: {task_text}<br>Category: {task_data.get('category', 'unknown')}<br>Score: {task_data.get('score', 0)}<br>Maturity: {task_data.get('maturity', 'unknown')}"
            
        elif node_type == 'article':
            size = 8
            color = '#3498DB'
            tooltip = f"Article: {node[:100]}..."
            
        elif node_type == 'tag':
            size = 6
            color = '#2ECC71'
            tooltip = f"Tag: {node}"
            
        else:
            size = 10
            color = '#95A5A6'
            tooltip = f"Node: {node}"
        
        node_traces[node_type]['size'].append(size)
        node_traces[node_type]['color'].append(color)
        node_traces[node_type]['text'].append(tooltip)
    
    # Создание фигуры
    fig = go.Figure()
    
    # Добавление ребер
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines',
        name='Connections'
    ))
    
    # Добавление узлов по типам
    type_colors = {'task': '#E74C3C', 'article': '#3498DB', 'tag': '#2ECC71'}
    type_names = {'task': 'Research Tasks', 'article': 'Articles', 'tag': 'Biomedical Tags'}
    
    for node_type, trace_data in node_traces.items():
        if trace_data['x']:  # Если есть узлы этого типа
            fig.add_trace(go.Scatter(
                x=trace_data['x'], y=trace_data['y'],
                mode='markers',
                hoverinfo='text',
                text=trace_data['text'],
                marker=dict(
                    size=trace_data['size'],
                    color=trace_data['color'],
                    line=dict(width=1, color='white')
                ),
                name=type_names.get(node_type, node_type.title())
            ))
    
    fig.update_layout(
        title={
            'text': "Knowledge Graph: Research Tasks, Articles & Biomedical Entities",
            'font': {'size': 16}
        },
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[ dict(
            text="Network visualization of longevity research priorities",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002,
            xanchor='left', yanchor='bottom',
            font=dict(color='#888', size=12)
        )],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white'
    )
    
    return fig

def create_metrics_dashboard(tasks_db, articles):
    """Создает дашборд с основными метриками"""
    
    # Подготовка данных
    categories = [task_data.get('category', 'unknown') for task_data in tasks_db.values()]
    maturities = [task_data.get('maturity', 'unknown') for task_data in tasks_db.values()]
    scores = [task_data.get('score', 0) for task_data in tasks_db.values()]
    impacts = [task_data.get('impact', 0) for task_data in tasks_db.values()]
    
    # Создание подграфиков
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Research Categories Distribution', 'Maturity Levels', 
                       'Score Distribution', 'Impact Distribution'),
        specs=[[{"type": "pie"}, {"type": "bar"}],
               [{"type": "histogram"}, {"type": "histogram"}]]
    )
    
    # 1. Категории (круговая диаграмма)
    category_counts = Counter(categories)
    fig.add_trace(go.Pie(
        labels=list(category_counts.keys()),
        values=list(category_counts.values()),
        name="Categories"
    ), row=1, col=1)
    
    # 2. Уровни зрелости (столбчатая диаграмма)
    maturity_counts = Counter(maturities)
    fig.add_trace(go.Bar(
        x=list(maturity_counts.keys()),
        y=list(maturity_counts.values()),
        name="Maturity"
    ), row=1, col=2)
    
    # 3. Распределение оценок
    fig.add_trace(go.Histogram(
        x=scores,
        nbinsx=20,
        name="Scores"
    ), row=2, col=1)
    
    # 4. Распределение импакта
    fig.add_trace(go.Histogram(
        x=impacts,
        nbinsx=20,
        name="Impact"
    ), row=2, col=2)
    
    fig.update_layout(
        title="Research Analytics Dashboard",
        height=600,
        showlegend=False
    )
    
    return fig

def create_timeline_analysis(articles):
    """Создает анализ временных трендов публикаций"""
    # Парсинг дат
    dates = []
    journals = []
    
    for article in articles:
        try:
            pub_date = article.get('pubdate', '')
            if pub_date:
                # Пытаемся парсить дату
                if len(pub_date) >= 4:
                    year = int(pub_date[:4])
                    if 2000 <= year <= 2025:
                        dates.append(year)
                        journals.append(article.get('journal', 'Unknown'))
        except:
            continue
    
    if not dates:
        return None
    
    # Создание DataFrame
    df = pd.DataFrame({'year': dates, 'journal': journals})
    
    # Группировка по годам
    yearly_counts = df.groupby('year').size().reset_index(name='count')
    
    # График временного тренда
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=yearly_counts['year'],
        y=yearly_counts['count'],
        mode='lines+markers',
        name='Publications',
        line=dict(width=3, color='#1f77b4'),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="Publication Timeline in Longevity Research",
        xaxis_title="Year",
        yaxis_title="Number of Publications",
        height=400,
        template='plotly_white'
    )
    
    return fig

def create_research_analysis(tasks_db):
    """Создает анализ исследовательских задач"""
    
    # Подготовка данных
    data = []
    for task_id, task_data in tasks_db.items():
        for task_text in task_data.get('found_tasks', []):
            data.append({
                'task_id': task_id,
                'task_text': task_text,
                'category': task_data.get('category', 'unknown'),
                'maturity': task_data.get('maturity', 'unknown'),
                'score': task_data.get('score', 0),
                'impact': task_data.get('impact', 0)
            })
    
    df = pd.DataFrame(data)
    
    # Scatter plot: Score vs Impact
    fig = px.scatter(
        df, 
        x='impact', 
        y='score', 
        color='category',
        size='score',
        hover_data=['task_text', 'maturity'],
        title="Research Task Analysis: Score vs Impact by Category",
        labels={'impact': 'Journal Impact Factor', 'score': 'Research Priority Score'}
    )
    
    fig.update_layout(height=500)
    
    return fig

def main():
    # Заголовок
    st.markdown('<div class="main-header">🧬 Longevity Research Dashboard</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Загрузка данных
    with st.spinner("Загрузка данных..."):
        nodes, edges, tasks_db, articles = load_data_from_neo4j()
    
    if not nodes or not edges:
        st.error("Не удалось загрузить данные из Neo4j. Проверьте подключение и наличие данных.")
        return
    
    # Соберите NetworkX граф из загруженных узлов и ребер
    graph = nx.Graph()
    for node_data in nodes:
        node_id = node_data["id"]
        node_labels = node_data["labels"]
        node_properties = node_data["n"]
        node_type = node_labels[0] if node_labels else "unknown"
        graph.add_node(node_id, type=node_type, **node_properties)

    for edge_data in edges:
        source_id = edge_data["source"]
        edge_type = edge_data["type"]
        target_id = edge_data["target"]
        graph.add_edge(source_id, target_id, type=edge_type)

    if not graph.nodes():
        st.warning("Не удалось создать граф из загруженных данных.")
        return
    
    # Sidebar с фильтрами
    st.sidebar.title("🔍 Filters & Settings")
    
    # Основная статистика в sidebar
    st.sidebar.markdown("### 📊 Overview")
    st.sidebar.metric("Research Tasks", len(tasks_db))
    st.sidebar.metric("Articles", len(articles))
    st.sidebar.metric("Total Nodes", graph.number_of_nodes())
    st.sidebar.metric("Connections", graph.number_of_edges())
    
    # Фильтры
    categories = list(set(task_data.get('category', 'unknown') for task_data in tasks_db.values()))
    category_filter = st.sidebar.selectbox("Category Filter", ["All"] + sorted(categories))
    
    task_filter = st.sidebar.text_input("Search Tasks", placeholder="Enter keywords...")
    
    # Основные вкладки
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🕸️ Knowledge Graph", 
        "📊 Analytics Dashboard", 
        "📈 Timeline Analysis",
        "🔬 Research Analysis",
        "📋 Data Explorer"
    ])
    
    with tab1:
        st.subheader("Interactive Knowledge Graph")
        st.markdown("Explore the network of research tasks, articles, and biomedical entities.")
        
        # Применение фильтров
        cat_filter = None if category_filter == "All" else category_filter
        task_search = task_filter if task_filter else None
        
        with st.spinner("Создание визуализации..."):
            network_fig = create_network_visualization(graph, tasks_db, task_search, cat_filter)
            
        if network_fig:
            st.plotly_chart(network_fig, use_container_width=True)
        else:
            st.warning("Нет данных для отображения с текущими фильтрами.")
    
    with tab2:
        st.subheader("Research Analytics Dashboard")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_score = np.mean([task_data.get('score', 0) for task_data in tasks_db.values()])
            st.metric("Avg Score", f"{avg_score:.3f}")
        
        with col2:
            high_impact = sum(1 for task_data in tasks_db.values() if task_data.get('impact', 0) > 0.5)
            st.metric("High Impact Tasks", high_impact)
        
        with col3:
            clinical_tasks = sum(1 for task_data in tasks_db.values() if task_data.get('maturity') == 'clinical')
            st.metric("Clinical Stage", clinical_tasks)
        
        with col4:
            top_category = Counter(task_data.get('category', 'unknown') for task_data in tasks_db.values()).most_common(1)
            if top_category:
                st.metric("Top Category", top_category[0][0])
        
        # Дашборд метрик
        metrics_fig = create_metrics_dashboard(tasks_db, articles)
        st.plotly_chart(metrics_fig, use_container_width=True)
    
    with tab3:
        st.subheader("Publication Timeline Analysis")
        
        timeline_fig = create_timeline_analysis(articles)
        if timeline_fig:
            st.plotly_chart(timeline_fig, use_container_width=True)
        else:
            st.warning("Недостаточно данных о датах публикаций для анализа.")
        
        # Топ журналы
        st.subheader("Top Journals")
        journal_counts = Counter(article.get('journal', 'Unknown') for article in articles)
        top_journals = journal_counts.most_common(10)
        
        if top_journals:
            journal_df = pd.DataFrame(top_journals, columns=['Journal', 'Publications'])
            st.dataframe(journal_df, use_container_width=True)
    
    with tab4:
        st.subheader("Research Task Analysis")
        
        research_fig = create_research_analysis(tasks_db)
        st.plotly_chart(research_fig, use_container_width=True)
        
        # Топ задачи по оценке
        st.subheader("Top Scoring Research Tasks")
        
        top_tasks = []
        for task_id, task_data in tasks_db.items():
            score = task_data.get('score', 0)
            if task_data.get('found_tasks'):
                task_text = task_data['found_tasks'][0]
                top_tasks.append({
                    'Task': task_text,
                    'Category': task_data.get('category', 'unknown'),
                    'Score': score,
                    'Maturity': task_data.get('maturity', 'unknown')
                })
        
        top_tasks_df = pd.DataFrame(top_tasks).sort_values('Score', ascending=False).head(10)
        st.dataframe(top_tasks_df, use_container_width=True)
    
    with tab5:
        st.subheader("Data Explorer")
        
        data_option = st.selectbox("Choose data to explore:", [
            "Research Tasks", "Articles", "Graph Statistics"
        ])
        
        if data_option == "Research Tasks":
            tasks_list = []
            for task_id, task_data in tasks_db.items():
                for task_text in task_data.get('found_tasks', []):
                    tasks_list.append({
                        'Task ID': task_id,
                        'Task Description': task_text,
                        'Category': task_data.get('category', 'unknown'),
                        'Maturity': task_data.get('maturity', 'unknown'),
                        'Score': task_data.get('score', 0),
                        'Impact': task_data.get('impact', 0)
                    })
            
            tasks_df = pd.DataFrame(tasks_list)
            st.dataframe(tasks_df, use_container_width=True)
            
        elif data_option == "Articles":
            articles_df = pd.DataFrame(articles)
            st.dataframe(articles_df, use_container_width=True)
            
        elif data_option == "Graph Statistics":
            st.write("### Graph Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Node Types:**")
                node_types = Counter(graph.nodes[n].get('type', 'unknown') for n in graph.nodes())
                for node_type, count in node_types.items():
                    st.write(f"- {node_type}: {count}")
            
            with col2:
                st.write("**Graph Properties:**")
                st.write(f"- Nodes: {graph.number_of_nodes()}")
                st.write(f"- Edges: {graph.number_of_edges()}")
                st.write(f"- Density: {nx.density(graph):.4f}")
                if nx.is_connected(graph):
                    st.write(f"- Diameter: {nx.diameter(graph)}")
                else:
                    st.write("- Graph is not connected")

if __name__ == "__main__":
    main() 
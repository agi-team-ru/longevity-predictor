import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from collections import Counter
import numpy as np
from plotly.subplots import make_subplots

# Базовые настройки страницы
st.set_page_config(
    page_title="Longevity Research Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# URL backend API (замените на актуальный при необходимости)
API_URL = "http://localhost:8000"

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

# Заготовка функций для загрузки данных через API (заполню позже)
def fetch_graph():
    try:
        response = requests.get(f"{API_URL}/graph", timeout=30)
        response.raise_for_status()
        # Ожидается, что backend вернет сериализованный граф (например, json или pickle)
        # Здесь пример для json-структуры (нужно будет адаптировать под реальный формат)
        return response.json()
    except Exception as e:
        st.warning(f"Не удалось загрузить граф: {e}")
        return None

def fetch_tasks_db():
    try:
        response = requests.get(f"{API_URL}/tasks_db", timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.warning(f"Не удалось загрузить tasks_db: {e}")
        return None

def fetch_articles():
    try:
        response = requests.get(f"{API_URL}/articles", timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.warning(f"Не удалось загрузить статьи: {e}")
        return None

# Кэшируем загрузку данных через API
def load_all_data():
    graph = fetch_graph()
    tasks_db = fetch_tasks_db()
    articles = fetch_articles()
    return graph, tasks_db, articles

@st.cache_data(show_spinner=False)
def get_data():
    return load_all_data()

def create_network_visualization(graph_data, tasks_db, task_filter=None, category_filter=None):
    """Создает интерактивную визуализацию сети на основе данных из API"""
    if not graph_data or not tasks_db:
        return None

    # graph_data должен содержать nodes и edges
    nodes = graph_data.get('nodes', [])
    edges = graph_data.get('edges', [])
    node_dict = {n['id']: n for n in nodes}

    # Фильтрация узлов
    nodes_to_include = set()
    task_nodes = [n['id'] for n in nodes if n.get('type') == 'task']
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
            for e in edges:
                if e['source'] == task:
                    nodes_to_include.add(e['target'])
                if e['target'] == task:
                    nodes_to_include.add(e['source'])

    if not nodes_to_include:
        return None

    # Строим subgraph
    sub_nodes = [n for n in nodes if n['id'] in nodes_to_include]
    sub_edges = [e for e in edges if e['source'] in nodes_to_include and e['target'] in nodes_to_include]

    # Генерируем layout (spring layout)
    G = nx.Graph()
    for n in sub_nodes:
        G.add_node(n['id'], **n)
    for e in sub_edges:
        G.add_edge(e['source'], e['target'])
    try:
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
    except:
        pos = {node: (0, 0) for node in G.nodes()}

    # Подготовка данных для Plotly
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_traces = {}
    for node in G.nodes():
        node_type = G.nodes[node].get('type', 'unknown')
        x, y = pos[node]
        if node_type not in node_traces:
            node_traces[node_type] = {'x': [], 'y': [], 'text': [], 'size': [], 'color': []}
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
            found_tasks = task_data.get('found_tasks', [])
            task_text = found_tasks[0] if found_tasks else node
            tooltip = f"Task: {task_text}<br>Category: {task_data.get('category', 'unknown')}<br>Score: {task_data.get('score', 0)}<br>Maturity: {task_data.get('maturity', 'unknown')}"
        elif node_type == 'article':
            size = 8
            color = '#3498DB'
            tooltip = f"Article: {str(node)[:100]}..."
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
        node_traces[node_type]['x'].append(x)
        node_traces[node_type]['y'].append(y)
        node_traces[node_type]['text'].append(tooltip)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines',
        name='Connections'
    ))
    type_names = {'task': 'Research Tasks', 'article': 'Articles', 'tag': 'Biomedical Tags'}
    for node_type, trace_data in node_traces.items():
        if trace_data['x']:
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
                if len(pub_date) >= 4:
                    year = int(pub_date[:4])
                    if 2000 <= year <= 2025:
                        dates.append(year)
                        journals.append(article.get('journal', 'Unknown'))
        except:
            continue
    if not dates:
        return None, None
    import pandas as pd
    df = pd.DataFrame({'year': dates, 'journal': journals})
    yearly_counts = df.groupby('year').size().reset_index(name='count')
    import plotly.graph_objects as go
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
    # Топ журналы
    from collections import Counter
    journal_counts = Counter(journals)
    top_journals = journal_counts.most_common(10)
    return fig, top_journals

def create_research_analysis(tasks_db):
    """Создает анализ исследовательских задач (scatter plot)"""
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
    import pandas as pd
    df = pd.DataFrame(data)
    import plotly.express as px
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
    return fig, df

# Основная функция приложения
def main():
    st.markdown('<div class="main-header">🧬 Longevity Research Dashboard</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Загрузка данных через API с кэшированием
    with st.spinner("Загрузка данных из backend API..."):
        graph, tasks_db, articles = get_data()

    # Проверка наличия данных
    if not (graph and tasks_db and articles):
        st.error("Не удалось загрузить все необходимые данные. Проверьте доступность backend API.")
        return

    # Sidebar с фильтрами и метриками
    st.sidebar.title("🔍 Filters & Settings")
    st.sidebar.markdown("### 📊 Overview")
    st.sidebar.metric("Research Tasks", len(tasks_db) if tasks_db else 0)
    st.sidebar.metric("Articles", len(articles) if articles else 0)
    st.sidebar.metric("Total Nodes", graph.get('number_of_nodes', 0) if isinstance(graph, dict) else 0)
    st.sidebar.metric("Connections", graph.get('number_of_edges', 0) if isinstance(graph, dict) else 0)

    # Категории для фильтра
    categories = list(set(task.get('category', 'unknown') for task in tasks_db.values())) if tasks_db else []
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
        # Метрики сверху
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
            from collections import Counter
            top_category = Counter(task_data.get('category', 'unknown') for task_data in tasks_db.values()).most_common(1)
            if top_category:
                st.metric("Top Category", top_category[0][0])
        # Дашборд метрик
        metrics_fig = create_metrics_dashboard(tasks_db, articles)
        st.plotly_chart(metrics_fig, use_container_width=True)

    with tab3:
        st.subheader("Publication Timeline Analysis")
        timeline_fig, top_journals = create_timeline_analysis(articles)
        if timeline_fig:
            st.plotly_chart(timeline_fig, use_container_width=True)
        else:
            st.warning("Недостаточно данных о датах публикаций для анализа.")
        st.subheader("Top Journals")
        if top_journals:
            import pandas as pd
            journal_df = pd.DataFrame(top_journals, columns=['Journal', 'Publications'])
            st.dataframe(journal_df, use_container_width=True)
        else:
            st.info("Нет данных по журналам.")

    with tab4:
        st.subheader("Research Task Analysis")
        research_fig, research_df = create_research_analysis(tasks_db)
        st.plotly_chart(research_fig, use_container_width=True)
        st.subheader("Top Scoring Research Tasks")
        if not research_df.empty:
            top_tasks_df = research_df.sort_values('score', ascending=False).head(10)[['task_text', 'category', 'score', 'maturity']]
            top_tasks_df = top_tasks_df.rename(columns={
                'task_text': 'Task',
                'category': 'Category',
                'score': 'Score',
                'maturity': 'Maturity'
            })
            st.dataframe(top_tasks_df, use_container_width=True)
        else:
            st.info("Нет данных для отображения топ задач.")

    with tab5:
        st.subheader("Data Explorer")
        data_option = st.selectbox("Choose data to explore:", [
            "Research Tasks", "Articles", "Graph Statistics"
        ])
        import pandas as pd
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
            nodes = graph.get('nodes', [])
            from collections import Counter
            node_types = Counter(n.get('type', 'unknown') for n in nodes)
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Node Types:**")
                for node_type, count in node_types.items():
                    st.write(f"- {node_type}: {count}")
            with col2:
                st.write("**Graph Properties:**")
                st.write(f"- Nodes: {len(nodes)}")
                st.write(f"- Edges: {len(graph.get('edges', []))}")
                import networkx as nx
                G = nx.Graph()
                for n in nodes:
                    G.add_node(n['id'])
                for e in graph.get('edges', []):
                    G.add_edge(e['source'], e['target'])
                st.write(f"- Density: {nx.density(G):.4f}")
                if nx.is_connected(G):
                    st.write(f"- Diameter: {nx.diameter(G)}")
                else:
                    st.write("- Graph is not connected")

if __name__ == "__main__":
    main()

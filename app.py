import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np  # Добавлен импорт numpy

# Настройка страницы
st.set_page_config(
    page_title="Cat Analytics Dashboard",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Стили CSS
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
    }
    .stMetricLabel {
        font-size: 1.1rem !important;
        color: #666 !important;
    }
    .stMetricValue {
        font-size: 1.4rem !important;
        color: #2c2c2c !important;
    }
    .plotly-chart {
        border-radius: 15px;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Заголовок
st.title("🐱 Advanced Cat Analytics Dashboard")
st.markdown("""
**Анализ характеристик кошек трёх пород:** Ангора, Рэгдолл, Мейн-кун  
*Источник данных: [It's Raining Cats Dataset](https://www.kaggle.com/datasets/joannanplkrk/its-raining-cats)*
""")

# Загрузка данных
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/cat_breeds_clean.csv", sep=";")
        
        # Преобразование булевых значений
        df["Neutered_or_spayed"] = df["Neutered_or_spayed"].astype(str).str.upper().map({
            "TRUE": True,
            "FALSE": False,
            "NAN": None  # Обработка пропущенных значений
        })
        
        df["Allowed_outdoor"] = df["Allowed_outdoor"].astype(str).str.upper().map({
            "TRUE": True,
            "FALSE": False,
            "NAN": None
        })
        
        # Проверка наличия необходимых столбцов
        required_columns = ["Breed", "Age_in_years", "Weight", "Owner_play_time_minutes", 
                           "Sleep_time_hours", "Body_length", "Gender", "Country"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Отсутствуют столбцы в данных: {missing_columns}")
            st.stop()
        
        return df
    except FileNotFoundError:
        st.error("Файл data/cat_breeds_clean.csv не найден")
        st.stop()

df = load_data()

# Фильтры в боковой панели
with st.sidebar:
    with st.expander("⚙️ Фильтры данных", expanded=True):
        breed_filter = st.selectbox("Порода", ["Все"] + df["Breed"].unique().tolist())
        gender_filter = st.selectbox("Пол", ["Все"] + df["Gender"].unique().tolist())
        age_filter = st.slider("Возраст (годы)", 
                             min_value=0, 
                             max_value=int(df["Age_in_years"].max()), 
                             value=(0, int(df["Age_in_years"].max())))
        country_filter = st.selectbox("Страна", ["Все"] + df["Country"].unique().tolist())

# Фильтрация данных
def filter_data(df):
    filtered_df = df.copy()
    if breed_filter != "Все":
        filtered_df = filtered_df[filtered_df["Breed"] == breed_filter]
    if gender_filter != "Все":
        filtered_df = filtered_df[filtered_df["Gender"] == gender_filter]
    filtered_df = filtered_df[(filtered_df["Age_in_years"] >= age_filter[0]) & 
                            (filtered_df["Age_in_years"] <= age_filter[1])]
    if country_filter != "Все":
        filtered_df = filtered_df[filtered_df["Country"] == country_filter]
    return filtered_df

filtered_df = filter_data(df)

# Ключевые метрики
st.subheader("📊 Основные показатели")
cols = st.columns(4)
metrics = {
    "Всего кошек": len(filtered_df),
    "Средний возраст": f"{filtered_df['Age_in_years'].mean():.1f} лет",
    "Средний вес": f"{filtered_df['Weight'].mean():.1f} кг",
    "Активность (ч/день)": f"{filtered_df['Owner_play_time_minutes'].mean() / 60:.1f} часов"  # Перевод минут в часы
}

for col, (label, value) in zip(cols, metrics.items()):
    with col:
        st.markdown(f"<div style='padding: 20px; background-color: #f8f9fa; border-radius: 10px;'>"
                    f"<h3 style='margin:0; color: #2c2c2c;'>{value}</h3>"
                    f"<p style='margin:0; color: #666;'>{label}</p></div>", 
                    unsafe_allow_html=True)

# Визуализации во вкладках
tab1, tab2, tab3 = st.tabs(["📈 Распределение", "📊 Сравнение", "🔍 Корреляции"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        # Распределение по породам
        fig = px.pie(filtered_df, names="Breed", title="Распределение по породам",
                    hole=0.4, color="Breed", 
                    color_discrete_map={
                        'Angora': '#FFA07A',
                        'Ragdoll': '#87CEEB',
                        'Maine Coon': '#778899'
                    })
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Распределение возраста
        fig = px.histogram(filtered_df, x="Age_in_years", nbins=20, 
                         title="Распределение возраста",
                         color="Breed", barmode="overlay",
                         opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        # Boxplot параметров
        fig = px.box(filtered_df, x="Breed", y="Weight", 
                    title="Распределение веса по породам",
                    color="Breed", points="all")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Scatter plot
        numerical_cols = ["Age_in_years", "Weight", "Body_length", 
                        "Sleep_time_hours", "Owner_play_time_minutes"]
        x_axis = st.selectbox("Ось X", numerical_cols, key="x_axis")
        y_axis = st.selectbox("Ось Y", numerical_cols, index=1, key="y_axis")
        
        fig = px.scatter(
            filtered_df, x=x_axis, y=y_axis, 
            color="Breed", size="Weight",
            hover_data=["Gender", "Country"],
            title=f"{x_axis} vs {y_axis}",
            trendline="lowess"
        )
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Матрица корреляций
    corr_matrix = filtered_df[numerical_cols].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='Blues',
        hoverongaps=False,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}"
    ))
    fig.update_layout(title="Матрица корреляций", height=600)
    st.plotly_chart(fig, use_container_width=True)

# Дополнительная информация в боковой панели
with st.sidebar:
    st.markdown("---")
    with st.expander("ℹ️ О проекте"):
        st.markdown("""
        **Автор:** [ast]  
        **Версия:** 1.0  
        **Обновлено:** 2023-12-20  
                    
        Этот дашборд позволяет анализировать данные о кошках трёх пород:
        - Ангора
        - Рэгдолл
        - Мейн-кун
                    
        Используйте фильтры для уточнения данных и переключайтесь между вкладками для просмотра различных визуализаций.
        """)
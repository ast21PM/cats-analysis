import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

st.set_page_config(
    page_title="Аналитика кошек",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

st.title("🐱 Аналитика кошек")
st.markdown("""
**Анализ характеристик трех пород кошек:** Ангора, Рэгдолл, Мейн-кун  
*Источник данных: [It's Raining Cats Dataset](https://www.kaggle.com/datasets/joannanplkrk/its-raining-cats)*
""")

@st.cache_data
def загрузить_данные():
    try:
        df = pd.read_csv("data/cat_breeds_clean.csv", sep=";")
        
        df["Кастрирован_или_стерилизован"] = df["Neutered_or_spayed"].astype(str).str.upper().map({
            "TRUE": True,
            "FALSE": False,
            "NAN": None
        })
        
        df["Разрешено_на_улицу"] = df["Allowed_outdoor"].astype(str).str.upper().map({
            "TRUE": True,
            "FALSE": False,
            "NAN": None
        })
        
        df["Пол"] = df["Gender"].map({'male': 0, 'female': 1})
        
        необходимые_столбцы = ["Порода", "Возраст_в_годах", "Вес", "Время_игры_с_хозяином_минуты", 
                              "Время_сна_часы", "Длина_тела", "Пол", "Страна"]
        отсутствующие_столбцы = [col for col in необходимые_столбцы if col not in df.columns]
        if отсутствующие_столбцы:
            st.error(f"Отсутствуют столбцы в данных: {отсутствующие_столбцы}")
            st.stop()
        
        df = df.rename(columns={
            "Breed": "Порода",
            "Age_in_years": "Возраст_в_годах",
            "Weight": "Вес",
            "Owner_play_time_minutes": "Время_игры_с_хозяином_минуты",
            "Sleep_time_hours": "Время_сна_часы",
            "Body_length": "Длина_тела",
            "Gender": "Пол",
            "Country": "Страна",
            "Neutered_or_spayed": "Кастрирован_или_стерилизован",
            "Allowed_outdoor": "Разрешено_на_улицу",
            "Fur_colour_dominant": "Цвет_шерсти_основной",
            "Fur_pattern": "Узор_шерсти",
            "Eye_colour": "Цвет_глаз",
            "Preferred_food": "Предпочитаемая_еда",
            "Age_in_months": "Возраст_в_месяцах"
        })
        
        return df
    except FileNotFoundError:
        st.error("Файл data/cat_breeds_clean.csv не найден")
        st.stop()

df = загрузить_данные()

with st.sidebar:
    with st.expander("⚙️ Фильтры данных", expanded=True):
        фильтр_породы = st.selectbox("Порода", ["Все"] + df["Порода"].unique().tolist())
        фильтр_пола = st.selectbox("Пол", ["Все"] + df["Пол"].unique().tolist())
        фильтр_возраста = st.slider("Возраст (годы)", 
                                   min_value=0, 
                                   max_value=int(df["Возраст_в_годах"].max()), 
                                   value=(0, int(df["Возраст_в_годах"].max())))
        фильтр_страны = st.selectbox("Страна", ["Все"] + df["Страна"].unique().tolist())

def отфильтровать_данные(df):
    отфильтрованные_данные = df.copy()
    if фильтр_породы != "Все":
        отфильтрованные_данные = отфильтрованные_данные[отфильтрованные_данные["Порода"] == фильтр_породы]
    if фильтр_пола != "Все":
        отфильтрованные_данные = отфильтрованные_данные[отфильтрованные_данные["Пол"] == фильтр_пола]
    отфильтрованные_данные = отфильтрованные_данные[(отфильтрованные_данные["Возраст_в_годах"] >= фильтр_возраста[0]) & 
                                                    (отфильтрованные_данные["Возраст_в_годах"] <= фильтр_возраста[1])]
    if фильтр_страны != "Все":
        отфильтрованные_данные = отфильтрованные_данные[отфильтрованные_данные["Страна"] == фильтр_страны]
    return отфильтрованные_данные

отфильтрованные_данные = отфильтровать_данные(df)

@st.cache_resource
def обучить_модель(df):
    df_ml = df.copy()
    df_ml['Кастрирован_или_стерилизован'] = df_ml['Кастрирован_или_стерилизован'].astype(int)
    df_ml['Разрешено_на_улицу'] = df_ml['Разрешено_на_улицу'].astype(int)

    X = df_ml.drop(['Порода', 'Возраст_в_месяцах', 'Страна'], axis=1)
    y = df_ml['Порода']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    категориальные_столбцы = ['Цвет_шерсти_основной', 'Узор_шерсти', 'Цвет_глаз', 'Предпочитаемая_еда']

    препроцессор = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), категориальные_столбцы)
        ],
        remainder='passthrough'
    )

    конвейер = Pipeline(steps=[
        ('препроцессор', препроцессор),
        ('классификатор', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])

    конвейер.fit(X_train, y_train)

    y_pred = конвейер.predict(X_test)
    отчет_классификации = classification_report(y_test, y_pred, output_dict=True)
    матрица_ошибок = confusion_matrix(y_test, y_pred)
    точность = accuracy_score(y_test, y_pred)
    
    return конвейер, отчет_классификации, матрица_ошибок, точность

конвейер, отчет_классификации, матрица_ошибок, точность = обучить_модель(df)

st.subheader("📊 Ключевые метрики")
cols = st.columns(4)
метрики = {
    "Всего кошек": len(отфильтрованные_данные),
    "Средний возраст": f"{отфильтрованные_данные['Возраст_в_годах'].mean():.1f} лет",
    "Средний вес": f"{отфильтрованные_данные['Вес'].mean():.1f} кг",
    "Активность (часы/день)": f"{отфильтрованные_данные['Время_игры_с_хозяином_минуты'].mean() / 60:.1f} часов"
}

for col, (метка, значение) in zip(cols, метрики.items()):
    with col:
        st.markdown(f"<div style='padding: 20px; background-color: #f8f9fa; border-radius: 10px;'>"
                    f"<h3 style='margin:0; color: #2c2c2c;'>{значение}</h3>"
                    f"<p style='margin:0; color: #666;'>{метка}</p></div>", 
                    unsafe_allow_html=True)

вкладка1, вкладка2, вкладка3 = st.tabs(["📈 Распределения", "📊 Сравнения", "🤖 Машинное обучение"])

with вкладка1:
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(отфильтрованные_данные, names="Порода", title="Распределение по породам",
                    hole=0.4, color="Порода", 
                    color_discrete_map={
                        'Angora': '#FFA07A',
                        'Ragdoll': '#87CEEB',
                        'Maine Coon': '#778899'
                    })
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(отфильтрованные_данные, x="Возраст_в_годах", nbins=20, 
                         title="Распределение по возрасту",
                         color="Порода", barmode="overlay",
                         opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)

with вкладка2:
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.box(отфильтрованные_данные, x="Порода", y="Вес", 
                    title="Распределение веса по породам",
                    color="Порода", points="all")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        числовые_столбцы = ["Возраст_в_годах", "Вес", "Длина_тела", 
                           "Время_сна_часы", "Время_игры_с_хозяином_минуты"]
        ось_x = st.selectbox("Ось X", числовые_столбцы, key="ось_x")
        ось_y = st.selectbox("Ось Y", числовые_столбцы, index=1, key="ось_y")
        
        fig = px.scatter(
            отфильтрованные_данные, x=ось_x, y=ось_y, 
            color="Порода", size="Вес",
            hover_data=["Пол", "Страна"],
            title=f"{ось_x} против {ось_y}",
            trendline="lowess"
        )
        st.plotly_chart(fig, use_container_width=True)

with вкладка3:
    st.subheader("Машинное обучение: Предсказание породы")
    
    st.write(f"**Точность модели:** {точность:.2f}")
    
    st.write("**Отчет по классификации:**")
    st.write(pd.DataFrame(отчет_классификации).transpose())
    
    st.subheader("Предскажите породу вашей кошки")
    with st.form("форма_предсказания"):
        возраст = st.slider("Возраст (годы)", 0.0, float(df["Возраст_в_годах"].max()), 2.0)
        вес = st.slider("Вес (кг)", 0.0, float(df["Вес"].max()), 5.0)
        длина_тела = st.slider("Длина тела (см)", 0.0, float(df["Длина_тела"].max()), 40.0)
        время_сна = st.slider("Время сна (часы)", 0, int(df["Время_сна_часы"].max()), 16)
        время_игры = st.slider("Время игры с хозяином (минуты)", 0, int(df["Время_игры_с_хозяином_минуты"].max()), 20)
        пол = st.selectbox("Пол", ["Мужской", "Женский"])
        кастрирован = st.selectbox("Кастрирован/Стерилизован", [True, False])
        разрешено_на_улицу = st.selectbox("Разрешено на улицу", [True, False])
        цвет_шерсти = st.selectbox("Цвет шерсти", df["Цвет_шерсти_основной"].unique())
        узор_шерсти = st.selectbox("Узор шерсти", df["Узор_шерсти"].unique())
        цвет_глаз = st.selectbox("Цвет глаз", df["Цвет_глаз"].unique())
        предпочитаемая_еда = st.selectbox("Предпочитаемая еда", df["Предпочитаемая_еда"].unique())
        
        кнопка_предсказать = st.form_submit_button("Предсказать")
        
        if кнопка_предсказать:
            входные_данные = pd.DataFrame({
                'Возраст_в_годах': [возраст],
                'Вес': [вес],
                'Длина_тела': [длина_тела],
                'Время_сна_часы': [время_сна],
                'Время_игры_с_хозяином_минуты': [время_игры],
                'Пол': [1 if пол == 'Женский' else 0],
                'Кастрирован_или_стерилизован': [int(кастрирован)],
                'Разрешено_на_улицу': [int(разрешено_на_улицу)],
                'Цвет_шерсти_основной': [цвет_шерсти],
                'Узор_шерсти': [узор_шерсти],
                'Цвет_глаз': [цвет_глаз],
                'Предпочитаемая_еда': [предпочитаемая_еда]
            })
            
            # Получаем все столбцы из обучающих данных
            обучающие_столбцы = конвейер.named_steps['препроцессор'].transformers_[0][2] + \
                               [col for col in X.columns if col not in категориальные_столбцы]
            # Добавляем недостающие столбцы с нулевыми значениями
            for col in обучающие_столбцы:
                if col not in входные_данные.columns:
                    входные_данные[col] = 0

            предсказание = конвейер.predict(входные_данные)[0]
            st.success(f"Предсказанная порода: **{предсказание}**")

with st.sidebar:
    st.markdown("---")
    with st.expander("ℹ️ О проекте"):
        st.markdown("""
        **Автор:** [ast]  
        **Версия:** 1.0  
        **Последнее обновление:** 2023-12-20  
                    
        Этот дашборд анализирует данные трех пород кошек:
        - Ангора
        - Рэгдолл
        - Мейн-кун
                    
        Используйте фильтры для уточнения данных и переключайтесь между вкладками для просмотра различных визуализаций.
        """)
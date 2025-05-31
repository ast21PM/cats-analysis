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

# Настройка страницы Streamlit
st.set_page_config(
    page_title="Аналитика кошек",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Пользовательский CSS для стилизации
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

# Заголовок и описание страницы
st.title("🐱 Аналитика кошек")
st.markdown("""
**Анализ характеристик трех пород кошек:** Ангора, Рэгдолл, Мейн-кун  
*Источник данных: [It's Raining Cats Dataset](https://www.kaggle.com/datasets/joannanplkrk/its-raining-cats)*
""")

# Функция загрузки и предобработки данных
@st.cache_data
def load_data():
    try:
        # Загрузка датасета
        df = pd.read_csv("data/cat_breeds_clean_russian.csv", sep=";")
        
        # Предобработка булевых и категориальных столбцов
        df["Кастрирован_или_Стерилизован"] = df["Кастрирован_или_Стерилизован"].astype(str).str.upper().map({
            "TRUE": True,
            "FALSE": False,
            "NAN": None
        })
        
        df["Разрешено_На_Улицу"] = df["Разрешено_На_Улицу"].astype(str).str.upper().map({
            "TRUE": True,
            "FALSE": False,
            "NAN": None
        })
        
        df["Пол"] = df["Пол"].map({'male': 0, 'female': 1})
        
        # Проверка наличия необходимых столбцов
        required_columns = ["Порода", "Возраст_Годы", "Вес_кг", "Время_Игры_с_Хозяином_Минуты", 
                           "Время_Сна_Часы", "Длина_Тела_см", "Пол", "Страна"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Отсутствуют столбцы в данных: {missing_columns}")
            st.stop()
        
        return df
    except FileNotFoundError:
        st.error("Файл data/cat_breeds_clean_russian.csv не найден")
        st.stop()

# Загрузка данных
df = load_data()

# Фильтры в боковой панели
with st.sidebar:
    with st.expander("⚙️ Фильтры данных", expanded=True):
        breed_filter = st.selectbox("Порода", ["Все"] + df["Порода"].unique().tolist())
        gender_filter = st.selectbox("Пол", ["Все"] + df["Пол"].unique().tolist())
        age_filter = st.slider("Возраст (годы)", 
                             min_value=0, 
                             max_value=int(df["Возраст_Годы"].max()), 
                             value=(0, int(df["Возраст_Годы"].max())))
        country_filter = st.selectbox("Страна", ["Все"] + df["Страна"].unique().tolist())

# Фильтрация данных на основе пользовательского ввода
def filter_data(df):
    filtered_df = df.copy()
    if breed_filter != "Все":
        filtered_df = filtered_df[filtered_df["Порода"] == breed_filter]
    if gender_filter != "Все":
        filtered_df = filtered_df[filtered_df["Пол"] == gender_filter]
    filtered_df = filtered_df[(filtered_df["Возраст_Годы"] >= age_filter[0]) & 
                            (filtered_df["Возраст_Годы"] <= age_filter[1])]
    if country_filter != "Все":
        filtered_df = filtered_df[filtered_df["Страна"] == country_filter]
    return filtered_df

filtered_df = filter_data(df)

# Обучение модели машинного обучения
@st.cache_resource
def train_model(df):
    df_ml = df.copy()
    df_ml['Кастрирован_или_Стерилизован'] = df_ml['Кастрирован_или_Стерилизован'].astype(int)
    df_ml['Разрешено_На_Улицу'] = df_ml['Разрешено_На_Улицу'].astype(int)

    X = df_ml.drop(['Порода', 'Возраст_Месяцы', 'Страна'], axis=1)
    y = df_ml['Порода']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    categorical_cols = ['Основной_Цвет_Шерсти', 'Узор_Шерсти', 'Цвет_Глаз', 'Предпочитаемая_Еда']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_cols)
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    return pipeline, class_report, conf_matrix, accuracy

pipeline, class_report, conf_matrix, accuracy = train_model(df)

# Отображение ключевых метрик
st.subheader("📊 Ключевые метрики")
cols = st.columns(4)
metrics = {
    "Всего кошек": len(filtered_df),
    "Средний возраст": f"{filtered_df['Возраст_Годы'].mean():.1f} лет",
    "Средний вес": f"{filtered_df['Вес_кг'].mean():.1f} кг",
    "Активность (часы/день)": f"{filtered_df['Время_Игры_с_Хозяином_Минуты'].mean() / 60:.1f} часов"
}

for col, (label, value) in zip(cols, metrics.items()):
    with col:
        st.markdown(f"<div style='padding: 20px; background-color: #f8f9fa; border-radius: 10px;'>"
                    f"<h3 style='margin:0; color: #2c2c2c;'>{value}</h3>"
                    f"<p style='margin:0; color: #666;'>{label}</p></div>", 
                    unsafe_allow_html=True)

# Вкладки для различных визуализаций
tab1, tab2, tab3, tab4 = st.tabs(["📈 Распределения", "📊 Сравнения", "🔍 Корреляции", "🤖 Машинное обучение"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(filtered_df, names="Порода", title="Распределение по породам",
                    hole=0.4, color="Порода", 
                    color_discrete_map={
                        'Angora': '#FFA07A',
                        'Ragdoll': '#87CEEB',
                        'Maine Coon': '#778899'
                    })
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(filtered_df, x="Возраст_Годы", nbins=20, 
                         title="Распределение по возрасту",
                         color="Порода", barmode="overlay",
                         opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.box(filtered_df, x="Порода", y="Вес_кг", 
                    title="Распределение веса по породам",
                    color="Порода", points="all")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        numerical_cols = ["Возраст_Годы", "Вес_кг", "Длина_Тела_см", 
                        "Время_Сна_Часы", "Время_Игры_с_Хозяином_Минуты"]
        x_axis = st.selectbox("Ось X", numerical_cols, key="x_axis")
        y_axis = st.selectbox("Ось Y", numerical_cols, index=1, key="y_axis")
        
        fig = px.scatter(
            filtered_df, x=x_axis, y=y_axis, 
            color="Порода", size="Вес_кг",
            hover_data=["Пол", "Страна"],
            title=f"{x_axis} vs {y_axis}",
            trendline="lowess"
        )
        st.plotly_chart(fig, use_container_width=True)

with tab3:
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

with tab4:
    st.subheader("Машинное обучение: Предсказание породы")
    
    st.write(f"**Точность модели:** {accuracy:.2f}")
    
    st.write("**Матрица ошибок:**")
    st.write(conf_matrix)
    
    st.write("**Отчет по классификации:**")
    st.write(pd.DataFrame(class_report).transpose())
    
    st.subheader("Предскажите породу вашей кошки")
    with st.form("prediction_form"):
        age = st.slider("Возраст (годы)", 0.0, float(df["Возраст_Годы"].max()), 2.0)
        weight = st.slider("Вес (кг)", 0.0, float(df["Вес_кг"].max()), 5.0)
        body_length = st.slider("Длина тела (см)", 0.0, float(df["Длина_Тела_см"].max()), 40.0)
        sleep_time = st.slider("Время сна (часы)", 0, int(df["Время_Сна_Часы"].max()), 16)
        play_time = st.slider("Время игры с хозяином (минуты)", 0, int(df["Время_Игры_с_Хозяином_Минуты"].max()), 20)
        gender = st.selectbox("Пол", ["Мужской", "Женский"])
        neutered = st.selectbox("Кастрирован/Стерилизован", [True, False])
        outdoor = st.selectbox("Разрешено на улицу", [True, False])
        fur_colour = st.selectbox("Цвет шерсти", df["Основной_Цвет_Шерсти"].unique())
        fur_pattern = st.selectbox("Узор шерсти", df["Узор_Шерсти"].unique())
        eye_colour = st.selectbox("Цвет глаз", df["Цвет_Глаз"].unique())
        preferred_food = st.selectbox("Предпочитаемая еда", df["Предпочитаемая_Еда"].unique())
        
        submit_button = st.form_submit_button("Предсказать")
        
        if submit_button:
            input_data = pd.DataFrame({
                'Возраст_Годы': [age],
                'Вес_кг': [weight],
                'Длина_Тела_см': [body_length],
                'Время_Сна_Часы': [sleep_time],
                'Время_Игры_с_Хозяином_Минуты': [play_time],
                'Пол': [1 if gender == 'Женский' else 0],
                'Кастрирован_или_Стерилизован': [int(neutered)],
                'Разрешено_На_Улицу': [int(outdoor)],
                'Основной_Цвет_Шерсти': [fur_colour],
                'Узор_Шерсти': [fur_pattern],
                'Цвет_Глаз': [eye_colour],
                'Предпочитаемая_Еда': [preferred_food]
            })
            
            prediction = pipeline.predict(input_data)[0]
            st.success(f"Предсказанная порода: **{prediction}**")

# Информация о проекте в боковой панели
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
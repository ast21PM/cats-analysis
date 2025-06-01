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
import os 
from pathlib import Path

# Первым делом настраиваем конфигурацию страницы
st.set_page_config(
    page_title="Аналитика кошек",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Определяем пути относительно текущего файла
SCRIPT_DIR = Path(__file__).parent.absolute()
DATA_DIR = SCRIPT_DIR / "data"
ASSETS_DIR = SCRIPT_DIR / "assets"

def load_image(image_path):
    """Загружает изображение из файла"""
    try:
        with open(image_path, "rb") as f:
            return f.read()
    except Exception as e:
        st.error(f"Ошибка при загрузке изображения {image_path}: {str(e)}")
        return None

def load_breed_image(img_path):
    """Загружает изображение породы"""
    try:
        with open(img_path, "rb") as f:
            return f.read()
    except Exception as e:
        st.error(f"Ошибка при загрузке изображения породы {img_path}: {str(e)}")
        return None

# Загружаем изображения
three_image = load_image(ASSETS_DIR / "three.png")
threes_image = load_image(ASSETS_DIR / "threes.png")

# Отладочная информация
st.write("Проверка путей:")
st.write("1. BASE_PATH =", DATA_DIR)
st.write("2. three.png существует?", three_image is not None)
st.write("3. threes.png существует?", threes_image is not None)

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

st.title("🐱 Расширенная аналитика кошек")
st.markdown("""
**Анализ характеристик кошек трёх пород:** Ангора, Рэгдолл, Мейн-кун  
*Источник данных: [It's Raining Cats Dataset](https://www.kaggle.com/datasets/joannanplkrk/its-raining-cats)*
""")

FUR_COLOUR_MAP = {
    'white': 'Белый',
    'black': 'Черный',
    'red/cream': 'Рыжий/Кремовый',
    'brown/chocolate': 'Шоколадный',
    'seal': 'Темно-коричневый',
    'lilac': 'Лиловый',
}

FUR_PATTERN_MAP = {
    'solid': 'Однотонный',
    'tabby': 'Полосатый',
    'tortie': 'Черепаховый',
    'bicolor': 'Двухцветный',
    'colorpoint': 'Цветные точки',
    'mitted': 'Серый леопард',
}

EYE_COLOUR_MAP = {
    'blue': 'Голубой',
    'green': 'Зеленый',
    'amber': 'Янтарный',
    'yellow': 'Желтый',
}

PREFERRED_FOOD_MAP = {
    'wet': 'Влажный корм',
    'dry': 'Сухой корм',
}

INV_FUR_COLOUR_MAP = {v: k for k, v in FUR_COLOUR_MAP.items()}
INV_FUR_PATTERN_MAP = {v: k for k, v in FUR_PATTERN_MAP.items()}
INV_EYE_COLOUR_MAP = {v: k for k, v in EYE_COLOUR_MAP.items()}
INV_PREFERRED_FOOD_MAP = {v: k for k, v in PREFERRED_FOOD_MAP.items()}


BREED_IMAGES = {
    'Angora': [DATA_DIR / f"Angora{i}.png" for i in range(1, 4)],
    'Maine coon': [DATA_DIR / f"Coon{i}.png" for i in range(1, 4)],
    'Ragdoll': [DATA_DIR / f"Ragdoll{i}.png" for i in range(1, 4)],
}


@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_DIR / "cat_breeds_clean.csv", sep=";")
        
        df["Neutered_or_spayed"] = df["Neutered_or_spayed"].astype(str).str.upper().map({
            "TRUE": 'Да',
            "FALSE": 'Нет',
            "NAN": None 
        })
        
        df["Allowed_outdoor"] = df["Allowed_outdoor"].astype(str).str.upper().map({
            "TRUE": 'Да',
            "FALSE": 'Нет',
            "NAN": None
        })
        
        df["Gender"] = df["Gender"].map({'male': 'Кот', 'female': 'Кошка'}) 
        

        df["Fur_colour_dominant"] = df["Fur_colour_dominant"].apply(lambda x: FUR_COLOUR_MAP.get(x, x))
        df["Fur_pattern"] = df["Fur_pattern"].apply(lambda x: FUR_PATTERN_MAP.get(x, x))
        df["Eye_colour"] = df["Eye_colour"].apply(lambda x: EYE_COLOUR_MAP.get(x, x))
        df["Preferred_food"] = df["Preferred_food"].apply(lambda x: PREFERRED_FOOD_MAP.get(x, x))

        required_columns = ["Breed", "Age_in_years", "Weight", "Owner_play_time_minutes", 
                            "Sleep_time_hours", "Body_length", "Gender", "Country",
                            "Fur_colour_dominant", "Fur_pattern", "Eye_colour", "Preferred_food"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Отсутствуют столбцы в данных: {missing_columns}")
            st.stop()
        
        return df
    except FileNotFoundError:
        st.error("Файл data/cat_breeds_clean.csv не найден")
        st.stop()

df = load_data()

with st.sidebar:
    with st.expander("⚙️ Фильтры данных", expanded=True):
        breed_filter = st.selectbox("Порода", ["Все"] + df["Breed"].unique().tolist())
        gender_filter = st.selectbox("Пол", ["Все"] + df["Gender"].unique().tolist())
        age_filter = st.slider("Возраст (годы)", 
                               min_value=0, 
                               max_value=int(df["Age_in_years"].max()), 
                               value=(0, int(df["Age_in_years"].max())))
        country_filter = st.selectbox("Страна", ["Все"] + df["Country"].unique().tolist())

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

@st.cache_resource
def train_model(df):
    df_ml = df.copy()
    
    df_ml["Gender"] = df_ml["Gender"].map({'Кот': 0, 'Кошка': 1})
    df_ml['Neutered_or_spayed'] = df_ml['Neutered_or_spayed'].map({'Да': 1, 'Нет': 0})
    df_ml['Allowed_outdoor'] = df_ml['Allowed_outdoor'].map({'Да': 1, 'Нет': 0})


    df_ml["Fur_colour_dominant"] = df_ml["Fur_colour_dominant"].apply(lambda x: INV_FUR_COLOUR_MAP.get(x, x))
    df_ml["Fur_pattern"] = df_ml["Fur_pattern"].apply(lambda x: INV_FUR_PATTERN_MAP.get(x, x))
    df_ml["Eye_colour"] = df_ml["Eye_colour"].apply(lambda x: INV_EYE_COLOUR_MAP.get(x, x))
    df_ml["Preferred_food"] = df_ml["Preferred_food"].apply(lambda x: INV_PREFERRED_FOOD_MAP.get(x, x))

    X = df_ml.drop(['Breed', 'Age_in_months', 'Country'], axis=1)
    y = df_ml['Breed']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    categorical_cols = ['Fur_colour_dominant', 'Fur_pattern', 'Eye_colour', 'Preferred_food']
    
    for col in categorical_cols:
        if df_ml[col].isnull().any():
            df_ml[col] = df_ml[col].fillna('unknown') 

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
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

st.subheader("📊 Основные показатели")
cols = st.columns(4)
metrics = {
    "Всего кошек": len(filtered_df),
    "Средний возраст": f"{filtered_df['Age_in_years'].mean():.1f} лет",
    "Средний вес": f"{filtered_df['Weight'].mean():.1f} кг",
    "Активность (ч/день)": f"{filtered_df['Owner_play_time_minutes'].mean() / 60:.1f} часов"
}

for col, (label, value) in zip(cols, metrics.items()):
    with col:
        st.markdown(f"<div style='padding: 20px; background-color: #f8f9fa; border-radius: 10px;'>"
                            f"<h3 style='margin:0; color: #2c2c2c;'>{value}</h3>"
                            f"<p style='margin:0; color: #666;'>{label}</p></div>", 
                            unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📈 Распределение", "📊 Сравнение", "🤖 Машинное обучение"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(filtered_df, names="Breed", title="Распределение по породам",
                     hole=0.4, color="Breed", 
                     color_discrete_map={
                         'Angora': '#FFA07A',
                         'Ragdoll': '#87CEEB',
                         'Maine coon': '#778899'
                     })
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(filtered_df, x="Age_in_years", nbins=20, 
                           title="Распределение возраста",
                           color="Breed", 
                           barmode="group",
                           opacity=1,
                           labels={
                               "Age_in_years": "Возраст (годы)",
                               "count": "Количество кошек",
                               "Breed": "Порода"
                           })
         
        fig.update_layout(
            bargap=0.2,
            bargroupgap=0.1
        )
        
        fig.update_traces(
            marker_line_width=1,
            marker_line_color="white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if three_image is not None:
            st.image(three_image, caption="Сравнение пород: Ангора, Рэгдолл и Мейн-кун", width=600)

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.box(filtered_df, x="Breed", y="Weight", 
                     title="Распределение веса по породам",
                     color="Breed", points="all")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        numerical_cols = ["Age_in_years", "Weight", "Body_length", 
                          "Sleep_time_hours", "Owner_play_time_minutes"]
        x_axis = st.selectbox("Ось X", numerical_cols, key="x_axis")
        y_axis = st.selectbox("Ось Y", numerical_cols, index=1, key="y_axis")
        
        fig = px.scatter(
            filtered_df, x=x_axis, y=y_axis, 
            color="Breed", size="Weight",
            hover_data=["Gender", "Country"],
            title=f"{x_axis} vs {y_axis}"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if threes_image is not None:
            st.image(threes_image, caption="Сравнительный анализ пород", width=600)

with tab3:
    st.subheader("Машинное обучение: предсказание породы")
    
    st.write(f"**Точность модели:** {accuracy:.2f}")
    
    st.write("**Матрица ошибок:**")
    st.write(conf_matrix)
    
    st.write("**Отчет по классификации:**")
    st.write(pd.DataFrame(class_report).transpose())
    
    st.subheader("Попробуй предсказать породу кошки")
    with st.form("prediction_form"):
        age = st.slider("Возраст (годы)", 0.0, float(df["Age_in_years"].max()), 2.0)
        weight = st.slider("Вес (кг)", 0.0, float(df["Weight"].max()), 5.0)
        body_length = st.slider("Длина тела (см)", 0.0, float(df["Body_length"].max()), 40.0)
        sleep_time = st.slider("Время сна (часы)", 0, int(df["Sleep_time_hours"].max()), 16)
        play_time = st.slider("Время игры с хозяином (минуты)", 0, int(df["Owner_play_time_minutes"].max()), 20)
        gender = st.selectbox("Пол", ["Кот", "Кошка"])
        neutered = st.selectbox("Стерилизован/кастрирован", ["Да", "Нет"])
        outdoor = st.selectbox("Разрешено выходить на улицу", ["Да", "Нет"])
        
        fur_colour = st.selectbox("Цвет шерсти", df["Fur_colour_dominant"].unique())
        fur_pattern = st.selectbox("Узор шерсти", df["Fur_pattern"].unique())
        eye_colour = st.selectbox("Цвет глаз", df["Eye_colour"].unique())
        preferred_food = st.selectbox("Предпочитаемая еда", df["Preferred_food"].unique())
        
        submit_button = st.form_submit_button("Предсказать")
        
        if submit_button:
            input_data = pd.DataFrame({
                'Age_in_years': [age],
                'Weight': [weight],
                'Body_length': [body_length],
                'Sleep_time_hours': [sleep_time],
                'Owner_play_time_minutes': [play_time],
                'Gender': [1 if gender == 'Кошка' else 0],
                'Neutered_or_spayed': [1 if neutered == 'Да' else 0],
                'Allowed_outdoor': [1 if outdoor == 'Да' else 0],
                'Fur_colour_dominant': [INV_FUR_COLOUR_MAP.get(fur_colour)],
                'Fur_pattern': [INV_FUR_PATTERN_MAP.get(fur_pattern)],
                'Eye_colour': [INV_EYE_COLOUR_MAP.get(eye_colour)],
                'Preferred_food': [INV_PREFERRED_FOOD_MAP.get(preferred_food)]
            })
            
            prediction = pipeline.predict(input_data)[0]
            st.success(f"Предсказанная порода: **{prediction}**")

            if prediction in BREED_IMAGES:
                st.write(f"Примеры кошек породы {prediction}:")
                cols = st.columns(len(BREED_IMAGES[prediction]))
                for i, img_path in enumerate(BREED_IMAGES[prediction]):
                    if img_path.exists():
                        image_data = load_breed_image(img_path)
                        if image_data is not None:
                            with cols[i]:
                                st.image(image_data, caption=f"{prediction} {i+1}", width=300)
                    else:
                        st.warning(f"Изображение не найдено: {img_path}")
            else:
                st.info("Для данной породы нет доступных изображений.")

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
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# --- ИЗМЕНЕНИЕ: ИМПОРТИРУЕМ CATBOOST ДЛЯ МАКСИМАЛЬНОЙ ТОЧНОСТИ ---
import catboost as cb
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
import warnings
import time

warnings.filterwarnings('ignore')

# Настройка страницы
st.set_page_config(page_title="Анализ оттока клиентов банка", page_icon="🌸", layout="wide")

# Кастомный CSS
st.markdown("""
<style>
    .main { background-color: #FFF0F5; }
    .stButton>button { background-color: #FFB6C1; color: #8B0000; border-radius: 10px; border: 1px solid #DB7093; }
    .stButton>button:hover { background-color: #DB7093; color: white; }
    h1, h2, h3 { color: #8B0000; font-family: 'Arial', sans-serif; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #8B0000;'>🌸 Анализ оттока клиентов банка 🌸</h1>",
            unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #DB7093;'>Прогнозирование ухода клиентов с помощью машинного обучения</p>",
    unsafe_allow_html=True)


# --- ИЗМЕНЕНИЕ: РАСШИРЕННАЯ ФУНКЦИЯ ГЕНЕРАЦИИ ПРИЗНАКОВ ---
def engineer_features(df):
    """Создает новые сложные признаки для максимальной производительности модели."""
    df_copy = df.copy()

    # Полиномиальные и сложные признаки
    df_copy['Age_x_Tenure'] = df_copy['Age'] * df_copy['Tenure']
    df_copy['CreditScore_x_Age'] = df_copy['CreditScore'] * df_copy['Age']

    # Признаки-отношения
    df_copy['BalanceSalaryRatio'] = df_copy['Balance'] / (df_copy['EstimatedSalary'] + 1e-6)
    df_copy['TenureAgeRatio'] = df_copy['Tenure'] / (df_copy['Age'] + 1e-6)
    df_copy['CreditScoreAgeRatio'] = df_copy['CreditScore'] / (df_copy['Age'] + 1e-6)

    # Взаимодействие с географией (очень важный признак)
    if 'Germany' in df_copy.columns:
        df_copy['Germany_x_Balance'] = df_copy['Germany'] * df_copy['Balance']
        df_copy['Germany_x_Age'] = df_copy['Germany'] * df_copy['Age']

    # Бинарные флаги
    df_copy['IsBalanceZero'] = (df_copy['Balance'] == 0).astype(int)
    df_copy['IsSenior'] = (df_copy['Age'] >= 60).astype(int)

    # Замена бесконечных значений и NaN
    df_copy.replace([np.inf, -np.inf], 0, inplace=True)
    df_copy.fillna(0, inplace=True)

    return df_copy


@st.cache_data
def load_and_prepare_data():
    try:
        try:
            train_df = pd.read_csv('bank_train.csv')
            valid_df = pd.read_csv('bank_valid.csv')
        except FileNotFoundError:
            st.warning("Файлы 'bank_train.csv' и 'bank_valid.csv' не найдены. Используется 'finance_new.csv'.")
            df_full = pd.read_csv('finance_new.csv')
            train_df, valid_df = train_test_split(df_full, test_size=0.2, random_state=42,
                                                  stratify=df_full.get('Exited'))

        prepared_dfs = {}
        for name, df in {'train': train_df, 'valid': valid_df}.items():
            df_copy = df.copy()
            for col in df_copy.columns:
                if df_copy[col].dtype == 'object':
                    df_copy[col] = pd.to_numeric(df_copy[col].str.replace('.', '', regex=False), errors='coerce')
            base_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                             'EstimatedSalary', 'Exited', 'France', 'Germany', 'Spain', 'Female', 'Male']
            cols_to_keep = [col for col in base_features if col in df_copy.columns]
            df_copy = df_copy[cols_to_keep]
            for col in df_copy.columns:
                if df_copy[col].isnull().any():
                    df_copy[col].fillna(df_copy[col].median(), inplace=True)
            df_featured = engineer_features(df_copy)
            prepared_dfs[name] = df_featured
        return prepared_dfs['train'], prepared_dfs['valid']
    except FileNotFoundError:
        st.error("Не найден файл 'finance_new.csv'. Загрузите данные для работы.")
        return None, None


train_df_featured, valid_df_featured = load_and_prepare_data()

if train_df_featured is not None:
    numerical_cols = [col for col in train_df_featured.select_dtypes(include=np.number).columns if
                      col not in ['Exited']]

    # Правильная строка
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Данные", "🤖 Обучение", "📋 Результаты", "🎯 Прогноз"])
    with tab1:
        st.header("Обзор данных с новыми признаками")
        st.dataframe(train_df_featured.head())

    with tab2:
        st.header("Обучение модели")

        X_train = train_df_featured.drop('Exited', axis=1)
        y_train = train_df_featured['Exited']
        X_valid = valid_df_featured.drop('Exited', axis=1, errors='ignore').reindex(columns=X_train.columns,
                                                                                    fill_value=0)
        y_valid = valid_df_featured['Exited']

        st.session_state.X_columns = X_train.columns

        with st.spinner("Масштабирование данных..."):
            scaler = StandardScaler()
            st.session_state.numerical_cols = [col for col in numerical_cols if col in X_train.columns]
            X_train[st.session_state.numerical_cols] = scaler.fit_transform(X_train[st.session_state.numerical_cols])
            X_valid[st.session_state.numerical_cols] = scaler.transform(X_valid[st.session_state.numerical_cols])
            st.session_state.scaler = scaler
        st.success("Данные подготовлены!")

        with st.spinner('Обучение CatBoost для максимальной точности (может занять время)...'):
            # --- ИЗМЕНЕНИЕ: CATBOOST С НАСТРОЙКАМИ ДЛЯ МАКСИМАЛЬНОЙ ТОЧНОСТИ ---
            cat_model = cb.CatBoostClassifier(
                iterations=2500,
                learning_rate=0.03,
                depth=7,
                l2_leaf_reg=3,
                loss_function='Logloss',
                eval_metric='Accuracy',
                random_seed=42,
                verbose=0,
                early_stopping_rounds=50
            )
            cat_model.fit(X_train, y_train, eval_set=(X_valid, y_valid))
            st.session_state.model = cat_model
        st.success(f"Модель CatBoost обучена! Оптимальное количество итераций: {cat_model.get_best_iteration()}")

    with tab3:
        st.header("Результаты на валидационном наборе")
        if 'model' in st.session_state:
            X_valid_eval = valid_df_featured.drop('Exited', axis=1, errors='ignore').reindex(
                columns=st.session_state.X_columns, fill_value=0)
            y_valid_eval = valid_df_featured['Exited']
            X_valid_eval[st.session_state.numerical_cols] = st.session_state.scaler.transform(
                X_valid_eval[st.session_state.numerical_cols])

            start_time = time.time()
            y_pred_valid = st.session_state.model.predict(X_valid_eval)
            prediction_time = time.time() - start_time
            y_pred_proba_valid = st.session_state.model.predict_proba(X_valid_eval)[:, 1]

            st.subheader("Оценка производительности")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("**Accuracy**", f"{accuracy_score(y_valid_eval, y_pred_valid):.4f}",
                          help="Доля верных предсказаний.")
            with col2:
                st.metric("**ROC-AUC Score**", f"{roc_auc_score(y_valid_eval, y_pred_proba_valid):.4f}",
                          help="Качество модели, чем ближе к 1, тем лучше.")
            with col3:
                st.metric("**Время предсказания**", f"{prediction_time:.4f} сек")
            st.subheader("Матрица ошибок")
            cm = confusion_matrix(y_valid_eval, y_pred_valid)
            st.plotly_chart(
                px.imshow(cm, text_auto=True, labels=dict(x="Предсказание", y="Истина"), color_continuous_scale='Reds'),
                use_container_width=True)
        else:
            st.warning("Модель не обучена.")

    with tab4:
        st.header("🎯 Прогнозирование оттока для нового клиента")
        if 'model' in st.session_state:
            with st.form("prediction_form"):
                # Форма для ввода данных клиента (без изменений)
                col1, col2, col3 = st.columns(3)
                with col1:
                    CreditScore = st.slider("Кредитный рейтинг", 300, 850, 650)
                    Age = st.slider("Возраст", 18, 100, 38)
                    Geography = st.selectbox("Страна", ["France", "Germany", "Spain"])
                    Gender = st.selectbox("Пол", ["Male", "Female"])
                with col2:
                    Tenure = st.slider("Срок в банке (лет)", 0, 10, 5)
                    Balance = st.number_input("Баланс", 0.0, 250000.0, 75000.0)
                    NumOfProducts = st.slider("Кол-во продуктов", 1, 4, 1)
                with col3:
                    HasCrCard = st.radio("Есть кред. карта?", [1, 0], format_func=lambda x: "Да" if x == 1 else "Нет")
                    IsActiveMember = st.radio("Активный клиент?", [1, 0],
                                              format_func=lambda x: "Да" if x == 1 else "Нет")
                    EstimatedSalary = st.number_input("Предп. з/п", 0.0, 200000.0, 100000.0)
                submitted = st.form_submit_button("Сделать прогноз")

            if submitted:
                input_data_dict = {'CreditScore': CreditScore, 'Age': Age, 'Tenure': Tenure, 'Balance': Balance,
                                   'NumOfProducts': NumOfProducts, 'HasCrCard': HasCrCard,
                                   'IsActiveMember': IsActiveMember,
                                   'EstimatedSalary': EstimatedSalary, 'France': 1 if Geography == 'France' else 0,
                                   'Germany': 1 if Geography == 'Germany' else 0,
                                   'Spain': 1 if Geography == 'Spain' else 0,
                                   'Female': 1 if Gender == 'Female' else 0, 'Male': 1 if Gender == 'Male' else 0}
                input_df = pd.DataFrame([input_data_dict])
                input_df_featured = engineer_features(input_df)
                input_df_reordered = input_df_featured.reindex(columns=st.session_state.X_columns, fill_value=0)
                input_df_reordered[st.session_state.numerical_cols] = st.session_state.scaler.transform(
                    input_df_reordered[st.session_state.numerical_cols])

                prediction_proba = st.session_state.model.predict_proba(input_df_reordered)[0][1]

                if prediction_proba > 0.55:  # Немного поднимаем порог для большей уверенности в уходе
                    st.error(f"Высокая вероятность ухода: **{prediction_proba:.1%}**", icon="🚨")
                elif prediction_proba > 0.4:
                    st.warning(f"Средняя вероятность ухода: **{prediction_proba:.1%}**", icon="⚠️")
                else:
                    st.success(f"Низкая вероятность ухода: **{prediction_proba:.1%}**", icon="✅")
        else:
            st.warning("Сначала обучите модель.")
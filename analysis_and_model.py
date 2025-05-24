import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_curve, roc_auc_score)
from imblearn.over_sampling import RandomOverSampler



def analysis_and_model_page():
    st.title("📊 Анализ данных и модель предиктивного обслуживания")

    # Загрузка данных с обработкой ошибок
    @st.cache_data
    def load_data():
        try:

            # Загрузка датасета
            data = pd.read_csv("data/predictive_maintenance.csv")

            # Проверка, что данные загрузились корректно
            if data.empty:
                raise ValueError("Загружен пустой DataFrame")

            # Предобработка данных
            required_columns = ['Type', 'Air temperature [K]', 'Process temperature [K]',
                                'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
                                'Machine failure']

            if not all(col in data.columns for col in required_columns):
                raise ValueError("В данных отсутствуют необходимые столбцы")

            data = data[required_columns]  # Оставляем только нужные столбцы
            data['Type'] = LabelEncoder().fit_transform(data['Type'])

            return data

        except Exception as e:
            st.error(f"Ошибка загрузки данных: {str(e)}")
            st.warning("Используются демонстрационные данные")

            # Создаем минимальный демо-набор данных
            return pd.DataFrame({
                'Type': [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
                'Air temperature [K]': [298, 305, 302, 300, 299, 303, 301, 304, 297, 299],
                'Process temperature [K]': [308, 315, 312, 310, 309, 313, 311, 314, 307, 309],
                'Rotational speed [rpm]': [1550, 1450, 1500, 1520, 1480, 1490, 1510, 1470, 1530, 1540],
                'Torque [Nm]': [42, 38, 40, 41, 39, 40, 41, 39, 43, 38],
                'Tool wear [min]': [10, 210, 110, 50, 180, 200, 30, 220, 5, 190],
                'Machine failure': [0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
            })

    data = load_data()

    if data is None:
        st.error("Не удалось загрузить данные. Пожалуйста, проверьте подключение к интернету.")
        return

    st.success(f"Данные успешно загружены! Записей: {len(data)}")

    # Проверка баланса классов
    class_balance = data['Machine failure'].value_counts(normalize=True)
    if class_balance.min() < 0.1:  # Если один из классов меньше 10%
        st.warning(f"Сильный дисбаланс классов: {class_balance[1]:.1%} отказов vs {class_balance[0]:.1%} нормы")

    # Разделы приложения
    tab1, tab2, tab3 = st.tabs(["📈 Анализ данных", "🤖 Обучение модели", "🔮 Прогнозирование"])

    with tab1:
        st.subheader("Предпросмотр данных")
        st.dataframe(data.head())

        st.subheader("Статистика данных")
        st.write(data.describe())

        st.subheader("Распределение целевой переменной")
        fig, ax = plt.subplots()
        sns.countplot(x='Machine failure', data=data, ax=ax)
        ax.set_title("Количество отказов оборудования")
        st.pyplot(fig)

        st.subheader("Корреляция признаков")
        corr = data.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    with tab2:
        st.subheader("Настройки обучения")

        # Разделение данных
        X = data.drop(columns=['Machine failure'])
        y = data['Machine failure']

        # Проверка минимального количества образцов
        min_samples = 10  # Минимальное количество образцов для обучения

        if len(X) < min_samples:
            st.error(f"Недостаточно данных для обучения. Требуется минимум {min_samples} образцов.")
            return

        # Обработка дисбаланса (используем RandomOverSampler вместо SMOTE)
        ros = RandomOverSampler(random_state=42)
        try:
            X_res, y_res = ros.fit_resample(X, y)
            st.info(f"Балансировка классов применена. Новый размер данных: {len(X_res)}")
        except Exception as e:
            st.error(f"Ошибка балансировки классов: {str(e)}")
            X_res, y_res = X, y  # Используем исходные данные если балансировка не удалась

        # Разделение на train/test
        test_size = st.slider("Размер тестовой выборки (%)", 10, 40, 20)
        X_train, X_test, y_train, y_test = train_test_split(
            X_res, y_res, test_size=test_size / 100, random_state=42
        )

        # Масштабирование числовых признаков
        numeric_cols = ['Air temperature [K]', 'Process temperature [K]',
                        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        scaler = StandardScaler()

        try:
            X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
            X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
        except Exception as e:
            st.error(f"Ошибка масштабирования данных: {str(e)}")
            return

        # Выбор модели
        model_type = st.selectbox(
            "Выберите модель",
            ["Логистическая регрессия", "Случайный лес", "XGBoost"],
            key="model_select"
        )

        # Настройки моделей
        if model_type == "Логистическая регрессия":
            C = st.slider("Параметр регуляризации (C)", 0.01, 10.0, 1.0)
            max_iter = st.slider("Максимальное число итераций", 100, 1000, 100)
            model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)

        elif model_type == "Случайный лес":
            n_estimators = st.slider("Количество деревьев", 10, 200, 100)
            max_depth = st.selectbox("Максимальная глубина", [None, 5, 10, 15, 20])
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )

        else:  # XGBoost
            n_estimators = st.slider("Количество деревьев", 10, 200, 100)
            learning_rate = st.slider("Learning rate", 0.01, 0.5, 0.1)
            model = XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=42,
                eval_metric='logloss'
            )

        if st.button("Обучить модель", type="primary"):
            with st.spinner("Идёт обучение модели..."):
                try:
                    model.fit(X_train, y_train)

                    # Оценка
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)[:, 1]

                    # Метрики
                    accuracy = accuracy_score(y_test, y_pred)
                    conf_matrix = confusion_matrix(y_test, y_pred)
                    class_report = classification_report(y_test, y_pred)
                    roc_auc = roc_auc_score(y_test, y_proba)

                    # Сохранение модели в session state
                    st.session_state.model = model
                    st.session_state.scaler = scaler
                    st.session_state.accuracy = accuracy
                    st.session_state.roc_auc = roc_auc
                    st.session_state.numeric_cols = numeric_cols

                    # Вывод результатов
                    st.success("Обучение завершено!")
                    st.metric("Accuracy", f"{accuracy:.2%}")
                    st.metric("ROC-AUC", f"{roc_auc:.2%}")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Матрица ошибок")
                        fig, ax = plt.subplots()
                        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
                        ax.set_xlabel("Предсказанные")
                        ax.set_ylabel("Фактические")
                        st.pyplot(fig)

                    with col2:
                        st.subheader("ROC-кривая")
                        fpr, tpr, _ = roc_curve(y_test, y_proba)
                        fig, ax = plt.subplots()
                        ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
                        ax.plot([0, 1], [0, 1], 'k--')
                        ax.set_xlabel("False Positive Rate")
                        ax.set_ylabel("True Positive Rate")
                        ax.legend()
                        st.pyplot(fig)

                    st.subheader("Отчёт классификации")
                    st.code(class_report)

                except Exception as e:
                    st.error(f"Ошибка при обучении модели: {str(e)}")

    with tab3:
        st.subheader("Прогнозирование отказа оборудования")

        if 'model' not in st.session_state:
            st.warning("Сначала обучите модель на вкладке 'Обучение модели'")
        else:
            with st.form("prediction_form"):
                st.write("Введите параметры оборудования:")

                col1, col2 = st.columns(2)
                with col1:
                    type_ = st.selectbox("Тип оборудования", ["L", "M", "H"])
                    air_temp = st.number_input("Температура воздуха [K]", min_value=250.0, max_value=350.0, value=300.0)
                    process_temp = st.number_input("Температура процесса [K]", min_value=250.0, max_value=350.0,
                                                   value=310.0)

                with col2:
                    rotational_speed = st.number_input("Скорость вращения [rpm]", min_value=1000, max_value=3000,
                                                       value=1500)
                    torque = st.number_input("Крутящий момент [Nm]", min_value=0.0, max_value=100.0, value=40.0)
                    tool_wear = st.number_input("Износ инструмента [min]", min_value=0, max_value=300, value=0)

                submitted = st.form_submit_button("Сделать прогноз")

                if submitted:
                    try:
                        # Подготовка входных данных
                        input_data = pd.DataFrame({
                            'Type': [0 if type_ == 'L' else 1 if type_ == 'M' else 2],
                            'Air temperature [K]': [air_temp],
                            'Process temperature [K]': [process_temp],
                            'Rotational speed [rpm]': [rotational_speed],
                            'Torque [Nm]': [torque],
                            'Tool wear [min]': [tool_wear]
                        })

                        # Масштабирование
                        input_data_scaled = input_data.copy()
                        input_data_scaled[st.session_state.numeric_cols] = st.session_state.scaler.transform(
                            input_data[st.session_state.numeric_cols]
                        )

                        # Прогнозирование
                        model = st.session_state.model
                        prediction = model.predict(input_data_scaled)[0]
                        proba = model.predict_proba(input_data_scaled)[0][1]

                        # Визуализация результата
                        if prediction == 1:
                            st.error(f"⚠️ ВНИМАНИЕ: Прогнозируется отказ оборудования!")
                            st.write(f"Вероятность отказа: {proba:.1%}")
                            st.image("https://img.icons8.com/color/96/high-risk.png", width=100)
                        else:
                            st.success(f"✅ Оборудование работает нормально")
                            st.write(f"Вероятность отказа: {proba:.1%}")
                            st.image("https://img.icons8.com/color/96/ok--v1.png", width=100)

                        # Дополнительная информация
                        st.info(f"Точность модели: {st.session_state.accuracy:.1%}")
                        st.info(f"ROC-AUC модели: {st.session_state.roc_auc:.2f}")

                    except Exception as e:
                        st.error(f"Ошибка при прогнозировании: {str(e)}")
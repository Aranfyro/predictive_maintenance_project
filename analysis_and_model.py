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

            # Переименование столбцов
            data = data.rename(columns={
                'Air temperature [K]': 'Air_temperature_K',
                'Process temperature [K]': 'Process_temperature_K',
                'Rotational speed [rpm]': 'Rotational_speed_rpm',
                'Torque [Nm]': 'Torque_Nm',
                'Tool wear [min]': 'Tool_wear_min'
            })

            # Обновленный список требуемых столбцов с новыми именами
            required_columns = ['Type', 'Air_temperature_K', 'Process_temperature_K',
                                'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min',
                                'TWF', 'HDF', 'PWF', 'OSF', 'RNF', 'Machine failure']

            if not all(col in data.columns for col in required_columns):
                missing = [col for col in required_columns if col not in data.columns]
                raise ValueError(f"В данных отсутствуют необходимые столбцы: {missing}")

            data = data[required_columns]
            data['Type'] = LabelEncoder().fit_transform(data['Type'])

            # Создаем новую целевую переменную
            data['Failure_Type'] = 0  # По умолчанию нет отказа
            data.loc[data['TWF'] == 1, 'Failure_Type'] = 1
            data.loc[data['HDF'] == 1, 'Failure_Type'] = 2
            data.loc[data['PWF'] == 1, 'Failure_Type'] = 3
            data.loc[data['OSF'] == 1, 'Failure_Type'] = 4
            data.loc[data['RNF'] == 1, 'Failure_Type'] = 5

            # Удаляем исходные колонки с отказами
            data = data.drop(columns=['TWF', 'HDF', 'PWF', 'OSF', 'RNF', 'Machine failure'])

            return data

        except Exception as e:
            st.error(f"Ошибка загрузки данных: {str(e)}")
            st.warning("Используются демонстрационные данные")
            return pd.DataFrame({
                'Type': [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
                'Air_temperature_K': [298, 305, 302, 300, 299, 303, 301, 304, 297, 299],
                'Process_temperature_K': [308, 315, 312, 310, 309, 313, 311, 314, 307, 309],
                'Rotational_speed_rpm': [1550, 1450, 1500, 1520, 1480, 1490, 1510, 1470, 1530, 1540],
                'Torque_Nm': [42, 38, 40, 41, 39, 40, 41, 39, 43, 38],
                'Tool_wear_min': [10, 210, 110, 50, 180, 200, 30, 220, 5, 190],
                'Failure_Type': [0, 1, 0, 0, 2, 3, 0, 4, 0, 5]
            })

    data = load_data()

    if data is None:
        st.error("Не удалось загрузить данные. Пожалуйста, проверьте подключение к интернету.")
        return

    st.success(f"Данные успешно загружены! Записей: {len(data)}")

    # Словарь для меток классов
    failure_labels = {
        0: "Нет отказа",
        1: "TWF (Износ инструмента)",
        2: "HDF (Теплоотвод)",
        3: "PWF (Мощность)",
        4: "OSF (Перегрузка)",
        5: "RNF (Случайный)"
    }

    # Разделы приложения
    tab1, tab2, tab3 = st.tabs(["📈 Анализ данных", "🤖 Обучение модели", "🔮 Прогнозирование"])

    with tab1:
        st.subheader("Предпросмотр данных")
        st.dataframe(data.head())

        st.subheader("Статистика данных")
        st.write(data.describe())

        st.subheader("Распределение типов отказов")
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_data = data.copy()
        plot_data['Failure_Label'] = plot_data['Failure_Type'].map(failure_labels)

        sns.countplot(
            y='Failure_Label',  # Горизонтальный график для лучшей читаемости
            data=plot_data,
            ax=ax,
            order=[failure_labels[i] for i in sorted(failure_labels.keys())]
        )
        ax.set_title("Распределение типов отказов")
        ax.set_xlabel("Количество")
        ax.set_ylabel("Тип отказа")
        st.pyplot(fig)

        st.subheader("Корреляция признаков")
        corr = data.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    with tab2:
        st.subheader("Настройки обучения")

        # Разделение данных
        X = data.drop(columns=['Failure_Type'])
        y = data['Failure_Type']

        if len(X) < 10:
            st.error("Недостаточно данных для обучения. Требуется минимум 10 образцов.")
            return

        # Обработка дисбаланса
        ros = RandomOverSampler(random_state=42)
        try:
            X_res, y_res = ros.fit_resample(X, y)
            st.info(f"Балансировка классов применена. Новый размер данных: {len(X_res)}")
        except Exception as e:
            st.error(f"Ошибка балансировки классов: {str(e)}")
            X_res, y_res = X, y

        # Разделение на train/test
        test_size = st.slider("Размер тестовой выборки (%)", 10, 40, 20)
        X_train, X_test, y_train, y_test = train_test_split(
            X_res, y_res, test_size=test_size / 100, random_state=42
        )

        # Масштабирование числовых признаков
        numeric_cols = ['Air_temperature_K', 'Process_temperature_K',
                        'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min']
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
            model = LogisticRegression(C=C, max_iter=max_iter, random_state=42, multi_class='multinomial')

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
                eval_metric='mlogloss',
                objective='multi:softprob'
            )

        if st.button("Обучить модель", type="primary"):
            with st.spinner("Идёт обучение модели..."):
                try:
                    # Для XGBoost очищаем имена признаков
                    if model_type == "XGBoost":
                        X_train_clean = X_train.copy()
                        X_train_clean.columns = [str(col).replace('[', '').replace(']', '').replace('<', '')
                                                 for col in X_train.columns]
                        model.fit(X_train_clean, y_train)
                    else:
                        model.fit(X_train, y_train)

                    # Оценка модели
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)

                    # Метрики
                    accuracy = accuracy_score(y_test, y_pred)
                    conf_matrix = confusion_matrix(y_test, y_pred)
                    class_report = classification_report(y_test, y_pred, target_names=failure_labels.values())

                    # Сохранение модели
                    st.session_state.update({
                        'model': model,
                        'scaler': scaler,
                        'accuracy': accuracy,
                        'class_report': class_report,
                        'numeric_cols': numeric_cols,
                        'failure_labels': failure_labels,
                        'model_type': model_type
                    })

                    # Вывод результатов
                    st.success("Обучение завершено!")
                    st.metric("Accuracy", f"{accuracy:.2%}")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Матрица ошибок")
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
                                    xticklabels=failure_labels.values(),
                                    yticklabels=failure_labels.values())
                        ax.set_xlabel("Предсказанные")
                        ax.set_ylabel("Фактические")
                        plt.xticks(rotation=45)
                        plt.yticks(rotation=0)
                        st.pyplot(fig)

                    with col2:
                        st.subheader("Отчёт классификации")
                        st.text(class_report)

                except Exception as e:
                    st.error(f"Ошибка при обучении модели: {str(e)}")

    with tab3:
        st.subheader("Прогнозирование типа отказа оборудования")

        if 'model' not in st.session_state:
            st.warning("Сначала обучите модель на вкладке 'Обучение модели'")
        else:
            with st.form("prediction_form"):
                st.write("Введите параметры оборудования:")

                col1, col2 = st.columns(2)
                with col1:
                    type_ = st.selectbox("Тип оборудования", ["L", "M", "H"])
                    air_temp = st.number_input("Температура воздуха (K)", min_value=250.0, max_value=350.0, value=300.0)
                    process_temp = st.number_input("Температура процесса (K)", min_value=250.0, max_value=350.0,
                                                   value=310.0)

                with col2:
                    rotational_speed = st.number_input("Скорость вращения (rpm)", min_value=1000, max_value=3000,
                                                       value=1500)
                    torque = st.number_input("Крутящий момент (Nm)", min_value=0.0, max_value=100.0, value=40.0)
                    tool_wear = st.number_input("Износ инструмента (min)", min_value=0, max_value=300, value=0)

                submitted = st.form_submit_button("Сделать прогноз")

                if submitted:
                    try:
                        # Подготовка входных данных
                        input_data = pd.DataFrame({
                            'Type': [0 if type_ == 'L' else 1 if type_ == 'M' else 2],
                            'Air_temperature_K': [air_temp],
                            'Process_temperature_K': [process_temp],
                            'Rotational_speed_rpm': [rotational_speed],
                            'Torque_Nm': [torque],
                            'Tool_wear_min': [tool_wear]
                        })

                        # Масштабирование
                        input_data_scaled = input_data.copy()
                        input_data_scaled[st.session_state['numeric_cols']] = st.session_state['scaler'].transform(
                            input_data[st.session_state['numeric_cols']]
                        )

                        # Для XGBoost очищаем имена признаков
                        if st.session_state.get('model_type') == "XGBoost":
                            input_data_scaled.columns = [str(col).replace('[', '').replace(']', '').replace('<', '')
                                                         for col in input_data_scaled.columns]

                        # Прогнозирование
                        model = st.session_state['model']
                        prediction = model.predict(input_data_scaled)[0]
                        proba = model.predict_proba(input_data_scaled)[0]

                        # Визуализация результата
                        st.subheader("Результат прогнозирования")
                        if prediction == 0:
                            st.success(f"✅ Оборудование работает нормально")
                        else:
                            st.error(
                                f"⚠️ ВНИМАНИЕ: Прогнозируется отказ типа {st.session_state['failure_labels'][prediction]}")

                        # Вероятности для всех классов
                        st.subheader("Вероятности для каждого типа отказа")
                        proba_df = pd.DataFrame(
                            {'Тип отказа': [st.session_state['failure_labels'][i] for i in
                                           range(len(st.session_state['failure_labels']))],
                            'Вероятность': proba
                        }).sort_values('Вероятность', ascending=False)

                        fig, ax = plt.subplots(figsize=(10, 5))
                        sns.barplot(x='Вероятность', y='Тип отказа', data=proba_df, ax=ax)
                        ax.set_title("Вероятности типов отказов")
                        st.pyplot(fig)

                        # Дополнительная информация
                        st.info(f"Точность модели: {st.session_state['accuracy']:.1%}")
                        st.text("Отчёт классификации модели:\n" + st.session_state['class_report'])

                    except Exception as e:
                        st.error(f"Ошибка при прогнозировании: {str(e)}")

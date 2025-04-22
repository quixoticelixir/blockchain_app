import streamlit as st
import numpy as np
import pandas as pd
import re

from utils.preprocessing import load_data, preprocess_data
from utils.clustering import find_optimal_clusters, perform_clustering
from utils.plots import (
    plot_elbow_method,
    plot_silhouette,
    plot_davies_bouldin,
    plot_pca_clusters
)
from utils.eda import generate_eda_plots
from utils.gigachat_api import get_ai_description_from_stats
from src.fetch_wallet import run_fetch_and_process


default_session_state = {
    'data_source': None, # 'csv' или 'api'
    'data_loaded': False,
    'fetch_error': None,
    'fetch_warnings': None,
    'api_address_input': "0x514910771AF9Ca656af840dff83E8264EcF986CA",
    'api_days_input': 15,
    'cluster_performed': False,
    'original_data': None,
    'processed_data': None,
    'scaled_features': None,
    'cluster_metrics': None,
    'cluster_description': None,
    'displayed_stats': None
}
for key, default in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.set_page_config(
    page_title="Wallet Clustering Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Анализ и кластеризация кошельков")

st.sidebar.title("Источник данных")
data_source_option = st.sidebar.radio(
    "Выберите способ получения данных:",
    ('Загрузить CSV', 'Собрать через API Etherscan'),
    index=None,
    key='data_source_choice'
)

# Обновляем состояние при выборе
if data_source_option == 'Загрузить CSV':
    st.session_state.data_source = 'csv'
elif data_source_option == 'Собрать через API Etherscan':
    st.session_state.data_source = 'api'

st.markdown("### 1. Загрузка или Сбор данных")

if st.session_state.data_source == 'csv':
    uploaded_file = st.file_uploader("Выберите CSV файл", type=["csv"], key="csv_uploader")
    if uploaded_file is not None and not st.session_state.data_loaded:
        try:
            with st.spinner("Загрузка и предобработка данных из CSV..."):
                data = load_data(uploaded_file)
                st.session_state.original_data = data
                # Предобработка сразу после загрузки
                scaled_features, processed_data = preprocess_data(data)
                st.session_state.scaled_features = scaled_features
                st.session_state.processed_data = processed_data
                st.session_state.data_loaded = True
                st.session_state.fetch_error = None #
                st.success("Данные из CSV успешно загружены и обработаны!")
                st.rerun()
        except Exception as e:
            st.error(f"Ошибка при загрузке или обработке CSV: {str(e)}")
            st.session_state.data_loaded = False
            st.session_state.original_data = None
            st.session_state.processed_data = None
            st.session_state.scaled_features = None

elif st.session_state.data_source == 'api':
    st.subheader("Параметры для сбора данных через API")
    etherscan_api_key = st.secrets.get("ETHERSCAN_API_KEY")

    if not etherscan_api_key:
        st.warning("""
            **Ключ API Etherscan не найден!**

            Пожалуйста, убедитесь, что вы добавили строку
            `ETHERSCAN_API_KEY = "ВАШ_КЛЮЧ"`
            в ваш файл секретов `.streamlit/secrets.toml`.

            Без ключа API сбор данных через Etherscan невозможен.
        """)
        st.stop() # Останавливаем выполнение приложения, так как ключ необходим

    # st.success("Ключ API Etherscan успешно загружен из секретов.")

    # Поля ввода для API
    api_address = st.text_input(
        "Адрес контракта токена (ERC-20)",
        value=st.session_state.api_address_input,
        key="api_address"
    )
    api_days = st.number_input(
        "Количество дней для анализа (назад от текущей даты)",
        min_value=1,
        max_value=365,
        value=st.session_state.api_days_input,
        step=1,
        key="api_days"
    )

    if st.button("Начать сбор данных", key="start_api_fetch"):
        if not re.match(r'^0x[a-fA-F0-9]{40}$', api_address):
             st.error("Неверный формат адреса Ethereum. Адрес должен начинаться с '0x' и содержать 40 шестнадцатеричных символов.")
        else:
            st.session_state.api_address_input = api_address
            st.session_state.api_days_input = api_days
            st.session_state.data_loaded = False
            st.session_state.fetch_error = None
            st.session_state.fetch_warnings = None
            st.session_state.original_data = None
            st.session_state.processed_data = None
            st.session_state.scaled_features = None

            st.info(f"Запуск сбора данных для токена {api_address} за последние {api_days} дней...")
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(percent_complete, message):
                progress_bar.progress(percent_complete / 100.0)
                status_text.info(message)

            try:
                # Вызов функции из fetch_wallet с передачей callback
                df_result, warnings_list = run_fetch_and_process(
                    target_token_contract_address=api_address,
                    days_back=api_days,
                    api_key=etherscan_api_key,
                    progress_callback=update_progress
                )

                status_text.empty()
                progress_bar.empty()

                if df_result is not None:
                    if not df_result.empty:
                        st.success(f"Сбор данных завершен! Получено {len(df_result)} записей.")
                        st.session_state.original_data = df_result
                        st.session_state.fetch_warnings = warnings_list # Сохраняем предупреждения

                        # Запускаем предобработку сразу после сбора
                        with st.spinner("Предобработка собранных данных..."):
                            scaled_features, processed_data = preprocess_data(df_result)
                            st.session_state.scaled_features = scaled_features
                            st.session_state.processed_data = processed_data
                            st.session_state.data_loaded = True # Устанавливаем флаг успешной загрузки/сбора
                        st.success("Предобработка данных завершена.")
                        st.rerun() # Перезапускаем для отображения EDA и следующих шагов

                    else:
                        st.warning("Сбор данных завершен, но не найдено кошельков или транзакций для анализа за указанный период.")
                        st.session_state.data_loaded = False # Данных нет
                else:
                    # Если run_fetch_and_process вернула None, значит была критическая ошибка
                    st.error("Произошла критическая ошибка во время сбора данных. Проверьте консоль или логи для деталей.")
                    st.session_state.fetch_error = "Критическая ошибка сбора данных."
                    st.session_state.data_loaded = False

            except Exception as e:
                status_text.empty()
                progress_bar.empty()
                st.error(f"Произошла ошибка во время выполнения сбора или обработки данных: {str(e)}")
                st.session_state.fetch_error = str(e)
                st.session_state.data_loaded = False
                # Можно добавить вывод traceback для отладки
                # st.exception(e)

elif not st.session_state.data_source:
    st.info("Пожалуйста, выберите источник данных в боковой панели слева.")


# === Последующие шаги анализа (выполняются только если data_loaded is True) ===

if st.session_state.data_loaded and st.session_state.original_data is not None:

    # Отображение предупреждений о лимите 10k, если они были при сборе через API
    if st.session_state.data_source == 'api' and st.session_state.fetch_warnings:
        st.warning("**Предупреждение о неполных данных:**")
        warning_message = "Из-за достижения лимита Etherscan в 10,000 транзакций для следующих дат, данные и результаты анализа могут быть неполными:\n"
        for dt in sorted(list(set(st.session_state.fetch_warnings))): # Уникальные даты
            warning_message += f"- {dt.strftime('%Y-%m-%d')}\n"
        st.markdown(warning_message)


    # === Секция 2: EDA ===
    st.markdown("---")
    st.markdown("### 2. Исследовательский анализ данных (EDA)")
    data = st.session_state.original_data # Используем загруженные/собранные данные

    st.subheader("Первые 5 строк данных")
    st.dataframe(data.head())

    st.subheader("Основная статистика")
    # Исключаем нечисловые колонки перед describe
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        st.dataframe(data[numeric_cols].describe())
    else:
        st.info("Нет числовых колонок для отображения статистики.")


    st.subheader("Распределения данных")
    try:
        # Передаем только числовые колонки в generate_eda_plots
        eda_plots = generate_eda_plots(data[numeric_cols])
        st.subheader("Распределения исходных числовых данных")
        st.pyplot(eda_plots['original_plots'])
        st.subheader("Распределения после log1p-преобразования")
        st.pyplot(eda_plots['log_plots'])
    except Exception as e:
        st.error(f"Ошибка при генерации EDA графиков: {e}")
        st.info("Возможно, в данных отсутствуют необходимые числовые колонки.")


    # === Секция 3: Определение кластеров ===
    st.markdown("---")
    st.markdown("### 3. Определение оптимального числа кластеров")

    if st.session_state.scaled_features is not None:
        max_k = st.slider("Максимальное k для анализа", 2, 20, 10, key="max_k_slider")

        if st.button("Рассчитать метрики кластеризации", key="calc_metrics_btn"):
             with st.spinner("Расчет метрик кластеризации..."):
                try:
                    metrics = find_optimal_clusters(
                        st.session_state.scaled_features,
                        max_k
                    )
                    st.session_state.cluster_metrics = metrics
                    st.success("Расчет метрик завершен.")
                except Exception as e:
                    st.error(f"Ошибка при расчете метрик: {e}")
                    st.session_state.cluster_metrics = None

        if st.session_state.cluster_metrics:
            # Отображение графиков метрик
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(plot_elbow_method(
                    st.session_state.cluster_metrics['inertia'],
                    st.session_state.cluster_metrics['K_range']
                ))
            with col2:
                st.pyplot(plot_silhouette(
                    st.session_state.cluster_metrics['silhouette'],
                    st.session_state.cluster_metrics['K_range']
                ))
            st.pyplot(plot_davies_bouldin(
                st.session_state.cluster_metrics['davies_bouldin'],
                st.session_state.cluster_metrics['K_range']
            ))

            # Рекомендации по k
            try:
                k_range = st.session_state.cluster_metrics['K_range']
                inertia = st.session_state.cluster_metrics['inertia']
                silhouette = st.session_state.cluster_metrics['silhouette']
                davies_bouldin = st.session_state.cluster_metrics['davies_bouldin']

                # Проверка на достаточность данных для расчета локтя
                elbow_k_index = -1
                if len(inertia) >= 3:
                     # Ищем индекс максимального второго дифференциала (изгиба)
                     diff2 = np.diff(inertia, 2)
                     # +2 к индексу, т.к. diff убирает точки и индексация с 0
                     elbow_k_index = np.argmax(diff2) + 2
                else: # Если k_range слишком мал для второго дифференциала
                     elbow_k_index = 0 # По умолчанию берем первый возможный k

                silhouette_k_index = np.argmax(silhouette)
                db_k_index = np.argmin(davies_bouldin)

                st.subheader("Рекомендуемые значения k:")
                st.info(f"""
                    - Метод локтя: **{k_range[elbow_k_index]}** (наибольший изгиб инерции)
                    - Silhouette Score: **{k_range[silhouette_k_index]}** (максимальный скор)
                    - Davies-Bouldin: **{k_range[db_k_index]}** (минимальный индекс)
                """)
                st.markdown("""
                       **Примечание:** Это рекомендации. Выберите 'k', которое наилучшим образом
                       соответствует вашим целям анализа и интерпретируемости кластеров.
                       """)
            except Exception as e:
                 st.warning(f"Не удалось рассчитать рекомендуемые k: {e}")


            # Выбор k пользователем и запуск кластеризации
            recommended_k_default = 4 # Значение по умолчанию, если рекомендации не сработали
            if st.session_state.cluster_metrics:
                 try:
                     # Предлагаем k по силуэту как наиболее часто используемый
                     recommended_k_default = k_range[silhouette_k_index]
                 except: pass # Оставляем 4 если что-то пошло не так

            selected_k = st.number_input(
                "Выберите количество кластеров (k) для финальной модели",
                min_value=2,
                max_value=max_k,
                value=int(recommended_k_default), # Преобразуем в int на всякий случай
                step=1,
                key="selected_k_input"
            )

            if st.button("Запустить кластеризацию", key="run_clustering_btn"):
                 with st.spinner(f"Выполнение KMeans с k={selected_k}..."):
                    try:
                        labels = perform_clustering(
                            st.session_state.scaled_features,
                            selected_k
                        )
                        # Добавляем метки кластеров к обработанным данным
                        processed_data_copy = st.session_state.processed_data.copy()
                        processed_data_copy['cluster'] = labels
                        st.session_state.processed_data = processed_data_copy # Обновляем стейт

                        # Важно: Добавляем метки кластеров и к ОРИГИНАЛЬНЫМ данным для статистики
                        if len(st.session_state.original_data) == len(labels):
                             original_data_copy = st.session_state.original_data.copy()
                             original_data_copy['cluster'] = labels
                             st.session_state.original_data = original_data_copy # Обновляем стейт
                             st.session_state.cluster_performed = True
                             st.success(f"Кластеризация завершена! Найдено кластеров: {selected_k}")
                             # Сбрасываем описание AI, т.к. кластеры изменились
                             st.session_state.cluster_description = None
                             st.session_state.displayed_stats = None # Сбрасываем старую статистику
                             st.rerun() # Обновляем страницу для показа результатов
                        else:
                            st.error("Ошибка: Несовпадение количества строк между оригинальными данными и результатами кластеризации. Не удалось добавить метки.")
                            st.session_state.cluster_performed = False

                    except Exception as e:
                        st.error(f"Ошибка при выполнении кластеризации: {e}")
                        st.session_state.cluster_performed = False

    else:
         st.info("Данные еще не загружены или не обработаны. Загрузите CSV или соберите данные через API.")


    # === Секция 4: Результаты кластеризации ===
    if st.session_state.cluster_performed and 'cluster' in st.session_state.original_data.columns:
        st.markdown("---")
        st.markdown("### 4. Результаты кластеризации")

        # Визуализация PCA
        st.subheader("Визуализация кластеров (PCA)")
        try:
            st.pyplot(plot_pca_clusters(
                st.session_state.scaled_features,
                st.session_state.original_data['cluster'] # Используем метки из original_data
            ))
        except Exception as e:
            st.error(f"Ошибка при построении PCA графика: {e}")

        st.subheader("Статистика по кластерам (на основе оригинальных данных)")
        try:
            # Выбираем только числовые колонки из оригинальных данных (ИСКЛЮЧАЯ 'cluster')
            original_numeric_cols = st.session_state.original_data.select_dtypes(include=np.number).columns.tolist()
            if 'cluster' in original_numeric_cols:
                original_numeric_cols.remove('cluster')

            if original_numeric_cols: # Если есть числовые колонки для анализа
                 # Группируем оригинальные данные по кластерам и считаем describe
                 stats = st.session_state.original_data.groupby('cluster')[original_numeric_cols].describe()

                 # Оставляем только основные метрики для отображения
                 stats_to_display = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
                 # Используем MultiIndex для выбора нужных уровней статистики
                 filtered_stats = stats.loc[:, pd.IndexSlice[:, stats_to_display]]

                 st.session_state.displayed_stats = filtered_stats # Сохраняем для GigaChat
                 st.dataframe(st.session_state.displayed_stats)
            else:
                 st.info("В оригинальных данных нет числовых колонок для расчета статистики по кластерам.")
                 st.session_state.displayed_stats = None


        except Exception as e:
            st.error(f"Ошибка при расчете статистики по кластерам: {e}")
            st.session_state.displayed_stats = None

        # Распределение адресов по кластерам
        st.subheader("Распределение записей по кластерам")
        st.bar_chart(st.session_state.original_data['cluster'].value_counts())

        # === Секция 5: Описание кластеров GigaChat ===
        st.markdown("---")
        st.markdown("### 5. Описание кластеров с помощью AI (GigaChat)")

        if st.session_state.displayed_stats is not None and not st.session_state.displayed_stats.empty:
            # Подготовка статистики для AI (например, только mean, std, min, max)
            ai_stats = None
            try:
                # Выбираем подмножество статистик для AI
                stats_for_ai = ['mean', 'std', 'min', 'max']
                ai_stats = st.session_state.displayed_stats.loc[:, pd.IndexSlice[:, stats_for_ai]]
                stats_markdown_text = ai_stats.to_markdown() # Конвертируем в Markdown
            except Exception as e:
                st.error(f"Ошибка при подготовке статистики для AI: {str(e)}")
                ai_stats = None

            if ai_stats is not None:
                # Кнопка для запроса описания
                if st.button("Получить описание кластеров от GigaChat", key="get_gigachat_desc"):
                    # Получение ключа GigaChat из секретов
                    auth_basic_value = st.secrets.get('GIGACHAT_AUTH_BASIC_VALUE')

                    if auth_basic_value:
                        with st.spinner("GigaChat анализирует статистику кластеров..."):
                            description = None
                            try:
                                # Вызов функции API GigaChat
                                description = get_ai_description_from_stats(
                                    auth_basic_value=auth_basic_value,
                                    stats_text=stats_markdown_text
                                )

                                if description:
                                    st.session_state.cluster_description = description
                                    st.success("Описание от GigaChat получено!")
                                else:
                                    st.error("Не удалось получить текст описания от GigaChat. API вернул пустой ответ.")
                                    st.session_state.cluster_description = None

                            except Exception as e:
                                st.error(f"Ошибка при взаимодействии с GigaChat API: {e}")
                                st.session_state.cluster_description = None
                                # st.exception(e) # Раскомментировать для детальной отладки

                    else:
                        st.warning(
                            "Для получения описания от GigaChat необходимо добавить `GIGACHAT_AUTH_BASIC_VALUE` (base64 строка Client ID:Client Secret) в секреты Streamlit (`.streamlit/secrets.toml`)."
                        )
            else:
                st.warning("Не удалось подготовить статистику для передачи в GigaChat.")

        elif st.session_state.cluster_performed: # Кластеризация была, но статистика не рассчиталась
            st.warning("Статистика по кластерам не была рассчитана или пуста. Невозможно получить описание от AI.")

        # Отображение полученного описания
        if st.session_state.cluster_description:
            st.markdown("#### Описание кластеров (сгенерировано GigaChat):")
            st.markdown(st.session_state.cluster_description) # Используем markdown для форматирования ответа AI


    # === Секция 6: Сброс ===
    st.markdown("---")
    if st.button("Сбросить и начать заново", key="reset_all"):
        # Очистка всего состояния сессии
        keys_to_clear = list(st.session_state.keys()) # Получаем список всех ключей
        for key in keys_to_clear:
             del st.session_state[key]
        # Можно добавить инициализацию с дефолтными значениями, если необходимо
        # for key, default in default_session_state.items():
        #      st.session_state[key] = default
        st.rerun()


# Сообщение, если данные еще не загружены/собраны в основной части
elif st.session_state.data_source == 'api' and not st.session_state.data_loaded and not st.session_state.fetch_error:
    st.info("Введите параметры и нажмите 'Начать сбор данных' для запуска анализа через API.")
elif st.session_state.fetch_error:
    st.error(f"Произошла ошибка при последней попытке загрузки/сбора данных: {st.session_state.fetch_error}")
    st.info("Исправьте ошибку (например, проверьте API ключ или формат файла) и попробуйте снова.")
import streamlit as st
import numpy as np
from utils.preprocessing import load_data, preprocess_data
from utils.clustering import find_optimal_clusters, perform_clustering
from utils.plots import (
    plot_elbow_method,
    plot_silhouette,
    plot_davies_bouldin,
    plot_pca_clusters
)
from utils.eda import generate_eda_plots

for key, default in {
    'data_loaded': False,
    'cluster_performed': False,
    'original_data': None,
    'processed_data': None,
    'scaled_features': None,
    'cluster_metrics': None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.set_page_config(
    page_title="Aave Wallet Clustering",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("Анализ кластеризации кошельков")

# Загрузка данных
st.markdown("### 1. Загрузка данных")
uploaded_file = st.file_uploader("Выберите CSV файл", type=["csv"])


if uploaded_file is not None and not st.session_state.data_loaded:
    try:
        with st.spinner("Загрузка и предобработка данных..."):
            # Загрузка данных
            data = load_data(uploaded_file)
            st.session_state.original_data = data

            # Предобработка
            scaled_features, processed_data = preprocess_data(data)
            st.session_state.scaled_features = scaled_features
            st.session_state.processed_data = processed_data
            st.session_state.data_loaded = True
            st.success("Данные успешно загружены!")
    except Exception as e:
        st.error(f"Ошибка при загрузке данных: {str(e)}")
        st.session_state.data_loaded = False
        st.session_state.original_data = None

#  EDA
if st.session_state.data_loaded and st.session_state.original_data is not None:
    st.markdown("### 2. Исследовательский анализ данных (EDA)")
    data = st.session_state.original_data

    st.subheader("Первые 5 строк данных")
    st.dataframe(data.head())

    st.subheader("Основная статистика")
    st.dataframe(data.describe())

    st.subheader("Распределения исходных данных")
    eda_plots = generate_eda_plots(data)
    st.pyplot(eda_plots['original_plots'])

    st.subheader("Распределения после log1p-преобразования")
    st.pyplot(eda_plots['log_plots'])

    # Поиск оптимального количества кластеров
    st.markdown("### 3. Определение оптимального числа кластеров")
    max_k = st.slider("Максимальное k для анализа", 2, 20, 10)

    if st.button("Рассчитать метрики"):
        if st.session_state.scaled_features is not None:
            with st.spinner("Расчет метрик кластеризации..."):
                metrics = find_optimal_clusters(
                    st.session_state.scaled_features,
                    max_k
                )
                st.session_state.cluster_metrics = metrics
        else:
            st.error("Ошибка: данные не обработаны")

    if st.session_state.cluster_metrics:
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
        st.subheader("Рекомендуемые значения k:")
        elbow_k = np.argmax(np.diff(st.session_state.cluster_metrics['inertia'], 2)) + 2
        st.info(f"""
            - Метод локтя: **{st.session_state.cluster_metrics['K_range'][elbow_k]}**
            - Silhouette Score: **{st.session_state.cluster_metrics['K_range'][np.argmax(st.session_state.cluster_metrics['silhouette'])]}**
            - Davies-Bouldin: **{st.session_state.cluster_metrics['K_range'][np.argmin(st.session_state.cluster_metrics['davies_bouldin'])]}**
        """)
        st.markdown("""
               **Примечание:** Это рекомендации, основанные на популярных метриках оценки кластеризации.
               Оптимальное число кластеров (k) для финального анализа следует выбирать,
               исходя из конкретных целей и особенностей данных вашей задачи (например,
               требуемого уровня детализации или интерпретируемости кластеров).
               """)

        # Выбор k пользователем
        selected_k = st.number_input(
            "Выберите количество кластеров (k)",
            2,
            max_k,
            4
        )

        if st.button("Запустить кластеризацию"):
            if st.session_state.scaled_features is not None:
                with st.spinner("Выполнение кластеризации..."):
                    labels = perform_clustering(
                        st.session_state.scaled_features,
                        selected_k
                    )
                    st.session_state.processed_data['cluster'] = labels
                    st.session_state.cluster_performed = True
                    st.success(f"Кластеризация завершена! Выбрано кластеров: {selected_k}")
            else:
                st.error("Ошибка: данные не обработаны")

    # Результаты кластеризаци
    if st.session_state.cluster_performed:
        st.markdown("### 4. Результаты кластеризации")

        # Визуализация PCA
        st.subheader("Визуализация кластеров (PCA)")
        st.pyplot(plot_pca_clusters(
            st.session_state.scaled_features,
            st.session_state.processed_data['cluster']
        ))

        st.subheader("Статистика по кластерам (по оригинальным данным)")

        if st.session_state.original_data is not None and 'cluster' in st.session_state.processed_data.columns:
            original_numeric_cols_for_stats = [
                "current_link_balance",
                "period_total_tx_count",
                "period_incoming_tx_count",
                "period_outgoing_tx_count",
                "period_total_volume_in",
                "period_total_volume_out",
                "period_avg_volume_in",
                "period_avg_volume_out",
                "period_unique_counterparties",
                "period_active_days"
            ]

            if all(col in st.session_state.original_data.columns for col in original_numeric_cols_for_stats):
                temp_df_for_stats = st.session_state.original_data[original_numeric_cols_for_stats].copy()

                if len(temp_df_for_stats) == len(st.session_state.processed_data):
                    temp_df_for_stats['cluster'] = st.session_state.processed_data[
                        'cluster'].values

                    stats = temp_df_for_stats.groupby('cluster')[original_numeric_cols_for_stats].describe()

                    stats_to_keep = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
                    filtered_stats = stats.loc[:, (slice(None), stats_to_keep)]

                    st.dataframe(filtered_stats)  # Выводим отфильтрованный DataFrame

                else:
                    st.error(
                        "Ошибка: Не совпадает количество строк между оригинальными и обработанными данными. Невозможно рассчитать статистику по оригинальным данным."
                    )

            else:
                st.error("Ошибка: Исходные данные не содержат всех необходимых колонок для расчета статистики.")
        else:
            st.info("Статистика по оригинальным данным будет доступна после завершения кластеризации.")
        # Распределение по кластерам
        st.subheader("Распределение адресов по кластерам")
        st.bar_chart(st.session_state.processed_data['cluster'].value_counts())

        # Кнопка сброса
        if st.button("Сбросить анализ"):
            st.session_state.clear()
            st.rerun()

else:
    st.info("Загрузите CSV файл для начала анализа")
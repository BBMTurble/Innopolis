import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import scipy.stats as stats

def main():
    st.title("Промежуточная аттестация")

    file = st.file_uploader("Загрузите датасет", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)

        st.header("Выберите две колонки:")
        col1 = st.selectbox("Первая колонка", df.columns.tolist())
        col2 = st.selectbox("Вторая колонка", df.columns.tolist())

        st.subheader(f"Гистограмма распределения '{col1}'")
        if df[col1].dtype == "object":
            plot_pie_chart(df, col1)
        else:
            plot_histogram(df, col1)

        st.subheader(f"Гистограмма распределения '{col2}'")
        if df[col2].dtype == "object":
            plot_pie_chart(df, col2)
        else:
            plot_histogram(df, col2)

        st.header("Выберите алгоритм теста гипотез:")
        tests = ['t-test', 'Mann Whitney U-test', 'Chi-square test']
        selected_test = st.selectbox("Выберите алгоритм", tests)

        if selected_test:
            st.subheader(f"Результаты проверки гипотезы с помощью '{selected_test}':")
            result = hypothesis_test(df, col1, col2, selected_test)
            st.write(result)

def plot_histogram(df, column):
    fig = px.histogram(df, x=column, nbins=20, title=f"Распределение переменной '{column}'")
    st.plotly_chart(fig)

def plot_pie_chart(df, column):
    value_counts = df[column].value_counts()
    fig = px.pie(value_counts, names=value_counts.index, title=f"Распределение переменной '{column}'")
    st.plotly_chart(fig)

def hypothesis_test(df, col1, col2, test):
    data1 = df[col1]
    data2 = df[col2]

    if test == 't-test':
        statistic, pvalue = stats.ttest_ind(data1, data2)
        result = f"Статистика теста: {statistic:.4f}\n" \
                 f"\np-value: {pvalue:.4f}\n"
    elif test == 'Mann Whitney U-test':
        statistic, pvalue = stats.mannwhitneyu(data1, data2)
        result = f"Статистика теста: {statistic:.4f}\n" \
                 f"\np-value: {pvalue:.4f}\n"
    elif test == 'Chi-square test':
        contingency_table = pd.crosstab(data1, data2)
        statistic, pvalue, df, expected = stats.chi2_contingency(contingency_table)
        result = f"Статистика теста: {statistic:.4f}\n" \
                 f"\np-value: {pvalue:.4f}\n"

    return result

if __name__ == "__main__":
    main()
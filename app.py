import streamlit as st
from analysis_and_model import analysis_and_model_page
from presentation import presentation_page

# Настройка навигации
pages = {
    "📊 Анализ и модель": analysis_and_model_page,
    "📝 Презентация": presentation_page,
}

# Отображение навигации
st.sidebar.title("Навигация")
selection = st.sidebar.radio("", list(pages.keys()))
page = pages[selection]
page()
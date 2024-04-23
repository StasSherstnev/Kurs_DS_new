import streamlit as st
import plotly.express as px
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Предсказание цен на квартиры в г. Москва", page_icon=":city_sunset:",layout="wide")

st.header(":city_sunrise: Предсказание цен на квартиры в г. Москва")
st.markdown('<style>div.block-container{padding-top:2rem;}</style>',unsafe_allow_html=True)

os.chdir(r'D:\Project\Kurs_DS\Kurs_DS_new\Streamlit\Data')
df = pd.read_csv("train.csv")
st.write(df[:5])

df = df[(df['num_room']!=19) & (df['num_room']!=17) & (df['num_room']!=10) & (df['num_room']!=0) & (df['num_room'].notna())]
df['num_room'] = df['num_room'].astype(int)
df['kitch_sq'] = df['kitch_sq'].fillna(0)
df = df[df['kitch_sq'] < 40]
df['kitch_sq'] = df['kitch_sq'].astype(int)

st.sidebar.subheader('Параметры для предсказания')
st.sidebar.selectbox('Комнаты', df['num_room'].drop_duplicates().sort_values())
st.sidebar.slider('Общая площадь',
                  min_value=min(df['full_sq']),
                  max_value=max(df['full_sq']))
st.sidebar.slider('Площадь кухни',
                  min_value=min(df['kitch_sq']),
                  max_value=max(df['kitch_sq']))

button = st.button('предсказать')
if button:
    st.write(df['price_doc'].mean())


import streamlit as st
import plotly.express as px
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Предсказание цен на квартиры в г. Москва", page_icon=":city_sunset:",layout="wide")

st.header(":city_sunrise: Предсказание цен на квартиры в г. Москва")
st.markdown('<style>div.block-container{padding-top:2rem;}</style>',unsafe_allow_html=True)

df = pd.read_csv("Data/train.csv")

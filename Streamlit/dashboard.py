import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import os
import warnings
# модели
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
# Метрики
from sklearn.metrics import mean_squared_log_error, r2_score, root_mean_squared_log_error, mean_squared_error
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Предсказание цен на квартиры в г. Москва", page_icon=":city_sunset:",layout="wide")

st.header(":city_sunrise: Предсказание цен на квартиры в г. Москва")
st.markdown('<style>div.block-container{padding-top:2rem;}</style>',unsafe_allow_html=True)

os.chdir(r'D:\Project\Kurs_DS\Kurs_DS_new\Streamlit\Data')
data_train = pd.read_csv("train.csv")

data_train['num_room'] = data_train['num_room'].fillna(0)
data_train['num_room'] = data_train['num_room'].astype(int)
data_train = data_train[(data_train['num_room']!=10) & (data_train['num_room']!=17) & (data_train['num_room']!=19)]
data_train = data_train[data_train['full_sq']!=0]
# посчитаем средний процент комнат, чтобы в дальнейшем заполнить пропущенные значения в комнатах
data_train_d = data_train[data_train['num_room']>0]
d = data_train_d['num_room']/data_train_d['full_sq']
r_d = d.mean()
data_train['num_room'] = data_train['num_room'].replace(0, np.nan)
data_train['num_room'] = round(data_train['num_room'].fillna(data_train['full_sq'] * r_d)).astype(int)
data_train = data_train[data_train['num_room']<9]
data_train.dropna(subset=['floor'],inplace=True)
data_train['max_floor'] = data_train['max_floor'].fillna(0)
data_train.loc[data_train['floor']>data_train['max_floor'], 'max_floor'] = data_train['floor']
data_train = data_train[data_train['floor']<data_train['max_floor']]
data_train['max_floor'] = data_train['max_floor'].astype(int)
data_train['1_sq'] = data_train['price_doc']/data_train['full_sq']
data_train = data_train[data_train['full_sq']<250]
data_train = data_train[data_train['1_sq']<500000]
data_train['kitch_sq'] = data_train['kitch_sq'].fillna(0)
# посчитаем средний процент от общей площади, чтобы в дальнейшем заполнить пропущенные значения в кухне
data_train_k = data_train[data_train['kitch_sq']>0]
d = data_train_k['kitch_sq']/data_train_k['full_sq']
k_d = d.mean()
data_train['kitch_sq'] = data_train['kitch_sq'].replace(0, np.nan)
data_train['kitch_sq'] = round(data_train['kitch_sq'].fillna(data_train['full_sq'] * k_d))
data_train = data_train[data_train['kitch_sq']<data_train['full_sq']]
data_train = data_train[(data_train['kitch_sq']<50)]
data_obj = data_train.select_dtypes('object')
data_train['product_type'] = data_train['product_type'].replace(['Investment','OwnerOccupier'],[1,2])
# рассмотрим столбец sub_area, это столбец с микрорайонами
d = data_train['sub_area'].value_counts()
d_sa = d.to_frame().reset_index().drop(['count'], axis=1)
# создадаим столбец с числовым представлением района
d_sa['sub_area_num'] = np.arange(d_sa.shape[0])+1
# добавляем столбец с числовым значением
data_train = pd.merge(data_train,d_sa,on='sub_area')
# рассмотрим столбец thermal_power_plant_raion, это столбец - Наличие тепловой электростанции в округе
d = data_train['thermal_power_plant_raion'].value_counts()
d_sa = d.to_frame().reset_index().drop(['count'], axis=1)
# создадаим столбец с числовым представлением
d_sa['thermal_power_plant_raion_num'] = np.arange(d_sa.shape[0])+1
data_train = pd.merge(data_train,d_sa,on='thermal_power_plant_raion')
# рассмотрим столбец - incineration_raion	Наличие мусоросжигательных заводов
d = data_train['incineration_raion'].value_counts()
d_sa = d.to_frame().reset_index().drop(['count'], axis=1)
# создадаим столбец с числовым представлением
d_sa['incineration_raion_num'] = np.arange(d_sa.shape[0])+1
data_train = pd.merge(data_train,d_sa,on='incineration_raion')
# рассмотрим столбец - oil_chemistry_raion Наличие грязных производств
d = data_train['oil_chemistry_raion'].value_counts()
d_sa = d.to_frame().reset_index().drop(['count'], axis=1)
# создадаим столбец с числовым представлением
d_sa['oil_chemistry_raion_num'] = np.arange(d_sa.shape[0])+1
data_train = pd.merge(data_train,d_sa,on='oil_chemistry_raion')
# рассмотрим столбец - radiation_raion	Наличие мест захоронения радиоактивных отходов
d = data_train['radiation_raion'].value_counts()
d_sa = d.to_frame().reset_index().drop(['count'], axis=1)
# создадаим столбец с числовым представлением
d_sa['radiation_raion_num'] = np.arange(d_sa.shape[0])+1
data_train = pd.merge(data_train,d_sa,on='radiation_raion')
# рассмотрим столбец - railroad_terminal_raion	Наличие железнодорожного терминала в районе
d = data_train['railroad_terminal_raion'].value_counts()
d_sa = d.to_frame().reset_index().drop(['count'], axis=1)
# создадаим столбец с числовым представлением
d_sa['railroad_terminal_raion_num'] = np.arange(d_sa.shape[0])+1
data_train = pd.merge(data_train,d_sa,on='railroad_terminal_raion')
# рассмотрим столбец - big_market_raion	Наличие крупных продуктовых / оптовых рынков
d = data_train['big_market_raion'].value_counts()
d_sa = d.to_frame().reset_index().drop(['count'], axis=1)
# создадаим столбец с числовым представлением
d_sa['big_market_raion_num'] = np.arange(d_sa.shape[0])+1
data_train = pd.merge(data_train,d_sa,on='big_market_raion')
# рассмотрим столбец - nuclear_reactor_raion	Наличие действующих ядерных реакторов
d = data_train['nuclear_reactor_raion'].value_counts()
d_sa = d.to_frame().reset_index().drop(['count'], axis=1)
# создадаим столбец с числовым представлением
d_sa['nuclear_reactor_raion_num'] = np.arange(d_sa.shape[0])+1
data_train = pd.merge(data_train,d_sa,on='nuclear_reactor_raion')
# рассмотрим столбец - detention_facility_raion	Наличие центров содержания под стражей
d = data_train['detention_facility_raion'].value_counts()
d_sa = d.to_frame().reset_index().drop(['count'], axis=1)
# создадаим столбец с числовым представлением
d_sa['detention_facility_raion_num'] = np.arange(d_sa.shape[0])+1
data_train = pd.merge(data_train,d_sa,on='detention_facility_raion')
# рассмотрим столбец - water_1line	Первая линия от реки (150 м)
d = data_train['water_1line'].value_counts()
d_sa = d.to_frame().reset_index().drop(['count'], axis=1)
# создадаим столбец с числовым представлением
d_sa['water_1line_num'] = np.arange(d_sa.shape[0])+1
data_train = pd.merge(data_train,d_sa,on='water_1line')
# рассмотрим столбец - big_road1_1line	Первая линия от дороги (100 м для скоростных автомагистралей)
d = data_train['big_road1_1line'].value_counts()
d_sa = d.to_frame().reset_index().drop(['count'], axis=1)
# создадаим столбец с числовым представлением
d_sa['big_road1_1line_num'] = np.arange(d_sa.shape[0])+1
data_train = pd.merge(data_train,d_sa,on='big_road1_1line')
# рассмотрим столбец - railroad_1line	Первая линия от железной дороги (100 м)
d = data_train['railroad_1line'].value_counts()
d_sa = d.to_frame().reset_index().drop(['count'], axis=1)
# создадаим столбец с числовым представлением
d_sa['railroad_1line_num'] = np.arange(d_sa.shape[0])+1
data_train = pd.merge(data_train,d_sa,on='railroad_1line')
# рассмотрим столбец - ecology	Экологическая зона, в которой расположен дом
d = data_train['ecology'].value_counts()
d_sa = d.to_frame().reset_index().drop(['count'], axis=1)
# создадаим столбец с числовым представлением
d_sa['ecology_num'] = np.arange(d_sa.shape[0])+1
data_train = pd.merge(data_train,d_sa,on='ecology')
# удаление текстовых столбцов, их значения в новых столбцах с числовыми значениями
data_train = data_train.drop(['timestamp',
                      'sub_area',
                      'culture_objects_top_25',
                      'thermal_power_plant_raion',
                      'incineration_raion',
                      'oil_chemistry_raion',
                      'radiation_raion',
                      'railroad_terminal_raion',
                      'big_market_raion',
                      'nuclear_reactor_raion',
                      'detention_facility_raion',
                      'water_1line',
                      'big_road1_1line',
                      'railroad_1line',
                      'ecology'], axis=1)
# подготовка к обучению
# загрузка правильных ответов
data_y = pd.DataFrame()
data_y['price_doc'] = data_train['price_doc']
# формирование тренировочных данных
data_X = data_train
# подсчет сколько пропущенных значений в столбцах
d_null = data_X.isnull().sum()[data_X.isnull().sum()>0]
d_null = d_null.to_frame(name='col').reset_index().sort_values('col',ascending=True)
# посмотрим какие столбцы с пустыми значениями более 10000
d_ncol = d_null[d_null['col']>10000]
# в столбце build_year пропущено значений почти половино, будет целесообразно удалить его для обучения
# удаляем столбцы с пустыми значениями больше 10000
for i in d_ncol['index']:
     data_X = data_X.drop(i,axis=1)
# заменим пустые значения на среднее
data_X = data_X.apply(lambda x: x.fillna(x.mean()) if x.isna().any() else x)
# подсчет сколько нулевых значений в столбцах
d_nnull = data_X[data_X == 0].count()[data_X[data_X == 0].count()>0]
d_nnull = d_nnull.to_frame(name='col').reset_index().sort_values('col',ascending=True)
# посмотрим какие столбцы с нулевыми значениями более 10000
d_col = d_nnull[d_nnull['col']>10000]
# удаляем столбцы с нулями больше половины строчек
for i in d_col['index']:
     data_X = data_X.drop(i,axis=1)
# разделяем на обучающую выборку и тестовую
# предварительно убрать правильные ответы
data_X = data_X.drop('price_doc', axis=1)
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.3, shuffle=False, random_state=42)

# обучаем модель с помощью дерева решений
clf = DecisionTreeRegressor(max_depth=7,min_samples_leaf=15)
clf.fit(X_train, y_train)

# получаем предсказания
predicted = clf.predict(X_test)

#df = df[(df['num_room']!=19) & (df['num_room']!=17) & (df['num_room']!=10) & (df['num_room']!=0) & (df['num_room'].notna())]
#df['num_room'] = df['num_room'].astype(int)
#df['kitch_sq'] = df['kitch_sq'].fillna(0)
#df = df[df['kitch_sq'] < 40]
#df['kitch_sq'] = df['kitch_sq'].astype(int)

st.write(data_X[:5])

st.sidebar.subheader('Параметры для предсказания')
st.sidebar.selectbox('Количество комнат', data_X['num_room'].drop_duplicates().sort_values())
st.sidebar.slider('Общая площадь',
                  min_value=min(data_X['full_sq']),
                  max_value=max(data_X['full_sq']))
st.sidebar.slider('Площадь кухни',
                  min_value=min(data_X['kitch_sq']),
                  max_value=max(data_X['kitch_sq']))
st.sidebar.selectbox('Этаж', data_X['floor'].drop_duplicates().sort_values())


st.write('RMSLE: ', mean_squared_log_error(y_test,predicted, squared=False))
st.write('r2', r2_score(y_test, predicted))
st.write('MSE', mean_squared_error(y_test,predicted))



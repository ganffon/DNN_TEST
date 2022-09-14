#-*- coding: utf-8 -*-
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import streamlit as st
import openpyxl
import matplotlib.pyplot as plt
# python -m pip install -U matplotlib
import seaborn as sns

np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.8f}".format(x)})
st.title("DNN 개발 페이지")
def scale_datasets(x_train, x_test):
  standard_scaler = MinMaxScaler()
  x_train_scaled = standard_scaler.fit_transform(x_train)
  x_test_scaled = standard_scaler.transform(x_test)

  return x_train_scaled, x_test_scaled

def rescale_datasets(original_data, predictions):
    reverse_scaler = MinMaxScaler()
    reverse_scaler.fit(original_data)

    rescale_data = reverse_scaler.inverse_transform(predictions)

    return rescale_data

def learning():
    try:
        ###########--DataSet 기본정보 : df.info()
        ###########--DataSet 통계정보요약 : df.describe()
        ###########--DataSet 결측값 확인 : df.isnull()  // 결측값 총 갯수 : df.isnull().sum()
        ###########--DataSet 중복 확인 : df.duplicated() // 중복값 총 갯수 : df.duplicated().sum() // 중복값 제거 : df.drop_duplicated()
        ###########--DataSet 상관관계분석 : df.corr()
        df = pd.read_excel('..\\Data\\file\\hns.xlsx',
                           header=0,
                           index_col=None,
                           skipfooter=0,
                           engine='openpyxl',
                           dtype=np.float32)
        print(df)
        ds = df.drop(columns=['stiky', 'regist'])
        print(ds)

        dss = df[['stiky', 'regist']]

        st.sidebar.markdown("# 선택하시라요")
        if st.sidebar.checkbox("Raw Data Show"):
            st.markdown("### 01. X Data ( Raw )")
            st.dataframe(ds)
            st.line_chart(ds)
            st.markdown("### 02. Y Data ( Raw )")
            st.dataframe(dss)
            st.line_chart(dss)

        x_train = ds
        y_train = dss

        # x_train = x_train.drop('TEST', axis=1)
        x_train, x_test = train_test_split(x_train, test_size=0.2, random_state = 777)
        y_train, y_test = train_test_split(y_train, test_size=0.2, random_state=777)

        # x_train_scaled, x_test_scaled = scale_datasets(x_train, x_test)
        # y_train_scaled, y_test_scaled = scale_datasets(y_train, y_test)

        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()

        scaler_x_fit = scaler_x.fit(x_train)
        x_train_scaled = scaler_x_fit.transform(x_train)
        x_test_scaled = scaler_x_fit.transform(x_test)

        scaler_y_fit = scaler_y.fit(y_train)
        y_train_scaled = scaler_y_fit.transform(y_train)
        y_test_scaled = scaler_y_fit.transform(y_test)

        if st.sidebar.checkbox("Scale Data Show"):
            st.markdown("### 03. X Data ( Scale )")
            st.dataframe(x_train_scaled)
            st.markdown("### 04. Y Data ( Scale )")
            st.dataframe(y_train_scaled)



        model = Sequential()
        model.add(Dense(units=10, input_shape=(1,16)))
        model.add(Dense(units=8, activation='swish'))
        model.add(Dense(units=4, activation='swish'))
        model.add(Dense(units=2, activation='softmax'))
        model.compile(optimizer='sgd', loss='mse')
        modelsum = model.summary()
        model.fit(x_train_scaled, y_train_scaled, epochs=10)
        print(x_test)
        print("**" * 20)

        df = pd.read_excel('..\\Data\\file\\hns_test_label.xlsx',
                           header=0,
                           index_col=None,
                           skipfooter=0,
                           engine='openpyxl',
                           dtype=np.float32)
        ds = df.drop(columns=['stiky', 'regist'])
        x_train = ds
        print(x_train)
        # x_train_scaled, x_test_scaled = scale_datasets(x_train, x_test)
        x_train_scaled = scaler_x_fit.transform(x_train)

        prey = model.predict(x_train_scaled)
        print(prey)
        print("**"*20+"dddddd")
        repre = scaler_y.inverse_transform(prey)
        print(repre)

        if st.sidebar.checkbox("Predict Data Show"):
            st.markdown("### 05. X Data ( Predict Raw )")
            st.dataframe(x_train)
            st.markdown("### 06. X Data ( Predict Scale )")
            st.dataframe(x_train_scaled)
            st.markdown("### 07. Y Data ( Predict Scale )")
            st.dataframe(prey)
            st.markdown("### 08. Y Data ( Predict Inverse )")
            st.dataframe(repre)

    except Exception as e:
        print(e)

i=1
if i==1:
    learning()
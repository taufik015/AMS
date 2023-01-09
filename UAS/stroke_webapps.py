import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pickle
from sklearn.naive_bayes import GaussianNB

st.write("""
# Klasifikasi Stroke (Web Apps)
Aplikasi berbasis web untuk memprediksi Seseorang Terkena Pnyakit Stroke
""")

img = Image.open('brain.jpg')

st.image(img, use_column_width = False)



st.sidebar.header('Parameter Inputan')

# Upload File csv untuk parameter inputan

upload_file = st.sidebar.file_uploader("Upload file CSV anda",type=["csv"])
if upload_file is not None:
    inputan = pd.read_csv(upload_file)
else:
    def input_user():
        age = st.sidebar.slider('age', 3, 80, 50)
        Residence_type = st.sidebar.selectbox('Residence_type',('Urban', 'Rural'))
        avg_glucose_level = st.sidebar.slider('Glcose Level',60, 223, 100)
        gender = st.sidebar.selectbox('Gender',('Male', 'Female'))
        data = {'age' : age,
                'Residence_type' : Residence_type,
                'gender' : gender,
                'avg_glucose_level' : avg_glucose_level}
        fitur =pd.DataFrame(data, index=[0])
        return fitur
    inputan = input_user()
    
stroke_raw = pd.read_csv('stroke.csv')
stroke = stroke_raw.drop(columns=['stroke'])
df = pd.concat([inputan, stroke], axis=0)

encode = ['Residence_type', 'gender']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1) 
    del df[col]
df = df[:1]

st.subheader('Parameter Inputan')

if upload_file is not None:
    st.write(df)
else:
    st.write('Menunggu file csv untuk diupload. saat ini memakai sampel inputan (seperti tampilan dibawah ini)')
    st.write(df)

load_model = pickle.load(open('modelNBC_stroke.pkl', 'rb'))

prediksi = load_model.predict(df)
prediksi_proba = load_model.predict_proba(df)

st.subheader('Keterangan Label Kelas')
status_stroke = np.array(['yes','No'])
st.write(status_stroke)

st.subheader('Hasil prediksi (Klasifikasi Stroke)')
st.write(status_stroke[prediksi])

st.subheader('Probabilitas Hasil prediksi (Klasifikasi Stroke)')
st.write(prediksi_proba)
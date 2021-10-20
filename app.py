import pickle
import streamlit as st
import pandas as pd


def cancer_pred():
    radius = st.number_input(label='Radius',value=5.00, step=0.01, format="%.2f")
    texture = st.number_input(label='Texture', value=8.00, step=0.01, format="%.2f")
    perimeter = st.number_input(label='Perimeter', value=40.00, step=0.01, format="%.2f")
    area = st.number_input(label='Area', value=140.00, step=0.01, format="%.2f")
    smooth = st.number_input(label='Smoothness', value=0.05, step=0.01, format="%.2f")

    cancer_pred_data = {
        'mean_radius': radius,
        'mean_texture': texture,
        'mean_perimeter': perimeter,
        'mean_area': area,
        'mean_smoothness': smooth
    }
    df2 = pd.DataFrame(cancer_pred_data, index=[0])
    return df2


model = pickle.load(open('Breast_Cancer_Prediction_Model.sav', 'rb'))
df = pd.read_csv('Breast_cancer_data.csv')
st.set_page_config(page_title='Breast Cancer Detection', page_icon="./icon.png")
st.title('Breast Cancer Detection')
st.subheader('Model detects whether a cancer is Malignant or Benign')
st.image('./img.png')
pred_data = cancer_pred()
prediction = model.predict(pred_data)

if prediction == 1:
    st.header("Sorry, It's a Malignant Cancer.")
else:
    st.header("Yay! It's a Benign Cancer.")

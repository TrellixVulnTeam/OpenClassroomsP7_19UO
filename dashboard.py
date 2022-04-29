# APP STREAMLIT : (commande : streamlit run XX/dashboard.py depuis le dossier python)
import streamlit as st
import numpy as np
import pandas as pd
import time
import requests
from urllib.request import urlopen
import ast
import matplotlib.pyplot as plt
import seaborn as sns


# Load Dataframe
path_df_red_pred = 'df_red_pred.csv'
path_df_red_train = 'df_red_train.csv'
path_df_pred_display =  'df_pred_display.csv'


@st.cache  # mise en cache de la fonction pour exécution unique
def chargement_data(path):
    dataframe = pd.read_csv(path)
    return dataframe


@st.cache  # mise en cache de la fonction pour exécution unique
def chargement_explanation(id_input, dataframe, model, sample):
    return interpretation(str(id_input),
                          dataframe,
                          model,
                          sample=sample)

def graph(dataframe,feature,id,df) :
    fig = plt.figure(figsize=(10, 4))
    sns.histplot(data = dataframe , x = feature, hue = 'TARGET', stat = 'density', kde=True, common_norm=False)
    plt.axvline(df[df['SK_ID_CURR']==id][feature].item(),0,2, color ='black', label='client')
    return st.pyplot(fig)


def request(API_url, data) :
    request = requests.get(API_url+"?data="+data)
    return request.json()

def clean(data) : 
    del data[0][0]
    data = str(data)
    data = data.replace("[[", "(")
    data = data.replace("]]", ")")
    data = data.replace(" ", "")
    return data

df = chargement_data(path_df_red_pred)
df_train = chargement_data(path_df_red_train)

liste_id = df['SK_ID_CURR'].tolist()

# affichage formulaire
st.title('Dashboard Scoring Credit')
st.subheader("Prédictions de scoring client et comparaison à l'ensemble des clients")

def main():
    id = st.selectbox('Veuillez saisir l\'identifiant d\'un client:', liste_id)

    API_url = "http://127.0.0.1:8000/predict/"

    predict_btn = st.radio('Prédire')
    if predict_btn:
        data = df[df['SK_ID_CURR']==id].to_numpy().tolist()
        data = clean(data)
        prediction = request(API_url,data)
        if prediction['prediction'] == 0 :
            st.write('Dossier validé par la banque')
        st.write('Probabilité de remboursement :',1- list(prediction['proba'].values())[0])
        details_btn = st.button('Voir caractéristiques du client vs autres clients')
        if details_btn:
            graph(df_train,'EXT_SOURCE_3',id,df)
            graph(df_train,'EXT_SOURCE_2',id,df)
            graph(df_train,'DAYS_EMPLOYED',id,df)


if __name__ == '__main__':

    main()






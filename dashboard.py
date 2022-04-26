# APP STREAMLIT : (commande : streamlit run XX/dashboard.py depuis le dossier python)
import streamlit as st
import numpy as np
import pandas as pd
import time
from urllib.request import urlopen
import json
from urllib.request import urlopen
import json
import requests
from urllib.request import urlopen
import ast

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

def request_prediction(API_url, data):
    headers = {"Content-Type": "application/json"}

    data_json = {'data': data}
    response = requests.request(
        method='GET', headers=headers, url=API_url, json=data_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response

def request(API_url, data) :
    request = requests.post(API_url+"?data="+data)
    return request.json()

df = chargement_data(path_df_red_pred)

liste_id = df['SK_ID_CURR'].tolist()

# affichage formulaire
st.title('Dashboard Scoring Credit')
st.subheader("Prédictions de scoring client et comparaison à l'ensemble des clients")

def main():
    id = st.selectbox('Veuillez saisir l\'identifiant d\'un client:', liste_id)

    API_url = "http://127.0.0.1:8000/predict/"

    predict_btn = st.button('Prédire')
    if predict_btn:
        data = df[df['SK_ID_CURR']==id].to_numpy().tolist()
        st.write(int(id))
        del data[0][0]
        data = str(data)
        data = data.replace("[[", "(")
        data = data.replace("]]", ")")
        data = data.replace(" ", "")
        pred = request(API_url,data)
        st.write(pred)
        st.write(
            'Le prix médian d\'une habitation est de {:.2f}'.format(pred))
        st.write(pred['prediction'])


if __name__ == '__main__':
    main()
    '''
    st.subheader("Caractéristiques influençant le score")

    # affichage de l'explication du score
    with st.spinner('Chargement des détails de la prédiction...'):
        explanation = chargement_explanation(str(id_input),
                                             dataframe,
                                             StackedClassifier(),
                                             sample=False)
    # st.success('Done!')

    # Affichage des graphes
    graphes_streamlit(explanation)

    st.subheader("Définition des groupes")
    st.markdown("\
    \n\
    * Client : la valeur pour le client considéré\n\
    * Moyenne : valeur moyenne pour l'ensemble des clients\n\
    * En Règle : valeur moyenne pour l'ensemble des clients en règle\n\
    * En Défaut : valeur moyenne pour l'ensemble des clients en défaut\n\
    * Similaires : valeur moyenne pour les 20 clients les plus proches du client\
    considéré sur les critères sexe/âge/revenu/durée/montant du crédit\n\n\
    ")

    # Affichage du dataframe d'explicabilité
    # st.write(explanation)

    # Détail des explications
    st.subheader('Traduction des explication')
    chaine_explanation, df_explanation = df_explain(explanation)
    chaine_features = '\n\
    '
    for x, y in zip(df_explanation['Feature'], df_explanation['Nom francais']):
        chaine_features += '* **' + str(x) + ' ** ' + str(y) + '\n' \
                                                               ''
    st.markdown(chaine_features)

    # st.write(df_explanation, unsafe_allow_html=True)

    # Modifier le profil client en modifiant une valeur
    # st.subheader('Modifier le profil client')
    st.sidebar.header("Modifier le profil client")
    st.sidebar.markdown(
        'Cette section permet de modifier une des valeurs les plus caractéristiques du client et de recalculer son score')
    features = explanation['feature'].values.tolist()
    liste_features = tuple([''] + features)
    feature_to_update = ''
    feature_to_update = st.sidebar.selectbox('Quelle caractéristique souhaitez vous modifier', liste_features)

    # st.write(dataframe.head())

    if feature_to_update != '':
        value_min = dataframe[feature_to_update].min()
        value_max = dataframe[feature_to_update].max()
        # st.write(list(explanation['feature'].values))
        # st.write(explanation['feature'].values[0])
        default_value = explanation[explanation['feature'] == feature_to_update]['customer_values'].values[0]
        # st.write(default_value)

        min_value = float(dataframe[feature_to_update].min())
        max_value = float(dataframe[feature_to_update].max())

        if (min_value, max_value) == (0, 1):
            step = float(1)
        else:
            step = float((max_value - min_value) / 20)

        update_val = st.sidebar.slider(label='Nouvelle valeur (valeur d\'origine : ' + str(default_value)[:4] + ')',
                                       min_value=min_value,
                                       max_value=max_value,
                                       value=default_value,
                                       step=step)

        if update_val != default_value:
            time.sleep(0.5)
            update_predict, proba_update = predict_update(id_input, dataframe, feature_to_update, update_val)
            if update_predict == 1:
                etat = 'client à risque'
            else:
                etat = 'client peu risqué'
            chaine = 'Nouvelle prédiction : **' + etat + '** avec **' + str(
                round((proba_update[0][1]) * 100)) + '%** de risque de défaut (classe réelle : ' + str(
                classe_reelle) + ')'
            st.sidebar.markdown(chaine)

    st.subheader('Informations relatives au client')
    df_client = chargement_ligne_data(id_input, dataframe).T
    df_client['nom_fr'] = [correspondance_feature(feature) for feature in df_client.index]
    st.write(df_client)

'''





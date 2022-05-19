# APP STREAMLIT : (commande : streamlit run XX/dashboard.py depuis le dossier python)
import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import SessionState


# Load Dataframe
path_df_red_pred = 'df_red_pred.csv'
path_df_red_train = 'df_red_train.csv'
path_df_pred_display =  'df_pred_display.csv'


@st.cache(allow_output_mutation=True)  # mise en cache de la fonction pour exécution unique
def chargement_data(path):
    dataframe = pd.read_csv(path)
    return dataframe


def graph(dataframe,feature,id,df) :
    fig = plt.figure(figsize=(10, 4))
    sns.histplot(data = dataframe , x = feature, hue = 'TARGET', stat = 'density', kde=True, common_norm=False, legend=True)
    plt.axvline(df[df['SK_ID_CURR']==id][feature].item(),0,2, color ='black', label='client')
    st.write("Valeur de la feature",feature,"pour le client :", df[df['SK_ID_CURR']==id][feature].item())
    if feature=='AMT_INCOME_TOTAL' :
        plt.xlim(0,0.5E6)
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



#chargement des différents dataframe

df_to_predict = chargement_data(path_df_red_pred)
df_train = chargement_data(path_df_red_train)
df_to_predict_display = chargement_data(path_df_pred_display)

df_train['TARGET'] = df_train['TARGET'].astype(int)

liste_id = df_to_predict['SK_ID_CURR'].tolist()

# affichage formulaire
st.title('Dashboard Scoring Credit')
st.subheader("Prédiction de scoring client et comparaison à l'ensemble des clients")


#aide à la compréhension des graphs
comment = """ 
TARGET 0 : client avec dossier validé

TARGET 1 : client avec dossier non validé

Trait vertical noir représente le client {0}
"""




def main():
    id = st.selectbox('Veuillez choisir l\'identifiant d\'un client:', liste_id)
    API_url = "https://apiopenclassrooms.herokuapp.com/predict/"
    ss = SessionState.get(predict_btn=False)
    predict_btn = st.empty()
    if predict_btn.button('Prédiction') :
        ss.predict_btn = True
    if ss.predict_btn:
        data = df_to_predict[df_to_predict['SK_ID_CURR']==id].to_numpy().tolist()
        data = clean(data)
        prediction = request(API_url,data)
        if prediction['prediction'] == 0 :
            st.write('Dossier validé par la banque')
        refund = (1- list(prediction['proba'].values())[0])*100
        st.write('Probabilité de remboursement :',int(refund),'%')
        st.progress(int(refund))
        ss = SessionState.get(details_btn=False)
        details_btn = st.empty()
        if details_btn.button('Client vs autres clients') :
            ss.details_btn = True
        try :
            if ss.details_btn :
                client_infos = st.multiselect("Filtre infos client:", ['EXT_SOURCE', 'AMT', 'OTHERS'],
                                              default=['EXT_SOURCE'])
                st.write(comment.format(id))

                if 'EXT_SOURCE' in client_infos :
                    graph(df_train,'EXT_SOURCE_3',id,df_to_predict_display)
                    st.latex(r'''
                    \underline{Courbes représentant les distributions des revenus ext\érieurs de type 3 ({EXT}_{SOURCE}_{3}) 
                             pour les dossiers validés (courbe bleue) et non validés (courbe orange)}''')
                    graph(df_train,'EXT_SOURCE_2',id,df_to_predict_display)
                    st.latex(r'''
                         \underline{a} + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
                         \sum_{k=0}^{n-1} ar^k =
                         a \left(\frac{1-r^{n}}{1-r}\right)
                         ''')
                    st.write('Probabilité de remboursement :', int(refund), '%')
                if 'AMT' in client_infos :
                    graph(df_train,'AMT_ANNUITY',id,df_to_predict_display)
                    graph(df_train,'AMT_CREDIT',id,df_to_predict_display)
                    graph(df_train,'AMT_INCOME_TOTAL',id,df_to_predict_display)
                    graph(df_train,'AMT_GOODS_PRICE',id,df_to_predict_display)
                if 'OTHERS' in client_infos :
                    graph(df_train, 'PAYMENT_RATE', id, df_to_predict_display)
                    graph(df_train, 'DAYS_EMPLOYED', id, df_to_predict_display)
                    graph(df_train, 'DAYS_BIRTH', id, df_to_predict_display)
        except :
            pass



if __name__ == '__main__':
    main()






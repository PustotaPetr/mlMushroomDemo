import streamlit as st
import pandas as pd
import numpy as np
import joblib


model = joblib.load("mushroom_model_dump.pkl")

columns = ['cap-shape', 'cap-surface', 'cap-color', 'odor', 'gill-color', 'habitat']
pred_column = 'cap-color'

df = pd.read_csv('mushroom_decode.csv')

st.sidebar.title('select parameters of mushroom:')

dict_values = {}
select_box = []
for col in columns:
    if col != pred_column:
        dict_values[col] = df[col].unique().tolist()
        select_box.append(st.sidebar.selectbox(col, df[col].unique().tolist()))



@st.cache
def make_predict_dataframe():
    df_pred = pd.DataFrame(columns=columns)

    rec_template = {}
    i = 0
    for col in columns:
        if col != pred_column:
            # rand_int = randrange(len(dict_values[col]))
            #print(col, len(dict_values[col]), select_box[i], )
            rec_template[col] = select_box[i]  # dict_values[col][rand_int]
            i += 1

    for var_value in df[pred_column].unique().tolist():
        rec_template.update({pred_column: var_value})
        df_pred = df_pred.append(rec_template, ignore_index=True)
    return df_pred


@st.cache
def make_prediction(df):
    prediction_proba = model.predict_proba(df)[:, 1]
    return prediction_proba

def background_color(val):
    return 'background-color: #ff8888' if val == 'poisonous' else 'background-color: #88dd88' if val == 'edible' else ''


df_pred = make_predict_dataframe()

#st.dataframe(df_pred)

st.title("Simple ML demo - predict mushroom poisonous.")
st.write("Demo based on Mushroom Classification kaggle "
        "dataset (https://www.kaggle.com/uciml/mushroom-classification).")

st.write("This demo allows you to select 5 attributes and make prediction "
         "for all possible variant of last attribute(mushroom cap color). Demonstration also "
         "allows you to select the boundary of the true value that mushroom is poisonous.")

st.write(""
         ""
         "")

slider_val = st.slider('select value of True border', 0.0, 1.0, 0.15, 0.01)

st.write('You select value: ' + str(slider_val))

st.write("Prediction about toxicity mushrooms in depends of mushroom cap color is:")

df_answer = df_pred[pred_column].to_frame()
y_pred = make_prediction(df_pred)
df_answer['poisonous_prob'] = np.round(y_pred, 3)
df_answer['poisonous_pred'] = y_pred > slider_val
df_answer['poisonous'] = df_answer['poisonous_pred'].apply(lambda x: 'poisonous' if x else 'edible')
df_answer = df_answer.set_index('cap-color')
st.dataframe(df_answer[['poisonous_prob', 'poisonous']].T.style.applymap(background_color).set_precision(3))


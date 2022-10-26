import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import streamlit as st
from category_encoders import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge
from PIL import Image


def wrangle(filepath):
    # Read CSV file
    df = pd.read_csv(filepath)

    # Create a new Count Nationality column
    df['Nationality Count'] = df.groupby('Nationality')['Nationality'].transform('count')

    # Only take countries that have more than 300 players
    mask_nation = df['Nationality Count'] > 300
    df = df[mask_nation]

    return df


df = wrangle('Fifa 23 Players Data.csv')

df_nationality_counts = df["Nationality"].value_counts()
df_nationality_counts.plot(kind="barh")
plt.xlabel("Count of players by Nationality")
target = "Wage(in Euro)"
features = ["Nationality"]
y = df[target]
X = df[features]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

y_mean = y_train.mean()
y_pred_baseline = [y_mean] * len(y_train)
print("Mean apt price: ", y_mean)
print("Baseline MAE: ", mean_absolute_error(y_train, y_pred_baseline))

model = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    Ridge()
)
model.fit(X_train, y_train)

# 1. Evaluate out model performance on training data
y_pred_training = model.predict(X_train)
mae_training = mean_absolute_error(y_train, y_pred_training)

# 2. Evaluate out model performance on test data
y_pred_test = pd.Series(model.predict(X_test))
mae_test = mean_absolute_error(y_test, y_pred_test)

intercept = model.named_steps["ridge"].intercept_.astype(float)
coefficients = model.named_steps["ridge"].coef_.astype(float)

feature_names = model.named_steps["onehotencoder"].get_feature_names()

feat_imp = pd.Series(coefficients, index=feature_names)
feat_imp.head()
print(f"price = {intercept.round(2)}")
for f, c in feat_imp.items():
    print(f"+ ({round(c, 2)} * {f})")


def make_prediction(nationality):
    data = {
        "Nationality": nationality
    }
    df_predict = pd.DataFrame(data, index=[0])
    prediction = model.predict(df_predict).round(2)[0]
    df_nation = df[df['Nationality'] == str(nationality)]
    mean_wage = df_nation['Wage(in Euro)'].mean().round(2)

    d = (f"Predicted wage of a player with {nationality} nationality is: {prediction} Euro/ week",

         f"Mean wage of players with {nationality} nationality is: {mean_wage} Euro/ week")

    return d


st.title('Predicting wage of player By Their Nationality')
nation = df['Nationality']
nations = st.selectbox('Nations', nation)
if st.button('Get Prediction By The Nation!'):
    t = (make_prediction(nations))
    st.write(t)
    image = Image.open('download.png')
    st.image(image, caption='Count of players by Nationality')

df1 = pd.read_csv('new.csv')
target1 = "Wage(in Euro)"
features1 = ["Full Name"]
y1 = df1[target1]
X1 = df1[features1]
X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X1, y1, test_size=0.2, random_state=42
)

y_mean1 = y_train1.mean()
y_pred_baseline1 = [y_mean1] * len(y_train1)

model = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    Ridge()
)
model.fit(X_train1, y_train1)

y_pred_training1 = model.predict(X_train1)
mae_training1 = mean_absolute_error(y_train1, y_pred_training1)

y_pred_test1 = pd.Series(model.predict(X_test1))
mae_test1 = mean_absolute_error(y_test1, y_pred_test1)

intercept1 = model.named_steps["ridge"].intercept_.astype(float)
coefficients1 = model.named_steps["ridge"].coef_.astype(float)

feature_names1 = model.named_steps["onehotencoder"].get_feature_names()

feat_imp1 = pd.Series(coefficients1, index=feature_names1)
feat_imp1.head()


def make_prediction1(full_name):
    data = {
        "Full Name": full_name
    }
    df_predict1 = pd.DataFrame(data, index=[0])
    prediction1 = model.predict(df_predict1).round(2)[0]
    df_nation1 = df1[df1['Full Name'] == str(full_name)]
    mean_wage1 = df_nation1['Wage(in Euro)'].mean().round(2)

    r = (f"Predicted wage of a player with {full_name}  is: {prediction1} Euro/ week",
            f"Mean wage of players with {full_name} F is: {mean_wage1} Euro/ week")

    return r


st.subheader('Predicting wage of players By Their Names !')
full_names = df1['Full Name']
full_name = st.selectbox('Full Name', full_names)

if st.button('Get Prediction By The Footballers Name !'):
    t = (make_prediction1(full_name))
    st.write(t)
    image = Image.open('download.png')
    st.image(image, caption='Count of players by Nationality')

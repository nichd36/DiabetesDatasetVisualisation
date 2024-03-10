import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

padding = 20

st.title("Let's clean up the data! 🧼")
st.image(image="/Users/nichdylan/Documents/DVID/Assignment 2/static/giphy.gif")

def load_data():
        df = pd.read_csv("/Users/nichdylan/Documents/DVID/Assignment 2/diabetes.csv")
        return df

df = load_data()

st.header("Here is a preview of the dataset")
df

st.write(df.isnull().sum())
st.markdown("*Luckily, there are no null values here*")

st.header("Using the describe() function, let's take a peek into our dataset")
st.write(df.describe())
st.markdown("*Some of the columns (outside outcome and pregnancies) have a value as low as 0. This shows there are some invalid data.*")
st.markdown("Prove its invalid:")
st.image("/Users/nichdylan/Documents/DVID/Assignment 2/static/chat.jpeg")
st.markdown("*Imagine the blood pressure was really 0* 🩸☠️😵")

st.header("Checking how many 0️⃣s")
exclude_columns = ["Outcome", "Pregnancies"]
zeros_df_filtered = df.drop(columns=exclude_columns).eq(0)
num_zeros_in_each_column = zeros_df_filtered.sum()
st.write("Number of zeros in each column: (*excluding Pregnancies and Outcome*)")
st.write(num_zeros_in_each_column)

st.header("Replacing 0s with median, later on will be used to train models. They were not removed as the dataset is relatively small (768 entries)")
median_excluding_zeros = df.apply(lambda x: x[x != 0].median())
for column in df.columns:
    if column not in ["Pregnancies", "Outcome"]:
        median = median_excluding_zeros[column]
        df[column] = df[column].replace(0, median)
st.write("DataFrame with zeros replaced by column medians (excluding zeros in median calculation):")
df
df.to_csv('diabetes_clean.csv', index=False)
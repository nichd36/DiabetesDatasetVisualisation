import streamlit as st

padding = 20
st.set_page_config(layout="wide")

st.title('Pima Female Indians Diabetes Dataset')

st.sidebar.success("Select a section above.")

st.markdown("By the 1970s, the occurence of type-2 diabetes was about forty percent among Pimas age thirty-five and older. They are a Native American group that resides around Arizona, US and this group was deemed to have a high incidence rate of diabetes mellitus. Hence, research around them was thought to be significant and the Pima Indian Diabetes dataset consisting of Pima Indian females 21 years and older is a popular benchmark dataset.")

import pandas as pd

# Create a DataFrame with one row for Arizona
arizona_data = pd.DataFrame({
        'latitude': [33.0489],
        'longitude': [-111.0937]
})

# Display the map centered on Arizona with a pin marker
st.map(data=arizona_data, zoom=3, color="#ffaa00", size=80000)

def load_data():
        df = pd.read_csv("/Users/nichdylan/Documents/DVID/Assignment 2/diabetes.csv")
        return df

df = load_data()

st.header("Here is a preview of the dataset")
df
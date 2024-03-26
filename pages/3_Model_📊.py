import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import numpy as np

st.title("Can we diagnose whether or not a patient is diabetic?")

def load_clean_data():
        df = pd.read_csv("static/diabetes_clean.csv")
        return df

def load_data():
        df = pd.read_csv("diabetes.csv")
        return df


def intro():
        st.warning("***Please select an option***")

def lstm():
        df = load_data()
        
        X = df.drop(columns=['Outcome'])  # Features
        y = df['Outcome']
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=2)

        model = tf.keras.models.load_model('static/lstm.h5')
        x_test = x_test.iloc[:, :8]

        y_pred = model.predict(x_test)
        y_pred_binary = np.round(y_pred)
        cm = confusion_matrix(y_test, y_pred_binary)
        accuracy = accuracy_score(y_test, y_pred_binary)
        
        x_labels = ['Predicted negative', 'Predicted positive']
        y_labels = ['Actual negative', 'Actual positive']
        
        plt.figure(figsize=(9,9))
        sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
        
        plt.xticks(ticks=[0.5, 1.5], labels=x_labels)
        plt.yticks(ticks=[0.5, 1.5], labels=y_labels)
        
        plt.ylabel('Actual label');
        plt.xlabel('Predicted label');
        
        all_sample_title = 'Accuracy Score: {0}'.format(accuracy)
        plt.title(all_sample_title, size = 15);

        st.pyplot()

def logistic_regression():
        lr = LogisticRegression()
        df = load_data()
        
        X = df.drop(columns=['Outcome'])  # Features
        y = df['Outcome']
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=2)

        lr.fit(x_train, y_train)

        test_score = lr.score(x_test, y_test)
        st.write('Accuracy score : ', test_score)

        predictions = lr.predict(x_test)

        cm = confusion_matrix(y_test, predictions)

        x_labels = ['Predicted negative', 'Predicted positive']
        y_labels = ['Actual negative', 'Actual positive']

        plt.figure(figsize=(9,9))
        sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues_r');

        plt.xticks(ticks=[0.5, 1.5], labels=x_labels)
        plt.yticks(ticks=[0.5, 1.5], labels=y_labels)

        plt.ylabel('Actual label');
        plt.xlabel('Predicted label');

        all_sample_title = 'Accuracy Score: {:.3f}'.format(test_score)
        plt.title(all_sample_title, size = 15);

        st.pyplot()

def support_vector():
        df = load_clean_data()
        
        X = df.drop(columns=['Outcome'])  # Features
        y = df['Outcome']
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=2)

        model = svm.SVC()
        model.fit(x_train, y_train)

        test_score = model.score(x_test, y_test)
        st.write('Accuracy score : ', test_score)

        predictions = model.predict(x_test)
        cm = confusion_matrix(y_test, predictions)
        x_labels = ['Predicted negative', 'Predicted positive']
        y_labels = ['Actual negative', 'Actual positive']

        plt.figure(figsize=(9,9))
        sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues_r');

        plt.xticks(ticks=[0.5, 1.5], labels=x_labels)
        plt.yticks(ticks=[0.5, 1.5], labels=y_labels)

        plt.ylabel('Actual label');
        plt.xlabel('Predicted label');

        all_sample_title = 'Accuracy Score: {:.3f}'.format(test_score)
        plt.title(all_sample_title, size = 15);

        st.pyplot()

def gaussianNB():
        df = load_clean_data()
        
        X = df.drop(columns=['Outcome'])  # Features
        y = df['Outcome']
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=2)

        model = GaussianNB()
        model.fit(x_train, y_train)

        predictions = model.predict(x_test)
        
        test_score = model.score(x_test, y_test)
        st.write('Accuracy score : ', test_score)

        cm = confusion_matrix(y_test, predictions)
        x_labels = ['Predicted negative', 'Predicted positive']
        y_labels = ['Actual negative', 'Actual positive']

        plt.figure(figsize=(9,9))
        sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues_r');

        plt.xticks(ticks=[0.5, 1.5], labels=x_labels)
        plt.yticks(ticks=[0.5, 1.5], labels=y_labels)

        plt.ylabel('Actual label');
        plt.xlabel('Predicted label');

        all_sample_title = 'Accuracy Score: {:.3f}'.format(test_score)
        plt.title(all_sample_title, size = 15);

        st.pyplot()

page_names_to_funcs = {
        "—": intro,
        "Logistic Regression": logistic_regression,
        "Support Vector Machine (SVM)": support_vector,
        "Gaussian Naive Bayes": gaussianNB,
        "LSTM": lstm
}

section_name = st.selectbox("Choose a model", page_names_to_funcs.keys())
page_names_to_funcs[section_name]()

df = load_clean_data()
X = df.drop(columns=['Outcome'])  # Features
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=2)

age = st.slider('How old is the patient?', 0, 100, 20)
pregnancies = st.slider('How many times has the patient been pregnant?', 0, 15, 2)
glucose = st.slider("What's their glucose level?", 10, 200, 30)
bp = st.slider("How about their blood pressure?", 10, 200, 70)
skin = st.slider("How thick is their skin?", 0, 100, 30)
insulin = st.slider("What's the patient's insulin level?", 10, 600, 120)
bmi = st.slider("What's the patient's body mass index?", 10, 100, 30)
dpf = st.slider("Their DPF (diabetes pedigree function)?", 0.05, 2.5, 0.5)

lr = LogisticRegression().fit(x_train, y_train)
support_vm = svm.SVC().fit(x_train, y_train)
naive = GaussianNB().fit(x_train, y_train)
lstm = tf.keras.models.load_model('static/lstm.h5')

feature = [[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]]

result_lr = lr.predict(feature)
result_svm = support_vm.predict(feature)
result_naive = naive.predict(feature)

feature_array = np.array(feature)
result_lstm = np.round(lstm.predict(feature_array))

print("TESTTT")
print(result_lstm)

if result_lr == 1:
    st.error("Patient predicted by Logistic Regression to have diabetes.")
elif result_lr == 0:
    st.success("Patient predicted by Logistic Regression not to have diabetes.")

if result_svm == 1:
    st.error("Patient predicted by SVM to have diabetes.")
elif result_svm == 0:
    st.success("Patient predicted by SVM not to have diabetes.")

if result_naive == 1:
    st.error("Patient predicted by Naive Bayes to have diabetes.")
elif result_naive == 0:
    st.success("Patient predicted by Naive Bayes not to have diabetes.")

if result_lstm == 1:
    st.error("Patient predicted by LSTM to have diabetes.")
elif result_lstm == 0:
    st.success("Patient predicted by LSTM not to have diabetes.")

from matplotlib import pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns

st.set_option('deprecation.showPyplotGlobalUse', False)

def load_clean_data():
        df = pd.read_csv("static/diabetes_clean.csv")
        return df

def load_data():
        df = pd.read_csv("diabetes.csv")
        return df

df = load_clean_data()

st.header("How many cases of diabetes 🆚 non-diabetes are in the dataset?")
st.write("As the Outcome column is integer, let's change it into boolean")
df['Outcome'] = df['Outcome'].astype(bool)

count_data = df['Outcome'].value_counts().reset_index()
count_data.columns = ['Outcome', 'Count']
count_data['text'] = 'Count: ' + count_data['Count'].astype(str)

st.warning("Click play to view bar chart")

trace = go.Bar(x=count_data['Outcome'], y=[0,0])

layout = go.Layout(
    yaxis=dict(range=[0, 550])  # Set the y-axis range from 0 to 500
)

frames = [go.Frame(
        data=[go.Bar(
                x=count_data['Outcome'],
                y=[count_data['Count'][j] if j < i else 0 for j in range(len(count_data['Outcome']))],
                text=count_data['Count'][:i].astype(str),
                textposition='outside',
                textfont=dict(color='black'),
                marker=dict(color=['blue', 'red']),
                showlegend=False
        )],
        name=f'FrameA {i}'
) for i in range(0, len(count_data)+1)]

for frame in frames:
    print(frame)

fig = go.Figure(data=[trace], frames=frames, layout=layout)

fig.update_layout(updatemenus=[
        {
        "buttons": [
        {
                "args": [None, {"frame": {"duration": 1000, "redraw": True}, "fromcurrent": True}],
                "label": "Play",
                "method": "animate",
        },
        ],
        "direction": "up",
        "showactive": False,
        "type": "buttons",
        }
        ])

fig.update_layout(updatemenus=[{"buttons": [], "direction": "up", "showactive": False, "type": "buttons", "x": 1.0, "xanchor": "right", "y": -0.2, "yanchor": "bottom"}])

st.plotly_chart(fig)

st.header("Heatmap of Features")
corr = df.corr()
plt.figure(figsize = (16, 9))
sns.heatmap(corr, annot = True, square = True)
st.pyplot()

st.header("Top 5 Correlated Features")
corr_abs = corr.abs()
mask = np.triu(np.ones_like(corr_abs, dtype=bool), k=1)
upper_corr_mat = corr_abs.where(mask)
unique_corr_pairs = upper_corr_mat.unstack().dropna()
sorted_mat = unique_corr_pairs.sort_values(ascending=False)
for index, value in sorted_mat.head(5).items():
    st.markdown(f"**{index[0]} - {index[1]}**: {value}")

def intro():
        st.warning("***Please select an option***")

def age_pregnancies():
        df = load_data()
        df.loc[df['Age'] == 0, 'Age'] = np.nan
        df = df.dropna(subset=['Age', 'Pregnancies'])

        fig = px.scatter(df, 
                        x='Age', 
                        y='Pregnancies',
                        trendline="ols",
                        trendline_color_override="green")
        st.plotly_chart(fig)

def bmi_skin():
        df_clean = load_data()
        df_clean.loc[df_clean['BMI'] == 0, 'BMI'] = np.nan
        df_clean.loc[df_clean['SkinThickness'] == 0, 'SkinThickness'] = np.nan

        df_clean = df_clean.dropna(subset=['BMI', 'SkinThickness'])

        fig = px.scatter(df_clean, 
                        x='BMI', 
                        y='SkinThickness',
                        trendline="ols",
                        trendline_color_override="green")
        st.plotly_chart(fig)

def glucose_histogram():
        df_clean = load_data()
        df_clean.loc[df_clean['Glucose'] == 0, 'Glucose'] = np.nan
        df_clean = df_clean.dropna(subset=['Glucose'])

        fig = px.histogram(df_clean, x='Glucose', color='Outcome', 
                   histnorm='probability density', marginal='rug', 
                   title='Distribution of Glucose Levels by Outcome')

        median_glucose_diabetes = df_clean[df_clean['Outcome'] == 1]['Glucose'].median()
        median_glucose_non_diabetes = df_clean[df_clean['Outcome'] == 0]['Glucose'].median()

        fig.add_annotation(x=median_glucose_diabetes, y=0.02,
                        text=f"Median Glucose (Diabetes): {median_glucose_diabetes:.2f}",
                        showarrow=True, arrowhead=1, ax=20, ay=-80)

        fig.add_annotation(x=median_glucose_non_diabetes, y=0.02,
                        text=f"Median Glucose (Non-Diabetes): {median_glucose_non_diabetes:.2f}",
                        showarrow=True, arrowhead=1, ax=-50, ay=-50)
        
        st.plotly_chart(fig)

        mean_glucose_diabetes = df_clean[df_clean['Outcome'] == 1]['Glucose'].mean()
        mean_glucose_non_diabetes = df_clean[df_clean['Outcome'] == 0]['Glucose'].mean()

        data = {
        'Outcome': ['Diabetes', 'Non-Diabetes'],
        'Mean Glucose': [mean_glucose_diabetes, mean_glucose_non_diabetes]
        }
        df_plot = pd.DataFrame(data)

        fig = px.bar(df_plot, x='Outcome', y='Mean Glucose', 
                color='Outcome', title='Mean Glucose Levels by Outcome',
                labels={'Mean Glucose': 'Mean Glucose Levels'})
        st.plotly_chart(fig)

def insulin_glucose():
        df_clean = load_data()
        df_clean.loc[df_clean['Insulin'] == 0, 'Insulin'] = np.nan
        df_clean.loc[df_clean['Glucose'] == 0, 'Glucose'] = np.nan

        df_clean = df_clean.dropna(subset=['Insulin', 'Glucose'])

        fig = px.scatter(df_clean, 
                        x='Glucose', 
                        y='Insulin',
                        trendline="ols",
                        trendline_color_override="green")
        st.plotly_chart(fig)

def age_bp():
        df_clean = load_data()
        df_clean.loc[df_clean['Age'] == 0, 'Age'] = np.nan
        df_clean.loc[df_clean['BloodPressure'] == 0, 'BloodPressure'] = np.nan

        df_clean = df_clean.dropna(subset=['Age', 'BloodPressure'])

        fig = px.scatter(df_clean, 
                        x='Age', 
                        y='BloodPressure',
                        trendline="ols",
                        trendline_color_override="green")
        st.plotly_chart(fig)

        

page_names_to_funcs = {
        "—": intro,
        "Age - Pregnancies": age_pregnancies,
        "BMI - Skin Thickness": bmi_skin,
        "Diabetes Status - Glucose": glucose_histogram,
        "Insulin - Glucose": insulin_glucose,
        "Age - Blood Pressure": age_bp
}

section_name = st.selectbox("Choose an option", page_names_to_funcs.keys())
page_names_to_funcs[section_name]()

st.header("Feature Importance based on Diabetes Status (Outcome)")
outcome_corr = corr['Outcome'].drop('Outcome').sort_values(ascending=True)
st.warning("Click play to view bar chart")
trace = go.Bar(x=[0] * len(outcome_corr),
               y=outcome_corr.index,
               orientation='h')

frames = [go.Frame(
        data=[go.Bar(
                x=[outcome_corr.values[j] if j < i else 0 for j in range(len(outcome_corr))],
                y=outcome_corr.index,
                # text=count_data['Count'][:i].astype(str),
                # textposition='outside',
                # textfont=dict(color='black'),
                marker=dict(color=outcome_corr.values, colorscale="sunset"),
                showlegend=False,
                orientation='h'
        )],
        name=f'Frame {i}'
) for i in range(0, len(outcome_corr)+1)]

for frame in frames:
    print(frame)

layout = go.Layout(
    xaxis=dict(range=[0, 0.5])  # Set the y-axis range from 0 to 500
)

fig = go.Figure(data=trace, frames=frames, layout=layout)

fig.update_layout(updatemenus=[
        {
        "buttons": [
        {
                "args": [None, {"frame": {"duration": 1000, "redraw": True}, "fromcurrent": True}],
                "label": "Play",
                "method": "animate",
        },
        ],
        "direction": "up",
        "showactive": False,
        "type": "buttons",
        }
        ])

fig.update_layout(updatemenus=[{"buttons": [], "direction": "up", "showactive": False, "type": "buttons", "x": 1.0, "xanchor": "right", "y": -0.2, "yanchor": "bottom"}])

st.plotly_chart(fig)

st.header("Top 5 Feature Selected with SelectKBest")
X = df.drop(columns=['Outcome'])
y = df['Outcome']
from sklearn.feature_selection import SelectKBest, f_classif
feature_names = df.columns.tolist()
skb = SelectKBest(score_func=f_classif, k=5)  # Set f_classif as our criteria to select features
X_data_new = skb.fit_transform(X, y)
scores = skb.scores_
selected_indices = skb.get_support(indices=True)
for i, feature_index in enumerate(selected_indices):
    st.write('- {} (Score: {:.3f})'.format(feature_names[feature_index], scores[feature_index]))

st.warning("Click play to view bar chart")

sorted_features = sorted(zip([feature_names[i] for i in selected_indices], [scores[i] for i in selected_indices]), key=lambda x: x[1], reverse=True)
sorted_feature_names, sorted_scores = zip(*sorted_features)

trace = go.Bar(x=list(sorted_feature_names),
               y=[0] * len(list(sorted_feature_names)))
frames = [go.Frame(
        data=[go.Bar(
                x=list(sorted_feature_names),
                y=[list(sorted_scores)[j] if j < i else 0 for j in range(len(list(sorted_scores)))],
                # text=count_data['Count'][:i].astype(str),
                # textposition='outside',
                # textfont=dict(color='black'),
                marker=dict(color=sorted_scores, colorscale="sunset"),
                showlegend=False,
        )],
        name=f'Frame {i}'
) for i in range(0, len(outcome_corr)+1)]

for frame in frames:
    print(frame)

layout = go.Layout(
    yaxis=dict(range=[0, 250])  # Set the y-axis range from 0 to 500
)
fig = go.Figure(data=trace, frames=frames, layout=layout)

fig.update_layout(updatemenus=[
        {
        "buttons": [
        {
                "args": [None, {"frame": {"duration": 1000, "redraw": True}, "fromcurrent": True}],
                "label": "Play",
                "method": "animate",
        },
        ],
        "direction": "up",
        "showactive": False,
        "type": "buttons",
        }
        ])

fig.update_layout(updatemenus=[{"buttons": [], "direction": "up", "showactive": False, "type": "buttons", "x": 1.0, "xanchor": "right", "y": -0.2, "yanchor": "bottom"}])

st.plotly_chart(fig)

def generate_intervals(start, end, interval):
        generated_intervals = []
        current_val = start
        while current_val < end:
                next_val = min(current_val + interval - 1, end)
                generated_intervals.append((current_val, next_val))
                current_val = next_val + 1
        return generated_intervals

import math

def categorize(val, interval):
        val = math.floor(val)
        for i, (start, end) in enumerate(interval):
                if start <= val <= end:
                        return f"{start}-{end}"
        print(f"No group found for value {val}")
        return "Outside value range"

def age_histogram():
        df_clean = load_data()
        df_clean.loc[df_clean['Age'] == 0, 'Age'] = np.nan

        df_clean = df_clean.dropna(subset=['Age'])

        fig = px.histogram(df_clean, x='Age', color='Outcome', 
                   histnorm='probability density', marginal='rug', 
                   title='Distribution of Age by Outcome')

        median_diabetes = df_clean[df_clean['Outcome'] == 1]['Age'].median()
        median_non_diabetes = df_clean[df_clean['Outcome'] == 0]['Age'].median()

        fig.add_annotation(x=median_diabetes, y=0.02,
                        text=f"Median (Diabetes): {median_diabetes:.0f}",
                        showarrow=True, arrowhead=1, ax=50, ay=-70)

        fig.add_annotation(x=median_non_diabetes, y=0.02,
                        text=f"Median (Non-Diabetes): {median_non_diabetes:.0f}",
                        showarrow=True, arrowhead=1, ax=-20, ay=-110)
        
        st.plotly_chart(fig)

        start_val = 0
        end_val = 100
        interval_size = 10

        generated_intervals = generate_intervals(start_val, end_val, interval_size)
        df_clean['Age_Group'] = df_clean['Age'].apply(lambda x: categorize(x, generated_intervals))
        group_counts = df_clean['Age_Group'].value_counts().reset_index()
        group_counts.columns = ['Age_Group', 'Count']

        group_counts['Age_Group_Lower'] = group_counts['Age_Group'].str.split('-').str[0].astype(int)
        group_counts = group_counts.sort_values(by='Age_Group_Lower')

        fig = px.bar(group_counts, x='Age_Group', y='Count', color='Count',
                labels={'Count': 'Number of Data Points'},
                title='Number of Data Points in Each Age Group',
                height=500)
        fig.update_traces(texttemplate='%{y}', textposition='outside')
        fig.update_layout(hovermode='x')

        df = df_clean

        def filter_data(df, removed_groups):
                return df[~df['Age_Group'].isin(removed_groups)]
        
        removed_groups = group_counts[group_counts['Count'] < 10]['Age_Group']

        action = st.radio("Show data with outlier groups?:", ("Yes, show me outlier", "No"))
        if action == "No":
                df_filtered = filter_data(df, removed_groups)
                group_counts_filtered = df_filtered['Age_Group'].value_counts().reset_index()
                group_counts_filtered.columns = ['Age_Group', 'Count']
                group_counts_filtered['Age_Group_Lower'] = group_counts_filtered['Age_Group'].str.split('-').str[0].astype(int)
                group_counts_filtered = group_counts_filtered.sort_values(by='Age_Group_Lower')
                fig = px.bar(group_counts_filtered, x='Age_Group', y='Count', color='Count',
                        labels={'Count': 'Number of Data Points'},
                        title='Number of Data Points in Each Age Group (Outliers removed)',
                        height=500)
                fig.update_traces(texttemplate='%{y}', textposition='outside')
                fig.update_layout(hovermode='x')
                st.plotly_chart(fig)
        else:
                st.plotly_chart(fig)

        df = filter_data(df, removed_groups)
        total_count = df['Age_Group'].value_counts().reset_index()
        total_count.columns = ['Age_Group', 'Total_Count']

        diabetes_count = df.groupby('Age_Group')['Outcome'].sum().reset_index()

        diabetes_percentage = pd.merge(total_count, diabetes_count, on='Age_Group')

        diabetes_percentage['Percentage'] = (diabetes_percentage['Outcome'] / diabetes_percentage['Total_Count']) * 100

        diabetes_percentage_sorted = diabetes_percentage.sort_values(by='Age_Group')
        fig = px.bar(diabetes_percentage_sorted, x='Age_Group', y='Percentage', color='Age_Group',
                labels={'Percentage': 'Percentage of Individuals with Diabetes'},
                title='Percentage of Individuals with Diabetes in Each Age Group',
                hover_data={'Percentage': True},
                color_continuous_scale='viridis',
                height=500)
        fig.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
        fig.update_layout(hovermode='x')
        st.plotly_chart(fig)

def pregnancies_histogram():
        df_clean = load_data()

        fig = px.histogram(df_clean, x='Pregnancies', color='Outcome', 
                   histnorm='probability density', marginal='rug', 
                   title='Distribution of Pregnancies by Outcome')

        median_diabetes = df_clean[df_clean['Outcome'] == 1]['Pregnancies'].median()
        median_non_diabetes = df_clean[df_clean['Outcome'] == 0]['Pregnancies'].median()

        fig.add_annotation(x=median_diabetes, y=0.02,
                        text=f"Median (Diabetes): {median_diabetes:.0f}",
                        showarrow=True, arrowhead=1, ax=75, ay=-120)

        fig.add_annotation(x=median_non_diabetes, y=0.02,
                        text=f"Median (Non-Diabetes): {median_non_diabetes:.0f}",
                        showarrow=True, arrowhead=1, ax=-10, ay=-120)
        st.plotly_chart(fig)

        group_counts = df_clean['Pregnancies'].value_counts().reset_index()
        group_counts.columns = ['Pregnancies', 'Count']

        df=df_clean

        fig = px.bar(group_counts, x='Pregnancies', y='Count', color='Count',
                labels={'Count': 'Number of Data Points'},
                title='Number of Data Points in Each Pregnancy Level',
                height=500)
        fig.update_traces(texttemplate='%{y}', textposition='outside')
        fig.update_layout(hovermode='x')

        df = df_clean

        def filter_data(df, removed_groups):
                return df[~df['Pregnancies'].isin(removed_groups)]
        
        removed_groups = group_counts[group_counts['Count'] < 5]['Pregnancies']

        action = st.radio("Show data with outlier groups?:", ("Yes, show me outlier", "No"))
        if action == "No":
                df_filtered = filter_data(df, removed_groups)
                group_counts_filtered = df_filtered['Pregnancies'].value_counts().reset_index()
                group_counts_filtered.columns = ['Pregnancies', 'Count']
                group_counts_filtered = group_counts_filtered.sort_values(by='Pregnancies')
                fig = px.bar(group_counts_filtered, x='Pregnancies', y='Count', color='Count',
                        labels={'Count': 'Number of Data Points'},
                        title='Number of Data Points in Each Pregnancy Level (Outliers removed)',
                        height=500)
                fig.update_traces(texttemplate='%{y}', textposition='outside')
                fig.update_layout(hovermode='x')
                st.plotly_chart(fig)
        else:
                st.plotly_chart(fig)

        df = filter_data(df, removed_groups)
        total_count = df['Pregnancies'].value_counts().reset_index()
        total_count.columns = ['Pregnancies', 'Total_Count']

        diabetes_count = df.groupby('Pregnancies')['Outcome'].sum().reset_index()

        diabetes_percentage = pd.merge(total_count, diabetes_count, on='Pregnancies')

        diabetes_percentage['Percentage'] = (diabetes_percentage['Outcome'] / diabetes_percentage['Total_Count']) * 100

        diabetes_percentage_sorted = diabetes_percentage.sort_values(by='Pregnancies')
        fig = px.bar(diabetes_percentage_sorted, x='Pregnancies', y='Percentage', color='Pregnancies',
                labels={'Percentage': 'Percentage of Individuals with Diabetes'},
                title='Percentage of Individuals with Diabetes in Each Pregnancy Level',
                hover_data={'Percentage': True},
                color_continuous_scale='viridis',
                height=500)
        fig.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
        fig.update_layout(hovermode='x')
        st.plotly_chart(fig)

def bmi_histogram():
        df_clean = load_data()
        df_clean.loc[df_clean['BMI'] == 0, 'BMI'] = np.nan

        df_clean = df_clean.dropna(subset=['BMI'])

        fig = px.histogram(df_clean, x='BMI', color='Outcome', 
                   histnorm='probability density', marginal='rug', 
                   title='Distribution of BMI by Outcome')

        median_diabetes = df_clean[df_clean['Outcome'] == 1]['BMI'].median()
        median_non_diabetes = df_clean[df_clean['Outcome'] == 0]['BMI'].median()

        fig.add_annotation(x=median_diabetes, y=0.02,
                        text=f"Median (Diabetes): {median_diabetes:.0f}",
                        showarrow=True, arrowhead=1, ax=50, ay=-120)

        fig.add_annotation(x=median_non_diabetes, y=0.02,
                        text=f"Median (Non-Diabetes): {median_non_diabetes:.0f}",
                        showarrow=True, arrowhead=1, ax=-50, ay=-120)
        st.plotly_chart(fig)

        start_bmi = 10
        end_bmi = 70
        interval_size_bmi = 5

        generated_intervals = generate_intervals(start_bmi, end_bmi, interval_size_bmi)
        df_clean['BMI_Group'] = df_clean['BMI'].apply(lambda x: categorize(x, generated_intervals))
        group_counts = df_clean['BMI_Group'].value_counts().reset_index()
        group_counts.columns = ['BMI_Group', 'Count']

        group_counts['BMI_Group_Lower'] = group_counts['BMI_Group'].str.split('-').str[0].astype(int)
        group_counts = group_counts.sort_values(by='BMI_Group_Lower')

        fig = px.bar(group_counts, x='BMI_Group', y='Count', color='Count',
                labels={'Count': 'Number of Data Points'},
                title='Number of Data Points in Each BMI Group',
                height=500)
        fig.update_traces(texttemplate='%{y}', textposition='outside')
        fig.update_layout(hovermode='x')

        df = df_clean

        def filter_data(df, removed_groups):
                return df[~df['BMI_Group'].isin(removed_groups)]
        
        removed_groups = group_counts[group_counts['Count'] < 5]['BMI_Group']

        action = st.radio("Show data with outlier groups?:", ("Yes, show me outlier", "No"))
        if action == "No":
                df_filtered = filter_data(df, removed_groups)
                group_counts_filtered = df_filtered['BMI_Group'].value_counts().reset_index()
                group_counts_filtered.columns = ['BMI_Group', 'Count']
                group_counts_filtered['BMI_Group_Lower'] = group_counts_filtered['BMI_Group'].str.split('-').str[0].astype(int)
                group_counts_filtered = group_counts_filtered.sort_values(by='BMI_Group_Lower')
                fig = px.bar(group_counts_filtered, x='BMI_Group', y='Count', color='Count',
                        labels={'Count': 'Number of Data Points'},
                        title='Number of Data Points in Each BMI Group (Outliers removed)',
                        height=500)
                fig.update_traces(texttemplate='%{y}', textposition='outside')
                fig.update_layout(hovermode='x')
                st.plotly_chart(fig)
        else:
                st.plotly_chart(fig)

        df = filter_data(df, removed_groups)
        total_count = df['BMI_Group'].value_counts().reset_index()
        total_count.columns = ['BMI_Group', 'Total_Count']

        diabetes_count = df.groupby('BMI_Group')['Outcome'].sum().reset_index()

        diabetes_percentage = pd.merge(total_count, diabetes_count, on='BMI_Group')

        diabetes_percentage['Percentage'] = (diabetes_percentage['Outcome'] / diabetes_percentage['Total_Count']) * 100

        diabetes_percentage_sorted = diabetes_percentage.sort_values(by='BMI_Group')
        fig = px.bar(diabetes_percentage_sorted, x='BMI_Group', y='Percentage', color='BMI_Group',
                labels={'Percentage': 'Percentage of Individuals with Diabetes'},
                title='Percentage of Individuals with Diabetes in Each BMI Group',
                hover_data={'Percentage': True},
                color_continuous_scale='viridis',
                height=500)
        fig.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
        fig.update_layout(hovermode='x')
        st.plotly_chart(fig)

def skin_histogram():
        df_clean = load_data()

        df_clean.loc[df_clean['SkinThickness'] == 0, 'SkinThickness'] = np.nan

        df_clean = df_clean.dropna(subset=['SkinThickness'])

        fig = px.histogram(df_clean, x='SkinThickness', color='Outcome', 
                   histnorm='probability density', marginal='rug', 
                   title='Distribution of Skin Thickness by Outcome')

        median_diabetes = df_clean[df_clean['Outcome'] == 1]['SkinThickness'].median()
        median_non_diabetes = df_clean[df_clean['Outcome'] == 0]['SkinThickness'].median()

        fig.add_annotation(x=median_diabetes, y=0.02,
                        text=f"Median (Diabetes): {median_diabetes:.0f}",
                        showarrow=True, arrowhead=1, ax=50, ay=-120)

        fig.add_annotation(x=median_non_diabetes, y=0.02,
                        text=f"Median (Non-Diabetes): {median_non_diabetes:.0f}",
                        showarrow=True, arrowhead=1, ax=-50, ay=-120)
        st.plotly_chart(fig)

        start_val = 0
        end_val = 100
        interval_size = 10

        generated_intervals = generate_intervals(start_val, end_val, interval_size)
        df_clean['SkinThickness_Group'] = df_clean['SkinThickness'].apply(lambda x: categorize(x, generated_intervals))
        group_counts = df_clean['SkinThickness_Group'].value_counts().reset_index()
        group_counts.columns = ['SkinThickness_Group', 'Count']

        group_counts['SkinThickness_Group_Lower'] = group_counts['SkinThickness_Group'].str.split('-').str[0].astype(int)
        group_counts = group_counts.sort_values(by='SkinThickness_Group_Lower')

        fig = px.bar(group_counts, x='SkinThickness_Group', y='Count', color='Count',
                labels={'Count': 'Number of Data Points'},
                title='Number of Data Points in Each Skin Thickness Group',
                height=500)
        fig.update_traces(texttemplate='%{y}', textposition='outside')
        fig.update_layout(hovermode='x')

        df = df_clean

        def filter_data(df, removed_groups):
                return df[~df['SkinThickness_Group'].isin(removed_groups)]
        
        removed_groups = group_counts[group_counts['Count'] < 10]['SkinThickness_Group']

        action = st.radio("Show data with outlier groups?:", ("Yes, show me outlier", "No"))
        if action == "No":
                df_filtered = filter_data(df, removed_groups)
                group_counts_filtered = df_filtered['SkinThickness_Group'].value_counts().reset_index()
                group_counts_filtered.columns = ['SkinThickness_Group', 'Count']
                group_counts_filtered['SkinThickness_Group_Lower'] = group_counts_filtered['SkinThickness_Group'].str.split('-').str[0].astype(int)
                group_counts_filtered = group_counts_filtered.sort_values(by='SkinThickness_Group_Lower')
                fig = px.bar(group_counts_filtered, x='SkinThickness_Group', y='Count', color='Count',
                        labels={'Count': 'Number of Data Points'},
                        title='Number of Data Points in Each Skin Thickness Group (Outliers removed)',
                        height=500)
                fig.update_traces(texttemplate='%{y}', textposition='outside')
                fig.update_layout(hovermode='x')
                st.plotly_chart(fig)
        else:
                st.plotly_chart(fig)

        df = filter_data(df, removed_groups)
        total_count = df['SkinThickness_Group'].value_counts().reset_index()
        total_count.columns = ['SkinThickness_Group', 'Total_Count']

        diabetes_count = df.groupby('SkinThickness_Group')['Outcome'].sum().reset_index()

        diabetes_percentage = pd.merge(total_count, diabetes_count, on='SkinThickness_Group')

        diabetes_percentage['Percentage'] = (diabetes_percentage['Outcome'] / diabetes_percentage['Total_Count']) * 100

        diabetes_percentage_sorted = diabetes_percentage.sort_values(by='SkinThickness_Group')
        fig = px.bar(diabetes_percentage_sorted, x='SkinThickness_Group', y='Percentage', color='SkinThickness_Group',
                labels={'Percentage': 'Percentage of Individuals with Diabetes'},
                title='Percentage of Individuals with Diabetes in Each Skin Thickness Group',
                hover_data={'Percentage': True},
                color_continuous_scale='viridis',
                height=500)
        fig.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
        fig.update_layout(hovermode='x')
        st.plotly_chart(fig)

def glucose_complete_histogram():
        df_clean = load_data()

        df_clean.loc[df_clean['Glucose'] == 0, 'Glucose'] = np.nan

        df_clean = df_clean.dropna(subset=['Glucose'])

        fig = px.histogram(df_clean, x='Glucose', color='Outcome', 
                   histnorm='probability density', marginal='rug', 
                   title='Distribution of Glucose Level by Outcome')

        median_diabetes = df_clean[df_clean['Outcome'] == 1]['Glucose'].median()
        median_non_diabetes = df_clean[df_clean['Outcome'] == 0]['Glucose'].median()

        fig.add_annotation(x=median_diabetes, y=0.02,
                        text=f"Median (Diabetes): {median_diabetes:.0f}",
                        showarrow=True, arrowhead=1, ax=50, ay=-120)

        fig.add_annotation(x=median_non_diabetes, y=0.02,
                        text=f"Median (Non-Diabetes): {median_non_diabetes:.0f}",
                        showarrow=True, arrowhead=1, ax=-50, ay=-120)
        st.plotly_chart(fig)

        start_val = 40
        end_val = 200
        interval_size = 10

        generated_intervals = generate_intervals(start_val, end_val, interval_size)
        df_clean['Glucose_Group'] = df_clean['Glucose'].apply(lambda x: categorize(x, generated_intervals))
        group_counts = df_clean['Glucose_Group'].value_counts().reset_index()
        group_counts.columns = ['Glucose_Group', 'Count']

        group_counts['Glucose_Group_Lower'] = group_counts['Glucose_Group'].str.split('-').str[0].astype(int)
        group_counts = group_counts.sort_values(by='Glucose_Group_Lower')

        fig = px.bar(group_counts, x='Glucose_Group', y='Count', color='Count',
                labels={'Count': 'Number of Data Points'},
                title='Number of Data Points in Each Skin Thickness Group',
                height=500)
        fig.update_traces(texttemplate='%{y}', textposition='outside')
        fig.update_layout(hovermode='x')

        df = df_clean

        def filter_data(df, removed_groups):
                return df[~df['Glucose_Group'].isin(removed_groups)]
        
        removed_groups = group_counts[group_counts['Count'] < 10]['Glucose_Group']

        action = st.radio("Show data with outlier groups?:", ("Yes, show me outlier", "No"))
        if action == "No":
                df_filtered = filter_data(df, removed_groups)
                group_counts_filtered = df_filtered['Glucose_Group'].value_counts().reset_index()
                group_counts_filtered.columns = ['Glucose_Group', 'Count']
                group_counts_filtered['Glucose_Group_Lower'] = group_counts_filtered['Glucose_Group'].str.split('-').str[0].astype(int)
                group_counts_filtered = group_counts_filtered.sort_values(by='Glucose_Group_Lower')
                fig = px.bar(group_counts_filtered, x='Glucose_Group', y='Count', color='Count',
                        labels={'Count': 'Number of Data Points'},
                        title='Number of Data Points in Each Skin Thickness Group (Outliers removed)',
                        height=500)
                fig.update_traces(texttemplate='%{y}', textposition='outside')
                fig.update_layout(hovermode='x')
                st.plotly_chart(fig)
        else:
                st.plotly_chart(fig)

        df = filter_data(df, removed_groups)
        total_count = df['Glucose_Group'].value_counts().reset_index()
        total_count.columns = ['Glucose_Group', 'Total_Count']

        diabetes_count = df.groupby('Glucose_Group')['Outcome'].sum().reset_index()

        diabetes_percentage = pd.merge(total_count, diabetes_count, on='Glucose_Group')

        diabetes_percentage['Percentage'] = (diabetes_percentage['Outcome'] / diabetes_percentage['Total_Count']) * 100

        diabetes_percentage['Glucose_Group_Lower'] = diabetes_percentage['Glucose_Group'].str.split('-').str[0].astype(int)

        diabetes_percentage_sorted = diabetes_percentage.sort_values(by='Glucose_Group_Lower')

        fig = px.bar(diabetes_percentage_sorted, x='Glucose_Group', y='Percentage', color='Glucose_Group',
                labels={'Percentage': 'Percentage of Individuals with Diabetes'},
                title='Percentage of Individuals with Diabetes in Each Skin Thickness Group',
                hover_data={'Percentage': True},
                color_continuous_scale='viridis',
                height=500)
        fig.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
        fig.update_layout(hovermode='x')
        st.plotly_chart(fig)

page_names_to_funcs = {
        "—": intro,
        "Glucose": glucose_complete_histogram,
        "Age": age_histogram,
        "Pregnancies": pregnancies_histogram,
        "BMI": bmi_histogram,
        "Skin Thickness": skin_histogram
}

section_name = st.selectbox("Choose an option", page_names_to_funcs.keys())
page_names_to_funcs[section_name]()

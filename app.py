import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error

# Load your dataframe
df = pd.read_csv('data.csv')

# Exploratory Data Analysis
st.header("Exploratory Data Analysis")

# Display the first few rows of the data
st.subheader("Preview of Data")
st.write(df.head())

# Display the shape of the dataframe
st.subheader("Shape of the Data")
st.write(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

# Display the different features in the data
st.subheader("Features in the Data")
st.write(df.columns.tolist())

# Display counts of each feature
st.subheader("Counts of Each Feature")
for column in df.columns:
    st.write(f"{column}: {df[column].nunique()} unique values")

# Display correlation matrix
st.subheader("Correlation Matrix")
numeric_cols = df.select_dtypes(include=[np.number])  # only select numeric columns
corr_matrix = numeric_cols.corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", ax=ax)
st.pyplot(fig)

st.sidebar.title('Filters')
brand = st.sidebar.selectbox('Brand', options=df['Brand'].unique())
feature = st.sidebar.selectbox('Feature', options=['Size', 'Capacity', 'Efficiency'])
delivery_time = st.sidebar.slider('Delivery Time', min_value=float(df['Avg. Delivery Time'].min()), max_value=float(df['Avg. Delivery Time'].max()))
expected_visitors = st.sidebar.slider('Expected Visitors', min_value=0, max_value=5000)
supply_costs = st.sidebar.slider('Supply Costs', min_value=0, max_value=500)

filtered_df = df[(df['Brand'] == brand) & (df['Avg. Delivery Time'] <= delivery_time)]


st.title("🧼 LWM Insights")
st.write("""
Welcome to LWM Insights. Here you can explore market trends, sales
predictions, and compare washing machine brands. Navigate through
the different features to gain a deeper understanding of the washing machine market.
""")

# st.sidebar.title('Filters')
# brand = st.sidebar.selectbox('Brand', options=df['Brand'].unique())
# feature = st.sidebar.selectbox('Feature', options=['Size', 'Capacity', 'Efficiency'])
# delivery_time = st.sidebar.slider('Delivery Time', min_value=float(df['Avg. Delivery Time'].min()), max_value=float(df['Avg. Delivery Time'].max()))
# expected_visitors = st.sidebar.slider('Expected Visitors', min_value=0, max_value=5000)
# supply_costs = st.sidebar.slider('Supply Costs', min_value=0, max_value=500)
#
# filtered_df = df[(df['Brand'] == brand) & (df['Avg. Delivery Time'] <= delivery_time)]

st.header('Descriptive Analytics')
# Scatter plot of Units Sold vs Seller Ratings for all brands
fig = px.scatter(df, x='Seller Ratings', y='Units Sold', color='Brand', title='Units Sold vs Seller Ratings by Brand')
st.plotly_chart(fig)

# Interactive box plots for comparison of features across brands
fig = px.box(df, x='Brand', y=feature, title=f'{feature} Comparison Across Brands')
st.plotly_chart(fig)

# Predictive analytics: multiple linear regression for 'Units Sold' based on multiple features
features = ['Seller Ratings', 'Customer Sentiment Indicator', 'Avg. Delivery Time']
X = df[features]
y = df['Units Sold']

# Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
# model = LinearRegression()
# model.fit(X_train, y_train)

# Make predictions and calculate the error
# y_pred = model.predict(X_test)
# error = mean_squared_error(y_test, y_pred)

st.header('Predictive Analytics')
st.write("Coming soon!")
# st.write(f'Mean Squared Error of the model is: {error}')

# Add predicted sales to the dataframe
# df['Predicted Units Sold'] = model.predict(df[features])

# Display a table of actual vs predicted sales
# st.dataframe(df[['Brand', 'Product Model', 'Units Sold', 'Predicted Units Sold']])

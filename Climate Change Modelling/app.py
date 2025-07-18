import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib

# Download VADER lexicon
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("climate_nasa.csv")
    df['text'] = df['text'].astype(str)
    df['text_length'] = df['text'].apply(len)
    df['sentiment_score'] = df['text'].apply(lambda x: sia.polarity_scores(x)['compound'])
    df['sentiment'] = df['sentiment_score'].apply(lambda x: 'Positive' if x >= 0.05 else 'Negative' if x <= -0.05 else 'Neutral')
    df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
    df['hour_of_day'] = pd.to_datetime(df['date']).dt.hour
    return df

df = load_data()

# Features and target
features = ['text_length', 'sentiment_score', 'sentiment', 'day_of_week', 'hour_of_day', 'commentsCount']
target = 'likesCount'

X = df[features]
y = df[target]

# Preprocessing
numeric_features = ['text_length', 'sentiment_score', 'day_of_week', 'hour_of_day', 'commentsCount']
categorical_features = ['sentiment']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first'), categorical_features)
])

# Pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.title("Likes Prediction App (Climate Comments)")

st.markdown("Fill in the form below to predict the number of likes.")

# Input form
with st.form("prediction_form"):
    text_input = st.text_area("Comment Text", "The earth is getting hotter every year.")
    comments_count = st.number_input("Number of Comments", min_value=0, max_value=1000, value=10)
    day_of_week = st.selectbox("Day of Week", options=list(range(7)), format_func=lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x])
    hour_of_day = st.slider("Hour of the Day", 0, 23, 14)

    submitted = st.form_submit_button("Predict Likes")

    if submitted:
        text_length = len(text_input)
        sentiment_score = sia.polarity_scores(text_input)['compound']
        sentiment = 'Positive' if sentiment_score >= 0.05 else 'Negative' if sentiment_score <= -0.05 else 'Neutral'

        input_df = pd.DataFrame([{
            'text_length': text_length,
            'sentiment_score': sentiment_score,
            'sentiment': sentiment,
            'day_of_week': day_of_week,
            'hour_of_day': hour_of_day,
            'commentsCount': comments_count
        }])

        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Likes: {int(prediction)}")

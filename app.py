import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Social Media User Analysis - KNN", layout="centered")

st.title("📊 Social Media User Analysis using KNN")
st.write("This app uses a **K-Nearest Neighbors (KNN)** model to classify users based on engagement.")

# Upload dataset
uploaded_file = st.file_uploader("Upload Kaggle CSV Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Create target column
    df["total_engagement"] = df["likes"] + df["comments"] + df["shares"]
    df["target"] = (df["total_engagement"] > df["total_engagement"].median()).astype(int)

    # Feature selection
    features = ["age", "followers", "following", "posts"]
    X = df[features]
    y = df["target"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Choose K
    k = st.slider("Select number of neighbors (K)", 1, 15, 5)

    # Train model
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Prediction
    y_pred = knn.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    st.success(f"✅ Model Accuracy: {accuracy:.2f}")

    st.write("**Target:**")
    st.write("0 → Low Engagement User")
    st.write("1 → High Engagement User")


import streamlit as st
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

@st.cache_resource
def train_model():
    df = pd.read_csv("train.csv")

    features = ["Pclass", "Sex", "Age", "SibSp",
                "Parch", "Fare", "Embarked"]
    X = df[features]
    y = df["Survived"]

    numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    categorical_features = ["Sex", "Embarked"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("oversampler", RandomOverSampler(random_state=42)),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(X, y)
    return model

st.title("🚢 Titanic Survival Predictor")
st.write(
    "A Machine Learning application that predicts whether "
    "a Titanic passenger would survive based on passenger details."
)

st.divider()
st.subheader("Enter Passenger Details")

with st.form("passenger_form"):
    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox("Passenger Class", [1, 2, 3])
        sex = st.selectbox("Gender", ["female", "male"])
        age = st.number_input("Age", min_value=0.0, max_value=100.0, value=25.0)
        sibsp = st.number_input("Siblings / Spouses", min_value=0, max_value=10, value=0)

    with col2:
        parch = st.number_input("Parents / Children", min_value=0, max_value=10, value=0)
        fare = st.number_input("Ticket Fare", min_value=0.0, value=80.0)
        embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

    submitted = st.form_submit_button("Predict Survival")

if submitted:
    try:
        model = train_model()

        passenger = pd.DataFrame([{
            "Pclass": pclass,
            "Sex": sex,
            "Age": age,
            "SibSp": sibsp,
            "Parch": parch,
            "Fare": fare,
            "Embarked": embarked
        }])

        prediction = model.predict(passenger)[0]
        probabilities = model.predict_proba(passenger)[0]
        survival_probability = probabilities[list(model.classes_).index(1)]

        st.divider()
        st.subheader("Prediction Result")

        if prediction == 1:
            st.success("Prediction: Survived")
        else:
            st.error("Prediction: Did not survive")

        st.metric(
            "Estimated Survival Probability",
            f"{survival_probability * 100:.2f}%"
        )
        st.progress(float(survival_probability))

        st.caption(
            "This is a machine-learning estimate based on historical "
            "Titanic passenger data, not a guarantee."
        )

    except Exception as e:
        st.error(f"Unable to make prediction: {e}")

st.divider()
st.caption("Titanic Survival Prediction | Machine Learning Course Project")

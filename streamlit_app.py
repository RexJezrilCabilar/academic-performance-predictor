from pathlib import Path

import joblib
import pandas as pd
import sklearn
import streamlit as st


st.set_page_config(page_title="Academic Performance Rating predictor", page_icon="📊")


MODEL_PATH = Path(__file__).parent / "rf_model.joblib"

FEATURES = [
    "Submission.Actions",
    "Quiz.Actions",
    "Forum.Actions",
    "Create.Actions",
    "Read.Actions",
    "Update.Actions",
    "Delete.Actions",
    "Days.Logged.In",
]

# What each box shows at the start: the median (middle) value in the training data.
DEFAULTS = {
    "Submission.Actions": 3,
    "Quiz.Actions": 0,
    "Forum.Actions": 0,
    "Create.Actions": 7,
    "Read.Actions": 131,
    "Update.Actions": 3,
    "Delete.Actions": 0,
    "Days.Logged.In": 14,
}

# Smallest and largest value of each column in the training data.
# Only used to warn about numbers the model has never seen.
TRAIN_RANGE = {
    "Submission.Actions": (0, 64),
    "Quiz.Actions": (0, 691),
    "Forum.Actions": (0, 143),
    "Create.Actions": (0, 88),
    "Read.Actions": (1, 398872),
    "Update.Actions": (0, 344),
    "Delete.Actions": (0, 3),
    "Days.Logged.In": (0, 55),
}


# ---------------------------------------------------------------------
# Load the model (loaded once, then kept in memory)
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the model...")
def load_model():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error(
        f"The model file `{MODEL_PATH.name}` was not found. "
        "Upload it to your GitHub repo, in the same folder as `streamlit_app.py`."
    )
    st.stop()

try:
    model = load_model()
except Exception as error:
    st.error(
        "The model file was found but could not be loaded. This usually means the "
        "scikit-learn version here is different from the one used in Colab. "
        "Make sure `requirements.txt` has the same `scikit-learn==` version that "
        "the Colab cell printed."
    )
    st.code(str(error))
    st.stop()


# ---------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------
st.title("Academic Performance Rating predictor")
st.write(
    "This website predicts whether a student is likely to pass or fail based on multiple predictors of student success.\n\n"
    "Enter the activity counts below, then click **Predict rating**. "
    "The model predicts whether the rating is **High** or **Low**."
)

with st.form("prediction_form"):
    left, right = st.columns(2)
    values = {}
    for i, feature in enumerate(FEATURES):
        box = left if i % 2 == 0 else right
        values[feature] = box.number_input(
            feature.replace(".", " "),  # "Submission.Actions" is shown as "Submission Actions"
            min_value=0,
            value=DEFAULTS[feature],
            step=1,
        )
    submitted = st.form_submit_button("Predict rating")

if submitted:
    # One row with the same column names and order as the training data
    row = pd.DataFrame([values], columns=FEATURES)

    prediction = str(model.predict(row)[0])
    chances = {
        str(label): float(p)
        for label, p in zip(model.classes_, model.predict_proba(row)[0])
    }

    st.subheader("Result")
    col1, col2 = st.columns(2)
    col1.metric("Predicted rating", prediction)
    col2.metric("Confidence", f"{chances[prediction]:.1%}")

    st.write("**Chance of each rating**")
    for label, p in chances.items():
        st.write(f"{label}: {p:.1%}")
        st.progress(min(max(p, 0.0), 1.0))

    # Warn if any number is outside what the model saw during training
    unusual = [
        name
        for name, (low, high) in TRAIN_RANGE.items()
        if not low <= values[name] <= high
    ]
    if unusual:
        st.warning(
            "These values are outside the range the model was trained on, so the "
            "result may be less reliable: "
            + ", ".join(n.replace(".", " ") for n in unusual)
            + "."
        )

    st.caption("This is the model's best estimate. It can be wrong.")

with st.expander("About this model"):
    st.write(
        f"- Description: This website predicts whether a student is likely to pass or fail based on multiple predictors of student success.\n"
        f"- Type: Random Forest with {len(model.estimators_)} trees\n"
        f"- Ratings it can predict: {', '.join(map(str, model.classes_))}\n"
        f"- scikit-learn version running this app: {sklearn.__version__}"
    )
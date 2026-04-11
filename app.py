from flask import Flask, render_template, request
import pandas as pd
import joblib
import shap

app = Flask(__name__)

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------

model = joblib.load("attrition_model_calibrated.joblib")

# Extract original pipeline from calibrated model
base_pipeline = model.calibrated_classifiers_[0].estimator

print("Base pipeline type:", type(base_pipeline))

# ------------------------------------------------
# PIPELINE COMPONENTS
# ------------------------------------------------

preprocessor = base_pipeline.named_steps["preprocess"]
final_model = base_pipeline.named_steps["model"]

FEATURES = list(preprocessor.feature_names_in_)
FEATURE_NAMES_TRANSFORMED = preprocessor.get_feature_names_out()

# ------------------------------------------------
# EXTRACT FEATURE TYPES
# ------------------------------------------------

cat_features = preprocessor.transformers_[0][2]
encoder = preprocessor.transformers_[0][1]

CATEGORY_MAP = {
    feature: categories.tolist()
    for feature, categories in zip(cat_features, encoder.categories_)
}

NUMERIC_FEATURES = preprocessor.transformers_[1][2]

# ------------------------------------------------
# SHAP EXPLAINER (created once)
# ------------------------------------------------

explainer = shap.TreeExplainer(final_model)

# ------------------------------------------------
# FLASK ROUTE
# ------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    probability = None
    risk = None
    shap_labels = []
    shap_values = []
    input_values = {}

    if request.method == "POST":

        # Collect form inputs
        for feature in FEATURES:
            input_values[feature] = request.form.get(feature)

        df = pd.DataFrame([input_values])

        # Convert numeric columns safely
        for col in NUMERIC_FEATURES:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass

        # -----------------------------
        # SHAP EXPLANATION
        # -----------------------------

        X_transformed = preprocessor.transform(df)
        shap_vals = explainer.shap_values(X_transformed)

        # Handle binary classification
        if isinstance(shap_vals, list):
            shap_contrib = shap_vals[1][0]  # class 1 (attrition)
        else:
            shap_contrib = shap_vals[0]

        # Top 5 features
        top_shap = sorted(
            zip(FEATURE_NAMES_TRANSFORMED, shap_contrib),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]

        shap_data = [
                {
                    "label": str(x[0]),
                    "value": round(float(x[1]), 4)
                }
                for x in top_shap
            ]

        # -----------------------------
        # PREDICTION
        # -----------------------------

        probability = model.predict_proba(df)[0][1]
        prediction = int(probability >= 0.35)

        if probability < 0.25:
            risk = "Low"
        elif probability < 0.55:
            risk = "Moderate"
        elif probability < 0.75:
            risk = "High"
        else:
            risk = "Critical"

    # Return template for GET and POST
    return render_template(
        "index.html",
        cat_features=CATEGORY_MAP,
        shap_data=shap_data,
        num_features=NUMERIC_FEATURES,
        prediction=prediction,
        probability=probability,
        input_values=input_values,
        risk=risk
    )

# ------------------------------------------------
# RUN APP
# ------------------------------------------------

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

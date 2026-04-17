from flask import Flask, render_template, request
import pandas as pd
import joblib
import shap

app = Flask(__name__)   # 🔥 MUST COME BEFORE ANY @app.route

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------

model = joblib.load("attrition_model_calibrated.joblib")

base_pipeline = model.calibrated_classifiers_[0].estimator

# ------------------------------------------------
# PIPELINE COMPONENTS
# ------------------------------------------------

preprocessor = base_pipeline.named_steps["preprocess"]
final_model = base_pipeline.named_steps["model"]

FEATURES = list(preprocessor.feature_names_in_)
FEATURE_NAMES_TRANSFORMED = preprocessor.get_feature_names_out()

# ------------------------------------------------
# FEATURE TYPES
# ------------------------------------------------

cat_features = preprocessor.transformers_[0][2]
encoder = preprocessor.transformers_[0][1]

CATEGORY_MAP = {
    feature: categories.tolist()
    for feature, categories in zip(cat_features, encoder.categories_)
}

NUMERIC_FEATURES = preprocessor.transformers_[1][2]

NUMERIC_RANGES = {
    "Age": (18, 60),
    "MonthlyIncome": (1000, 20000),
    "MonthlyRate": (1000, 20000),
    "HourlyRate": (30, 200),
    "DistanceFromHome": (1, 30),
    "YearsAtCompany": (0, 40),
    "DailyRate": (300, 3000),
    "YearsInCurrentRole": (0, 10),
    "PercentSalaryHike": (5, 30),
    "TrainingTimesLastYear": (0, 7),
    "YearsSinceLastPromotion": (0, 10),
    "YearsWithCurrManager": (0, 10),
    "TotalWorkingYears": (0, 40),

    # Satisfaction columns → fixed 1–5
    "JobSatisfaction": (1, 5),
    "EnvironmentSatisfaction": (1, 5),
    "PerformanceRating": (1, 5),
    "JobInvolvement": (1, 5),
    "JobLevel": (1, 5),
    "RelationshipSatisfaction": (1, 5),
    "WorkLifeBalance": (1, 5),
    "Education": (1, 5),
    "StockOptionLevel": (0, 5),
}

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    probability = None
    risk = None
    shap_data = []
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
        # TRANSFORM DATA
        # -----------------------------
        X_transformed = preprocessor.transform(df)

        # -----------------------------
        # SHAP EXPLANATION
        # -----------------------------
        shap_data = []
        
        try:
            explainer = shap.TreeExplainer(final_model)
            shap_vals = explainer.shap_values(X_transformed)
        
            # binary handling
            if isinstance(shap_vals, list):
                shap_contrib = shap_vals[1][0]
            else:
                shap_contrib = shap_vals[0]
        
            feature_names = preprocessor.get_feature_names_out()
        
            top_shap = sorted(
                zip(feature_names, shap_contrib),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]
        
            shap_data = [
                {"label": str(name), "value": round(float(val), 4)}
                for name, val in top_shap
            ]
        
        except Exception as e:
            print("SHAP ERROR:", e)
            shap_data = []
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

    return render_template(
    "index.html",
    cat_features=CATEGORY_MAP,
    num_features=NUMERIC_FEATURES,
    prediction=prediction,
    probability=probability,
    input_values=input_values,
    risk=risk,
    shap_data=shap_data
   
    )

import joblib
import pandas as pd
import shap
import numpy as np


# ============================================================
# SMART CROP AI
# EXPLAINABLE AI USING SHAP
# ============================================================

print("Loading ML model...")

model = joblib.load(
    "models/crop_model.pkl"
)


# ============================================================
# FARM INPUT
# ============================================================

input_data = {
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 25,
    "humidity": 80,
    "ph": 6.5,
    "rainfall": 200
}


input_df = pd.DataFrame(
    [input_data]
)


# ============================================================
# ML PREDICTION
# ============================================================

print("Running prediction...")

prediction = model.predict(
    input_df
)[0]


probabilities = model.predict_proba(
    input_df
)[0]


classes = model.classes_

prediction_index = list(
    classes
).index(prediction)


confidence = (
    probabilities[prediction_index] * 100
)


# ============================================================
# SHAP EXPLAINER
# ============================================================

print("Calculating SHAP explanation...")

explainer = shap.TreeExplainer(
    model
)


shap_values = explainer.shap_values(
    input_df
)


# ============================================================
# CONVERT SHAP OUTPUT TO NUMPY ARRAY
# ============================================================

if isinstance(shap_values, list):

    # Older SHAP format:
    # list of arrays, one array per class

    values = np.asarray(
        shap_values[prediction_index]
    )

else:

    values = np.asarray(
        shap_values
    )


# ============================================================
# FIX DIFFERENT SHAP DIMENSIONS
# ============================================================

print(
    "SHAP output shape:",
    values.shape
)


# Remove dimensions of size 1

values = np.squeeze(
    values
)


# If multiple class values remain,
# select the predicted class

if values.ndim > 1:

    # Common shape:
    # features × classes

    if values.shape[0] == len(input_df.columns):

        values = values[
            :,
            prediction_index
        ]

    # Common shape:
    # classes × features

    elif values.shape[1] == len(input_df.columns):

        values = values[
            prediction_index,
            : 
        ]


# Make sure it is one-dimensional

values = np.asarray(
    values
).flatten()


# ============================================================
# SAFETY CHECK
# ============================================================

if len(values) != len(input_df.columns):

    print(
        "\nERROR: SHAP returned an unexpected "
        "number of feature values."
    )

    print(
        "Number of features:",
        len(input_df.columns)
    )

    print(
        "Number of SHAP values:",
        len(values)
    )

    print(
        "\nPlease send me the SHAP output shape "
        "shown above."
    )

    exit()


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

features = list(
    input_df.columns
)


feature_importance = []


for feature, value in zip(
    features,
    values
):

    # Convert NumPy value to normal Python float

    value = float(value)


    feature_importance.append(
        (
            feature,
            value
        )
    )


# Sort by absolute contribution

feature_importance.sort(
    key=lambda x: abs(x[1]),
    reverse=True
)


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n")
print("================================================")
print("             🌾 SMART CROP AI")
print("================================================")


print(
    f"\nRecommended Crop: {prediction}"
)


print(
    f"Confidence: {confidence:.2f}%"
)


# ============================================================
# FEATURE CONTRIBUTIONS
# ============================================================

print(
    "\nWhy did the model choose this crop?"
)


print(
    "\nFeature Contributions:"
)


for feature, value in feature_importance:

    if value > 0:

        effect = "supports"

    elif value < 0:

        effect = "reduces"

    else:

        effect = "has little effect on"


    print(
        f"{feature:15} "
        f"{value:+.4f} "
        f"→ {effect} {prediction}"
    )


# ============================================================
# TOP 3 IMPORTANT FEATURES
# ============================================================

print(
    "\nTop Factors Influencing Prediction:"
)


for i, (
    feature,
    value
) in enumerate(
    feature_importance[:3],
    start=1
):

    if value > 0:

        direction = "positive"

    elif value < 0:

        direction = "negative"

    else:

        direction = "neutral"


    print(
        f"{i}. {feature} "
        f"({direction}, "
        f"{abs(value):.4f})"
    )


# ============================================================
# EXPLANATION SUMMARY
# ============================================================

print("\n")
print("================================================")
print("             EXPLANATION SUMMARY")
print("================================================")


print(
    f"\nThe machine-learning model predicted "
    f"{prediction}."
)


print(
    f"The model confidence is "
    f"{confidence:.2f}%."
)


print(
    "\nThe SHAP values explain how each "
    "input feature influenced this prediction."
)


# ============================================================
# STRONGEST FACTOR
# ============================================================

if len(feature_importance) > 0:

    strongest_feature = (
        feature_importance[0][0]
    )

    strongest_value = (
        feature_importance[0][1]
    )


    if strongest_value > 0:

        print(
            f"\nStrongest positive influence: "
            f"{strongest_feature}"
        )

    elif strongest_value < 0:

        print(
            f"\nStrongest influence: "
            f"{strongest_feature}"
        )


# ============================================================
# UNCERTAINTY
# ============================================================

if len(probabilities) >= 2:

    sorted_probabilities = np.sort(
        probabilities
    )[::-1]


    difference = (
        sorted_probabilities[0]
        -
        sorted_probabilities[1]
    )


    if difference < 0.10:

        print(
            "\n⚠️ Important:"
        )

        print(
            "The top two crop probabilities "
            "are close. The prediction has "
            "significant uncertainty."
        )


# ============================================================
# DISCLAIMER
# ============================================================

print(
    "\nNote:"
)

print(
    "SHAP explains the machine-learning "
    "model's prediction. It does not "
    "guarantee crop suitability, yield, "
    "or profit."
)


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("================================================")
print("          ✅ EXPLANATION COMPLETE")
print("================================================")
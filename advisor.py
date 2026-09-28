import joblib
import faiss
import ollama
import pandas as pd
import shap
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# SMART CROP AI
# ML + SHAP + FAISS RAG + LOCAL LLAMA
# ENGLISH + HINDI
# ============================================================


# ============================================================
# 1. LOAD ML MODEL
# ============================================================

print("Loading ML model...")

model = joblib.load(
    "models/crop_model.pkl"
)


# ============================================================
# 2. LOAD RAG
# ============================================================

print("Loading agricultural knowledge...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

index = faiss.read_index(
    "models/agriculture.index"
)


with open(
    "models/agriculture_chunks.txt",
    "r",
    encoding="utf-8"
) as file:

    chunks = [
        chunk.strip()
        for chunk in file.read().split("\n\n")
        if chunk.strip()
    ]


print(
    f"Knowledge chunks loaded: {len(chunks)}"
)


# ============================================================
# 3. RAG RETRIEVAL
# ============================================================

def retrieve_knowledge(query, top_k=3):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx == -1:
            continue

        if idx >= len(chunks):
            continue

        results.append({
            "score": float(score),
            "text": chunks[idx]
        })

    return results


# ============================================================
# 4. SHAP
# ============================================================

def explain_prediction(
    model,
    input_df,
    prediction
):

    print(
        "\nCalculating SHAP explanation..."
    )

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer.shap_values(
        input_df
    )

    classes = model.classes_

    prediction_index = list(
        classes
    ).index(prediction)


    if isinstance(
        shap_values,
        list
    ):

        values = np.asarray(
            shap_values[prediction_index]
        )

    else:

        values = np.asarray(
            shap_values
        )


    values = np.squeeze(
        values
    )


    if values.ndim > 1:

        if values.shape[0] == len(
            input_df.columns
        ):

            values = values[
                :,
                prediction_index
            ]

        elif values.shape[1] == len(
            input_df.columns
        ):

            values = values[
                prediction_index,
                :
            ]


    values = np.asarray(
        values
    ).flatten()


    if len(values) != len(
        input_df.columns
    ):

        print(
            "Warning: SHAP output could "
            "not be interpreted."
        )

        return []


    feature_importance = []

    for feature, value in zip(
        input_df.columns,
        values
    ):

        feature_importance.append(
            (
                feature,
                float(value)
            )
        )


    feature_importance.sort(
        key=lambda x: abs(x[1]),
        reverse=True
    )


    return feature_importance


# ============================================================
# 5. FEATURE NAMES
# ============================================================

ENGLISH_FEATURE_NAMES = {

    "N": "Nitrogen (N)",
    "P": "Phosphorus (P)",
    "K": "Potassium (K)",
    "temperature": "Temperature",
    "humidity": "Humidity",
    "ph": "Soil pH",
    "rainfall": "Rainfall"
}


HINDI_FEATURE_NAMES = {

    "N": "नाइट्रोजन (N)",
    "P": "फॉस्फोरस (P)",
    "K": "पोटैशियम (K)",
    "temperature": "तापमान",
    "humidity": "आर्द्रता",
    "ph": "मिट्टी का pH",
    "rainfall": "वर्षा"
}


# ============================================================
# 6. LANGUAGE SELECTION
# ============================================================

def select_language():

    print("\n")
    print("================================================")
    print("              भाषा चुनें / SELECT LANGUAGE")
    print("================================================")

    print("\n1. English")
    print("2. Hindi")

    while True:

        choice = input(
            "\nEnter choice (1/2): "
        ).strip()

        if choice == "1":

            return "English"

        elif choice == "2":

            return "Hindi"

        else:

            print(
                "Please enter 1 or 2."
            )


# ============================================================
# 7. SHAP DISPLAY
# ============================================================

def display_shap(
    shap_results,
    language,
    prediction
):

    print("\n")
    print("================================================")
    print("                 फसल का चयन क्यों?")
    print("================================================")


    if not shap_results:

        print(
            "SHAP explanation unavailable."
        )

        return


    if language == "English":

        for feature, value in shap_results:

            name = ENGLISH_FEATURE_NAMES.get(
                feature,
                feature
            )

            if value > 0:

                meaning = (
                    f"supports {prediction}"
                )

                arrow = "↑"

            elif value < 0:

                meaning = (
                    f"reduces {prediction}"
                )

                arrow = "↓"

            else:

                meaning = (
                    f"has little effect on {prediction}"
                )

                arrow = "→"

            print(
                f"{name:20} "
                f"{value:+.4f} "
                f"{arrow} {meaning}"
            )


    else:

        print(
            "\nSHAP मॉडल की व्याख्या:"
        )

        for feature, value in shap_results:

            name = HINDI_FEATURE_NAMES.get(
                feature,
                feature
            )

            if value > 0:

                meaning = "समर्थन करता है"
                arrow = "↑"

            elif value < 0:

                meaning = "कम करता है"
                arrow = "↓"

            else:

                meaning = "बहुत कम प्रभाव डालता है"
                arrow = "→"

            print(
                f"{name:20} "
                f"{value:+.4f} "
                f"{arrow} {meaning}"
            )

            # ============================================================
# 8. GENERATE LOCAL AI EXPLANATION
# ============================================================
def generate_ai_explanation(
    crop,
    confidence,
    input_data,
    top_crops,
    retrieved,
    language
):

    context = "\n\n".join(
        item["text"]
        for item in retrieved
    )

    top_text = "\n".join(
        f"{i}. {crop_name}: {probability:.2f}%"
        for i, (crop_name, probability)
        in enumerate(top_crops, start=1)
    )

    if language == "English":

        prompt = f"""
You are Smart Crop AI.

Give a short and simple explanation for a farmer.

Recommended crop: {crop}
Model confidence: {confidence:.2f}%

Top 3 predictions:
{top_text}

Farm conditions:
Nitrogen: {input_data["N"]}
Phosphorus: {input_data["P"]}
Potassium: {input_data["K"]}
Temperature: {input_data["temperature"]} °C
Humidity: {input_data["humidity"]} %
Soil pH: {input_data["ph"]}
Rainfall: {input_data["rainfall"]} mm

Agricultural knowledge:
{context}

Write only:

Why This Crop:
Explain briefly why the model selected this crop.

Practical Advice:
Give simple, general agricultural guidance based only on the supplied knowledge.
Do not tell the farmer to apply a specific fertilizer or nutrient.
Recommend soil testing before making fertilizer decisions.

Do not invent fertilizer or pesticide quantities.
Do not guarantee crop yield or profit.
Mention that the prediction is not a guarantee.
"""

    else:

        prompt = f"""
आप Smart Crop AI हैं।

किसान के लिए सरल और स्पष्ट हिंदी में उत्तर दें।

अनुशंसित फसल: {crop}

मॉडल का विश्वास स्तर: {confidence:.2f}%

शीर्ष 3 फसल भविष्यवाणियां:
{top_text}

खेत की जानकारी:

नाइट्रोजन (N): {input_data["N"]}
फॉस्फोरस (P): {input_data["P"]}
पोटैशियम (K): {input_data["K"]}
तापमान: {input_data["temperature"]} °C
आर्द्रता: {input_data["humidity"]} %
मिट्टी का pH: {input_data["ph"]}
वर्षा: {input_data["rainfall"]} mm

कृषि ज्ञान:
{context}

केवल इन दो भागों में उत्तर दें:

यह फसल क्यों चुनी गई:
दिए गए डेटा और कृषि ज्ञान के आधार पर 2-4 सरल वाक्यों में बताएं।

व्यावहारिक सलाह:
दिए गए कृषि ज्ञान के आधार पर 2-4 सरल वाक्यों में सलाह दें।

कोई उर्वरक या कीटनाशक की मात्रा स्वयं निर्धारित न करें।
फसल की उपज या सफलता की गारंटी न दें।
मॉडल की भविष्यवाणी को केवल निर्णय लेने में सहायता के रूप में बताएं।

केवल प्राकृतिक और सही हिंदी में लिखें।
"""

    response = ollama.chat(
        model="qwen2.5:1.5b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]
# ============================================================
# 9. CONTROLLED SAFETY MESSAGE
# ============================================================

def print_safety_message(
    confidence,
    language
):

    print("\n")
    print("================================================")


    if language == "English":

        print("IMPORTANT PRECAUTIONS")

        print("================================================")

        print(
            f"• Model confidence: "
            f"{confidence:.2f}%."
        )

        print(
            "• This is a decision-support "
            "prediction, not a guarantee."
        )

        print(
            "• Soil testing is recommended "
            "before fertilizer decisions."
        )

        print(
            "• Follow locally recommended "
            "agricultural practices."
        )

        print(
            "• Consider local climate, soil "
            "conditions and pest/disease pressure."
        )

        print(
            "• SHAP shows how input features "
            "influenced the model prediction."
        )

        print(
            "• SHAP does not prove that a crop "
            "will definitely grow successfully."
        )


    else:

        print("महत्वपूर्ण सावधानियां")

        print("================================================")

        print(
            f"• मॉडल का विश्वास स्तर: "
            f"{confidence:.2f}%।"
        )

        print(
            "• यह केवल निर्णय लेने में सहायता "
            "करने वाला मॉडल है, गारंटी नहीं।"
        )

        print(
            "• उर्वरक संबंधी निर्णय से पहले "
            "मिट्टी की जांच करवाने की सलाह दी जाती है।"
        )

        print(
            "• स्थानीय कृषि विशेषज्ञों और "
            "स्थानीय कृषि सिफारिशों का पालन करें।"
        )

        print(
            "• स्थानीय जलवायु, मिट्टी और "
            "कीट/रोग की स्थिति को भी ध्यान में रखें।"
        )

        print(
            "• SHAP यह बताता है कि कौन से "
            "इनपुट मॉडल की भविष्यवाणी को प्रभावित करते हैं।"
        )

        print(
            "• SHAP यह साबित नहीं करता कि "
            "फसल निश्चित रूप से सफल होगी।"
        )


# ============================================================
# 10. DISPLAY RAG SOURCES
# ============================================================

def display_sources(
    retrieved,
    language
):

    print("\n")
    print("================================================")


    if language == "English":

        print("RETRIEVED AGRICULTURAL SOURCES")

    else:

        print("प्राप्त कृषि जानकारी के स्रोत")


    print("================================================")


    for i, result in enumerate(
        retrieved,
        start=1
    ):

        print(
            f"\nSource {i}"
        )

        print(
            f"Similarity: "
            f"{result['score']:.4f}"
        )

        print(
            result["text"]
        )


# ============================================================
# 11. CALCULATE UNCERTAINTY
# ============================================================

def calculate_uncertainty(
    top_crops
):

    if len(top_crops) < 2:

        return 100.0


    return (
        top_crops[0][1]
        -
        top_crops[1][1]
    )

    # ============================================================
# 12. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("================================================")
    print("                 SMART CROP AI")
    print("================================================")

    print(
        "\nOffline AI-powered crop recommendation system"
    )


    # ========================================================
    # LANGUAGE
    # ========================================================

    language = select_language()


    # ========================================================
    # FARM INPUT
    # ========================================================

    print("\n")
    print("================================================")
    print("              FARM INFORMATION")
    print("================================================")

    try:

        N = float(
            input("\nNitrogen (N): ")
        )

        P = float(
            input("Phosphorus (P): ")
        )

        K = float(
            input("Potassium (K): ")
        )

        temperature = float(
            input("Temperature (°C): ")
        )

        humidity = float(
            input("Humidity (%): ")
        )

        ph = float(
            input("Soil pH: ")
        )

        rainfall = float(
            input("Rainfall (mm): ")
        )

    except ValueError:

        print(
            "\nERROR: Please enter numbers only."
        )

        raise SystemExit


    # ========================================================
    # CREATE INPUT DATA
    # ========================================================

    input_data = {

        "N": N,

        "P": P,

        "K": K,

        "temperature": temperature,

        "humidity": humidity,

        "ph": ph,

        "rainfall": rainfall
    }


    input_df = pd.DataFrame(
        [input_data]
    )


    # ========================================================
    # MACHINE LEARNING PREDICTION
    # ========================================================

    print("\n")
    print("================================================")
    print("          RUNNING ML PREDICTION")
    print("================================================")


    prediction = model.predict(
        input_df
    )[0]


    probabilities = model.predict_proba(
        input_df
    )[0]


    classes = model.classes_


    # ========================================================
    # SORT PREDICTIONS
    # ========================================================

    prediction_results = list(
        zip(
            classes,
            probabilities
        )
    )


    prediction_results.sort(
        key=lambda x: x[1],
        reverse=True
    )


    # Top 3

    top_crops = [

        (
            crop,
            probability * 100
        )

        for crop, probability
        in prediction_results[:3]
    ]


    confidence = top_crops[0][1]


    # ========================================================
    # DISPLAY PREDICTION
    # ========================================================

    print("\n")
    print("================================================")
    print("              CROP PREDICTION")
    print("================================================")


    print(
        f"\nRecommended Crop: "
        f"{prediction}"
    )


    print(
        f"Confidence: "
        f"{confidence:.2f}%"
    )


    print(
        "\nTop 3 Crop Predictions:"
    )


    for i, (
        crop,
        probability
    ) in enumerate(
        top_crops,
        start=1
    ):

        print(
            f"{i}. {crop}: "
            f"{probability:.2f}%"
        )


    # ========================================================
    # UNCERTAINTY
    # ========================================================

    prediction_gap = calculate_uncertainty(
        top_crops
    )


    print(
        f"\nPrediction gap: "
        f"{prediction_gap:.2f}%"
    )


    if prediction_gap < 10:

        print(
            "⚠️ The top predictions are close. "
            "Model uncertainty is relatively high."
        )

    elif confidence < 70:

        print(
            "⚠️ Model confidence is moderate/low."
        )

    else:

        print(
            "Model confidence is relatively high."
        )


    # ========================================================
    # SHAP EXPLANATION
    # ========================================================

    shap_results = explain_prediction(

        model=model,

        input_df=input_df,

        prediction=prediction
    )


    # ========================================================
    # DISPLAY SHAP
    # ========================================================

    display_shap(

        shap_results=shap_results,

        language=language,

        prediction=prediction
    )


    # ========================================================
    # RAG RETRIEVAL
    # ========================================================

    print("\n")
    print("================================================")
    print("        RETRIEVING AGRICULTURAL KNOWLEDGE")
    print("================================================")


    rag_query = f"""
    {prediction} crop.

    Soil requirements.
    Water and irrigation.
    Nutrient requirements.
    Temperature.
    Humidity.
    Rainfall.
    Agricultural precautions.
    """

    retrieved = retrieve_knowledge(

        rag_query,

        top_k=3
    )


    if not retrieved:

        print(
            "No agricultural knowledge retrieved."
        )

    else:

        print(
            f"Retrieved {len(retrieved)} "
            f"knowledge sources."
        )


    # ========================================================
    # LOCAL AI
    # ========================================================

    print("\n")
    print("================================================")
    print("          LOCAL GENERATIVE AI")
    print("================================================")


    print(
        f"Generating {language} advisory..."
    )


    try:

        ai_explanation = generate_ai_explanation(

            crop=prediction,

            confidence=confidence,

            input_data=input_data,

            top_crops=top_crops,

            retrieved=retrieved,

            language=language
        )

    except Exception as error:

        print(
            "\nLocal AI generation failed."
        )

        print(
            f"Error: {error}"
        )

        ai_explanation = ""


    # ========================================================
    # DISPLAY AI ADVISORY
    # ========================================================

    print("\n")
    print("================================================")
    print("                AI ADVISORY")
    print("================================================")


    if ai_explanation:

        print(
            "\n" + ai_explanation
        )

    else:

        print(
            "AI advisory could not be generated."
        )


    # ========================================================
    # CONTROLLED SAFETY MESSAGE
    # ========================================================

    print_safety_message(

        confidence=confidence,

        language=language
    )


    # ========================================================
    # RAG SOURCES
    # ========================================================

    display_sources(

        retrieved=retrieved,

        language=language
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("================================================")
    print("               FINAL SUMMARY")
    print("================================================")


    if language == "English":

        print(
            f"\nRecommended Crop : {prediction}"
        )

        print(
            f"Confidence       : {confidence:.2f}%"
        )

        print(
            f"Top Alternative  : "
            f"{top_crops[1][0] if len(top_crops) > 1 else 'N/A'}"
        )

        print(
            "\nPipeline:"
        )

        print(
            "Dataset → ML → Top 3 → SHAP → "
            "FAISS RAG → Local Ollama → Advisory"
        )


    else:

        print(
            f"\nअनुशंसित फसल : {prediction}"
        )

        print(
            f"विश्वास स्तर : {confidence:.2f}%"
        )

        print(
            f"मुख्य वैकल्पिक फसल : "
            f"{top_crops[1][0] if len(top_crops) > 1 else 'उपलब्ध नहीं'}"
        )

        print(
            "\nसिस्टम पाइपलाइन:"
        )

        print(
            "डेटासेट → ML → Top 3 → SHAP → "
            "FAISS RAG → Local Ollama → Advisory"
        )


    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")
    print("================================================")
    print("             PIPELINE COMPLETE")
    print("================================================")

    print(
        "\nSMART CROP AI finished successfully."
    )
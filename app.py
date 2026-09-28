import streamlit as st
import plotly.express as px
import joblib
import pandas as pd
import numpy as np
import faiss
import shap
import ollama

from sentence_transformers import SentenceTransformer
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from streamlit_mic_recorder import speech_to_text
from pathlib import Path
import base64


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Crop AI",
    page_icon="🌾",
    layout="wide"
)

# ============================================================
# BACKGROUND IMAGE
# ============================================================

background_path = (
    Path(__file__).resolve().parent.parent
    / "assests"
    / "farmer_ai_background.jpg"
)

with open(background_path, "rb") as image_file:
    background_base64 = base64.b64encode(
        image_file.read()
    ).decode()
# ============================================================
# CUSTOM STYLE
# ============================================================
# ============================================================
# CUSTOM STYLE
# ============================================================

st.markdown(
    f"""
    <style>

    .stApp {{
        background-image: url("data:image/jpeg;base64,{background_base64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    .main-title {{
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }}

    .subtitle {{
        font-size: 18px;
        opacity: 0.75;
        margin-bottom: 20px;
    }}

    .card {{
        padding: 20px;
        border-radius: 18px;
        border: 1px solid rgba(255, 255, 255, 0.25);
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
        margin-bottom: 15px;
    }}

    div[data-testid="stMetric"] {{
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.25);
        border-radius: 16px;
        padding: 15px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.10);
    }}

    .big-crop {{
        font-size: 32px;
        font-weight: 800;
    }}

    .glass-card {{
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.25);
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
    }}

    </style>
    """,
    unsafe_allow_html=True
)
# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
        🌾 Smart Crop AI
    </div>

    <div class="subtitle">
        Intelligent Crop Recommendation &
        Agricultural Decision Support
    </div>
    """,
    unsafe_allow_html=True
)

st.caption(
    "🤖 Offline AI • 🌱 Machine Learning • "
    "🔍 SHAP • 📚 FAISS RAG • 💬 Local GenAI"
)

st.divider()




# ============================================================
# LOAD MODEL AND RESOURCES
# ============================================================

@st.cache_resource
def load_resources():

    model = joblib.load(
        "models/crop_model.pkl"
    )

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
            item.strip()
            for item in file.read().split("\n\n")
            if item.strip()
        ]

    return (
        model,
        embedding_model,
        index,
        chunks
    )


try:

    (
        model,
        embedding_model,
        index,
        chunks
    ) = load_resources()

except Exception as error:

    st.error(
        "Unable to load Smart Crop AI resources."
    )

    st.code(
        str(error)
    )

    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

if "predicted" not in st.session_state:
    st.session_state.predicted = False

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "confidence" not in st.session_state:
    st.session_state.confidence = 0.0

if "top3" not in st.session_state:
    st.session_state.top3 = []

if "input_df" not in st.session_state:
    st.session_state.input_df = None

if "context" not in st.session_state:
    st.session_state.context = ""

if "retrieved" not in st.session_state:
    st.session_state.retrieved = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ============================================================
# LANGUAGE
# ============================================================

language = st.radio(
    "Language / भाषा",
    [
        "English",
        "हिंदी"
    ],
    horizontal=True
)


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown(
    """
    <div style="
        text-align:center;
        padding:10px 0 18px 0;
    ">
        <div style="font-size:38px;">🌾</div>
        <div style="
            font-size:22px;
            font-weight:700;
        ">
            Smart Crop AI
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    "🌱 Farm Information"
)

st.sidebar.caption(
    "Enter soil and climate conditions"
)


N = st.sidebar.number_input(
    "Nitrogen (N)",
    min_value=0.0,
    value=117.0,
    step=1.0
)


P = st.sidebar.number_input(
    "Phosphorus (P)",
    min_value=0.0,
    value=46.0,
    step=1.0
)


K = st.sidebar.number_input(
    "Potassium (K)",
    min_value=0.0,
    value=19.0,
    step=1.0
)


temperature = st.sidebar.number_input(
    "Temperature (°C)",
    value=24.50,
    step=0.1
)


humidity = st.sidebar.number_input(
    "Humidity (%)",
    min_value=0.0,
    max_value=100.0,
    value=70.0,
    step=1.0
)


ph = st.sidebar.number_input(
    "Soil pH",
    min_value=0.0,
    max_value=14.0,
    value=6.0,
    step=0.1
)


rainfall = st.sidebar.number_input(
    "Rainfall (mm)",
    min_value=0.0,
    value=750.0,
    step=1.0
)


predict_button = st.sidebar.button(
    "🌾 Predict Crop",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    input_df = pd.DataFrame(
        [
            {
                "N": N,
                "P": P,
                "K": K,
                "temperature": temperature,
                "humidity": humidity,
                "ph": ph,
                "rainfall": rainfall
            }
        ]
    )


    with st.spinner(
        "Running crop prediction..."
    ):

        prediction = model.predict(
            input_df
        )[0]

        probabilities = model.predict_proba(
            input_df
        )[0]

        classes = model.classes_


    results = sorted(
        zip(
            classes,
            probabilities
        ),
        key=lambda item: item[1],
        reverse=True
    )


    top3 = results[:3]

    confidence = (
        top3[0][1] * 100
    )


    st.session_state.predicted = True

    st.session_state.prediction = prediction

    st.session_state.confidence = confidence

    st.session_state.top3 = top3

    st.session_state.input_df = input_df

    st.session_state.chat_history = []


    # ========================================================
    # FAISS RETRIEVAL
    # ========================================================

    query = f"""
    {prediction} crop requirements,
    soil requirements, water requirements,
    irrigation, nutrients, temperature,
    humidity and rainfall.
    """


    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )


    scores, indices = index.search(
        query_embedding,
        3
    )


    retrieved = []


    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if (
            idx >= 0
            and idx < len(chunks)
        ):

            retrieved.append(
                chunks[idx]
            )


    st.session_state.retrieved = retrieved

    st.session_state.context = (
        "\n\n".join(retrieved)
    )


# ============================================================
# RESULT VARIABLES
# ============================================================

if st.session_state.predicted:

    prediction = (
        st.session_state.prediction
    )

    confidence = (
        st.session_state.confidence
    )

    top3 = (
        st.session_state.top3
    )

    input_df = (
        st.session_state.input_df
    )

    context = (
        st.session_state.context
    )

    retrieved = (
        st.session_state.retrieved
    )

# ========================================================
    # FARM SUMMARY
    # ========================================================

    st.header(
        "🌱 Farm Summary"
    )

    st.caption(
    "Your current soil and weather conditions"
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Nitrogen (N)",
            f"{N:.2f}"
        )


    with col2:

        st.metric(
            "Phosphorus (P)",
            f"{P:.2f}"
        )


    with col3:

        st.metric(
            "Potassium (K)",
            f"{K:.2f}"
        )


    with col4:

        st.metric(
            "Soil pH",
            f"{ph:.2f}"
        )


    col5, col6, col7 = st.columns(3)


    with col5:

        st.metric(
            "Temperature",
            f"{temperature:.2f} °C"
        )


    with col6:

        st.metric(
            "Humidity",
            f"{humidity:.2f}%"
        )


    with col7:

        st.metric(
            "Rainfall",
            f"{rainfall:.2f} mm"
        )


    st.divider()


    # ========================================================
    # CROP RECOMMENDATION
    # ========================================================

    st.header(
        "🌾 Crop Recommendation"
    )


    result_col1, result_col2 = st.columns(2)


    with result_col1:

        st.subheader("🌾 Recommended Crop")

        st.success(
            f"{prediction.title()} — {confidence:.2f}%"
        )

        st.caption(
            "Based on your soil and climate conditions."
        )
        
        st.metric(
            "Model Probability",
            f"{confidence:.2f}%"
        )


    with result_col2:

        st.subheader(
            "Top 3 Crop Predictions"
        )


        for number, (
            crop,
            probability
        ) in enumerate(
            top3,
            start=1
        ):

            percentage = (
                probability * 100
            )


            st.write(
                f"**{number}. {crop.title()}** "
                f"— {percentage:.2f}%"
            )


            st.progress(
                float(probability)
            )


    # ========================================================
    # TOP 3 LIGHT GREEN BAR CHART
    # ========================================================

    st.subheader(
        "📊 Crop Prediction Comparison"
    )


    chart_df = pd.DataFrame(
        {
            "Crop": [
                crop.title()
                for crop, probability in top3
            ],

            "Probability (%)": [
                probability * 100
                for crop, probability in top3
            ]
        }
    )


    fig = px.bar(
        chart_df,
        x="Crop",
        y="Probability (%)",
        text="Probability (%)"
    )


    fig.update_traces(
        marker_color="#90EE90",
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )


    fig.update_layout(
        yaxis_title="Probability (%)",
        xaxis_title="Crop",
        yaxis_range=[
            0,
            105
        ],
        showlegend=False
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ========================================================
    # CROP COMPARISON TABLE
    # ========================================================

    st.subheader(
        "🌾 Crop Comparison"
    )


    comparison_df = pd.DataFrame(
        {
            "Rank": range(
                1,
                len(top3) + 1
            ),

            "Crop": [
                crop.title()
                for crop, probability in top3
            ],

            "Model Probability (%)": [
                probability * 100
                for crop, probability in top3
            ]
        }
    )


    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )


    st.caption(
        "Higher model probability means the model "
        "considered that crop more suitable for "
        "the supplied farm conditions."
    )


    # ========================================================
    # UNCERTAINTY
    # ========================================================

    if confidence < 50:

        st.warning(
            f"⚠️ Model probability: "
            f"{confidence:.2f}%. "
            "This is a relatively uncertain prediction."
        )


    elif confidence < 70:

        st.info(
            f"ℹ️ Model probability: "
            f"{confidence:.2f}%. "
            "The prediction has some uncertainty."
        )


    else:

        st.success(
            f"Model probability: "
            f"{confidence:.2f}%"
        )


    if len(top3) >= 2:

        difference = (
            top3[0][1]
            -
            top3[1][1]
        ) * 100


        if difference < 10:

            st.warning(
                "⚠️ The top two crops have similar "
                "model probabilities. Treat the "
                "prediction with caution."
            )


    st.divider()

# ========================================================
    # SHAP EXPLANATION
    # ========================================================

    st.header(
        "🔍 Why did the model choose this crop?"
    )

    st.caption(
    "Understand which farm conditions influenced "
    "the model's recommendation."
    )


    try:

        explainer = shap.TreeExplainer(
            model
        )


        shap_values = explainer.shap_values(
            input_df
        )


        if isinstance(
            shap_values,
            list
        ):

            class_index = list(
                model.classes_
            ).index(
                prediction
            )

            values = np.asarray(
                shap_values[class_index]
            )

        else:

            values = np.asarray(
                shap_values
            )


        values = np.squeeze(
            values
        )


        feature_count = len(
            input_df.columns
        )


        if values.ndim == 2:

            if values.shape[1] == feature_count:

                values = values[0]

            elif values.shape[0] == feature_count:

                values = values[:, 0]


        values = values.flatten()


        if len(values) == feature_count:

            shap_df = pd.DataFrame(
                {
                    "Feature": input_df.columns,
                    "SHAP Impact": values
                }
            )


            shap_df["Absolute Impact"] = (
                shap_df["SHAP Impact"].abs()
            )


            shap_df = shap_df.sort_values(
                "Absolute Impact",
                ascending=False
            )


            st.subheader(
                "SHAP Feature Impact"
            )


            for _, row in shap_df.iterrows():

                feature = row["Feature"]

                impact = float(
                    row["SHAP Impact"]
                )


                if impact > 0:

                    direction = (
                        "↑ supports the model prediction"
                    )

                elif impact < 0:

                    direction = (
                        "↓ reduces the model prediction"
                    )

                else:

                    direction = (
                        "→ little influence"
                    )


                st.write(
                    f"**{feature}** "
                    f"`{impact:+.4f}` "
                    f"{direction}"
                )


            # ------------------------------------------------
            # SHAP LIGHT GREEN GRAPH
            # ------------------------------------------------

            st.subheader(
                "📊 SHAP Impact Graph"
            )


            shap_chart = shap_df[
                [
                    "Feature",
                    "SHAP Impact"
                ]
            ].copy()


            shap_fig = px.bar(
                shap_chart,
                x="SHAP Impact",
                y="Feature",
                orientation="h",
                text="SHAP Impact"
            )


            shap_fig.update_traces(
                marker_color="#90EE90",
                texttemplate="%{text:.4f}",
                textposition="outside"
            )


            shap_fig.update_layout(
                xaxis_title="SHAP Impact",
                yaxis_title="Feature",
                showlegend=False
            )


            st.plotly_chart(
                shap_fig,
                use_container_width=True
            )


            st.caption(
                "SHAP shows how each input feature "
                "influenced this model prediction. "
                "It does not prove crop success."
            )


        else:

            st.info(
                "SHAP explanation could not be "
                "matched with the model features."
            )


    except Exception as error:

        st.warning(
            "SHAP explanation is temporarily unavailable."
        )

        st.caption(
            str(error)
        )


    st.divider()


    # ========================================================
    # FARM INPUT ANALYTICS
    # ========================================================

    st.header(
        "📊 Farm Input Analytics"
    )

    st.caption(
    "Visual overview of the soil and climate inputs "
    "used by the AI model."
    )


    # --------------------------------------------------------
    # NPK GRAPH
    # --------------------------------------------------------

    st.subheader(
        "🌱 Soil Nutrients"
    )


    npk_df = pd.DataFrame(
        {
            "Nutrient": [
                "Nitrogen (N)",
                "Phosphorus (P)",
                "Potassium (K)"
            ],

            "Value": [
                N,
                P,
                K
            ]
        }
    )


    npk_fig = px.bar(
        npk_df,
        x="Nutrient",
        y="Value",
        text="Value"
    )


    npk_fig.update_traces(
        marker_color="#90EE90",
        texttemplate="%{text:.2f}",
        textposition="outside"
    )


    npk_fig.update_layout(
        yaxis_title="Value",
        xaxis_title="Nutrient",
        showlegend=False
    )


    st.plotly_chart(
        npk_fig,
        use_container_width=True
    )


    # --------------------------------------------------------
    # WEATHER GRAPH
    # --------------------------------------------------------

    st.subheader(
        "🌦️ Weather Conditions"
    )


    weather_df = pd.DataFrame(
        {
            "Condition": [
                "Temperature (°C)",
                "Humidity (%)",
                "Rainfall (mm)"
            ],

            "Value": [
                temperature,
                humidity,
                rainfall
            ]
        }
    )


    weather_fig = px.bar(
        weather_df,
        x="Condition",
        y="Value",
        text="Value"
    )


    weather_fig.update_traces(
        marker_color="#90EE90",
        texttemplate="%{text:.2f}",
        textposition="outside"
    )


    weather_fig.update_layout(
        yaxis_title="Value",
        xaxis_title="Condition",
        showlegend=False
    )


    st.plotly_chart(
        weather_fig,
        use_container_width=True
    )


    # --------------------------------------------------------
    # SOIL PH
    # --------------------------------------------------------

    st.subheader(
        "🧪 Soil pH"
    )


    ph_df = pd.DataFrame(
        {
            "Parameter": [
                "Soil pH"
            ],

            "Value": [
                ph
            ]
        }
    )


    ph_fig = px.bar(
        ph_df,
        x="Parameter",
        y="Value",
        text="Value"
    )


    ph_fig.update_traces(
        marker_color="#90EE90",
        texttemplate="%{text:.2f}",
        textposition="outside"
    )


    ph_fig.update_layout(
        yaxis_title="pH",
        xaxis_title="",
        yaxis_range=[
            0,
            14
        ],
        showlegend=False
    )


    st.plotly_chart(
        ph_fig,
        use_container_width=True
    )


    st.divider()

# ========================================================
    # LOCATION / CLIMATE INFORMATION
    # ========================================================

    st.header(
        "📍 Location & Climate Information"
    )

    st.write(
        "Enter your farm location to provide "
        "additional local climate context."
    )


    location_name = st.text_input(
        "Farm Location",
        placeholder="Example: Indore, Madhya Pradesh"
    )


    if location_name.strip():

        st.success(
            f"📍 Farm location: {location_name}"
        )


        st.info(
            "Climate conditions can vary by location "
            "and season. Use the measured temperature, "
            "humidity and rainfall values above together "
            "with local agricultural guidance."
        )


        climate_col1, climate_col2, climate_col3 = (
            st.columns(3)
        )


        with climate_col1:

            st.metric(
                "Temperature",
                f"{temperature:.2f} °C"
            )


        with climate_col2:

            st.metric(
                "Humidity",
                f"{humidity:.2f}%"
            )


        with climate_col3:

            st.metric(
                "Rainfall",
                f"{rainfall:.2f} mm"
            )


        st.caption(
            "The location is recorded for report/context "
            "purposes. The current crop prediction is "
            "based on the numerical farm inputs supplied "
            "above."
        )

    else:

        st.info(
            "Enter a location to see the location "
            "and climate section."
        )


    st.divider()


    # ========================================================
    # AGRICULTURAL KNOWLEDGE
    # ========================================================

    st.header(
        "📚 Agricultural Knowledge"
    )


    with st.expander(
        "View retrieved FAISS knowledge"
    ):

        if retrieved:

            for number, text in enumerate(
                retrieved,
                start=1
            ):

                st.markdown(
                    f"### Source {number}"
                )

                st.write(
                    text
                )

                st.divider()

        else:

            st.info(
                "No relevant agricultural knowledge "
                "was retrieved."
            )


    st.divider()


    # ========================================================
    # LOCAL OLLAMA AI ADVISORY
    # ========================================================

    st.header(
        "🤖 Local AI Advisory"
    )


    if language == "English":

        advisory_prompt = f"""
You are Smart Crop AI.

Recommended crop: {prediction}

Model probability: {confidence:.2f}%

Top 3 predictions:
{top3}

Farm conditions:

Nitrogen: {N}
Phosphorus: {P}
Potassium: {K}
Temperature: {temperature} °C
Humidity: {humidity} %
Soil pH: {ph}
Rainfall: {rainfall} mm

Retrieved agricultural knowledge:

{context}

Give a concise farmer-friendly advisory.

Use these sections:

Why This Crop
Water / Irrigation
Soil / Nutrients
Climate
Important Precautions

Rules:

- Model probability is not a guarantee.
- Never say 100 minus confidence is the failure probability.
- Never call probability below 50% high confidence.
- Do not guarantee crop success, yield or profit.
- Do not invent agricultural requirements.
- Do not prescribe exact fertilizer quantities.
- Do not prescribe exact pesticide quantities.
- Recommend soil testing before fertilizer decisions.
- Encourage locally recommended agricultural practices.
- Consider local climate, soil and pest/disease conditions.
- SHAP explains model feature influence.
- SHAP does not prove crop success.
"""

    else:

        advisory_prompt = f"""
आप Smart Crop AI हैं।

अनुशंसित फसल: {prediction}

मॉडल probability: {confidence:.2f}%

शीर्ष 3 भविष्यवाणियां:
{top3}

खेत की जानकारी:

नाइट्रोजन: {N}
फॉस्फोरस: {P}
पोटैशियम: {K}
तापमान: {temperature} °C
आर्द्रता: {humidity} %
मिट्टी का pH: {ph}
वर्षा: {rainfall} mm

कृषि जानकारी:

{context}

सरल और किसान के लिए उपयोगी हिंदी में सलाह दें।

इन भागों में उत्तर दें:

यह फसल क्यों चुनी गई
पानी / सिंचाई
मिट्टी / पोषक तत्व
जलवायु
महत्वपूर्ण सावधानियां

नियम:

- मॉडल probability सफलता की गारंटी नहीं है।
- 100 minus confidence को failure probability न बताएं।
- 50% से कम probability को high confidence न कहें।
- फसल, उपज या लाभ की गारंटी न दें।
- कृषि तथ्य स्वयं से न बनाएं।
- उर्वरक की सटीक मात्रा न बताएं।
- कीटनाशक की सटीक मात्रा न बताएं।
- उर्वरक निर्णय से पहले मिट्टी की जांच की सलाह दें।
- स्थानीय कृषि सिफारिशों का पालन करने की सलाह दें।
- SHAP मॉडल feature के प्रभाव को बताता है।
- SHAP फसल की सफलता साबित नहीं करता।
"""


    with st.spinner(
        "🤖 Generating local AI advisory..."
    ):

        try:

            response = ollama.chat(
                model="qwen2.5:1.5b",
                messages=[
                    {
                        "role": "user",
                        "content": advisory_prompt
                    }
                ]
            )


            advisory = response[
                "message"
            ][
                "content"
            ]


            st.write(
                advisory
            )


        except Exception as error:

            st.error(
                "Local Ollama could not generate "
                "the advisory."
            )

            st.caption(
                str(error)
            )


    st.divider()

# ========================================================
    # ASK SMART CROP AI
    # ========================================================

    st.header(
        "💬 Ask Smart Crop AI"
    )

    st.caption(
        "Ask questions about your crop recommendation, "
        "soil, irrigation, nutrients or climate."
    )


    if language == "English":

        default_question = (
            "Why did the model choose this crop?"
        )

    else:

        default_question = (
            "मॉडल ने इस फसल को क्यों चुना?"
        )


    question = st.text_input(
        "Your question",
        placeholder=default_question
    )

    voice_question = speech_to_text(
    language="en",
    start_prompt="🎙️ Record Voice",
    stop_prompt="⏹️ Stop Recording",
    just_once=True,
    use_container_width=True,
    key="voice_question"
    
    )

    if voice_question:
        question = voice_question
        st.success(
            f"🎙️ Voice question: {question}"
    )


    if st.button(
        "🤖 Ask Smart Crop AI",
        use_container_width=True
    ):

        if not question.strip():

            st.warning(
                "Please enter a question first."
            )

        else:

            if language == "English":

                chat_prompt = f"""
You are Smart Crop AI.

Recommended crop:
{prediction}

Model probability:
{confidence:.2f}%

Top 3 predictions:
{top3}

Farm conditions:

Nitrogen: {N}
Phosphorus: {P}
Potassium: {K}
Temperature: {temperature} °C
Humidity: {humidity} %
Soil pH: {ph}
Rainfall: {rainfall} mm

Retrieved agricultural knowledge:

{context}

Farmer question:

{question}

Answer in simple, practical language.

Rules:

- Use the supplied farm information.
- Use retrieved knowledge when relevant.
- Do not invent facts.
- Model probability is not a guarantee.
- Do not say that 100 minus confidence is failure probability.
- Do not call a probability below 50% high confidence.
- Do not guarantee crop success, yield or profit.
- Do not prescribe exact fertilizer or pesticide quantities.
- Recommend soil testing before fertilizer decisions.
- Consider local climate and agricultural conditions.
"""

            else:

                chat_prompt = f"""
आप Smart Crop AI हैं।

अनुशंसित फसल:
{prediction}

मॉडल probability:
{confidence:.2f}%

शीर्ष 3 भविष्यवाणियां:
{top3}

खेत की जानकारी:

नाइट्रोजन: {N}
फॉस्फोरस: {P}
पोटैशियम: {K}
तापमान: {temperature} °C
आर्द्रता: {humidity} %
मिट्टी का pH: {ph}
वर्षा: {rainfall} mm

कृषि जानकारी:

{context}

किसान का प्रश्न:

{question}

सरल हिंदी में उपयोगी उत्तर दें।

नियम:

- दी गई खेत की जानकारी का उपयोग करें।
- उपलब्ध कृषि जानकारी का उपयोग करें।
- तथ्य न बनाएं।
- मॉडल probability सफलता की गारंटी नहीं है।
- 100 minus confidence को failure probability न बताएं।
- 50% से कम probability को high confidence न कहें।
- फसल, उपज या लाभ की गारंटी न दें।
- उर्वरक या कीटनाशक की सटीक मात्रा न बताएं।
- उर्वरक निर्णय से पहले मिट्टी की जांच की सलाह दें।
"""


            with st.spinner(
                "🤖 Thinking..."
            ):

                try:

                    answer = ollama.chat(
                        model="qwen2.5:1.5b",
                        messages=[
                            {
                                "role": "user",
                                "content": chat_prompt
                            }
                        ]
                    )


                    answer_text = answer[
                        "message"
                    ][
                        "content"
                    ]


                    st.session_state.chat_history.append(
                        (
                            question,
                            answer_text
                        )
                    )


                except Exception as error:

                    st.error(
                        "Unable to connect to local Ollama."
                    )

                    st.caption(
                        str(error)
                    )


    # --------------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------------

    if st.session_state.chat_history:

        st.subheader(
            "📝 Conversation"
        )


        for question_text, answer_text in reversed(
            st.session_state.chat_history
        ):

            st.markdown(
                f"**You:** {question_text}"
            )

            st.markdown(
                f"**Smart Crop AI:** {answer_text}"
            )

            st.divider()


    # ========================================================
    # DOWNLOAD FARM REPORT
    # ========================================================

    st.header(
        "📥 Farm Report"
    )

    st.caption(
    "Generate a PDF summary of your farm conditions, "
    "crop recommendation and model performance."
    )


    def create_farm_report():

        file_path = (
            "smart_crop_farm_report.pdf"
        )


        pdf = canvas.Canvas(
            file_path,
            pagesize=A4
        )


        width, height = A4

        y = height - 50


        pdf.setFont(
            "Helvetica-Bold",
            22
        )

        pdf.drawString(
            50,
            y,
            "Smart Crop AI - Farm Report"
        )


        y -= 30


        pdf.setFont(
            "Helvetica",
            10
        )

        pdf.drawString(
            50,
            y,
            "Offline AI-powered agricultural "
            "decision-support system"
        )


        y -= 35


        pdf.setFont(
            "Helvetica-Bold",
            14
        )

        pdf.drawString(
            50,
            y,
            "Farm Information"
        )


        y -= 25


        pdf.setFont(
            "Helvetica",
            10
        )


        farm_data = [
            f"Nitrogen (N): {N:.2f}",
            f"Phosphorus (P): {P:.2f}",
            f"Potassium (K): {K:.2f}",
            f"Temperature: {temperature:.2f} C",
            f"Humidity: {humidity:.2f}%",
            f"Soil pH: {ph:.2f}",
            f"Rainfall: {rainfall:.2f} mm"
        ]


        for item in farm_data:

            pdf.drawString(
                60,
                y,
                item
            )

            y -= 18


        y -= 15


        pdf.setFont(
            "Helvetica-Bold",
            14
        )

        pdf.drawString(
            50,
            y,
            "Crop Recommendation"
        )


        y -= 25


        pdf.setFont(
            "Helvetica",
            10
        )


        pdf.drawString(
            60,
            y,
            f"Recommended Crop: "
            f"{prediction.title()}"
        )


        y -= 18


        pdf.drawString(
            60,
            y,
            f"Model Probability: "
            f"{confidence:.2f}%"
        )


        y -= 30


        pdf.setFont(
            "Helvetica-Bold",
            14
        )

        pdf.drawString(
            50,
            y,
            "Top 3 Crop Predictions"
        )


        y -= 25


        pdf.setFont(
            "Helvetica",
            10
        )


        for number, (
            crop,
            probability
        ) in enumerate(
            top3,
            start=1
        ):

            pdf.drawString(
                60,
                y,
                f"{number}. "
                f"{crop.title()} - "
                f"{probability * 100:.2f}%"
            )

            y -= 18


        y -= 15


        # ----------------------------------------------------
        # MODEL PERFORMANCE
        # ----------------------------------------------------

        try:

            performance = joblib.load(
                "models/model_performance.pkl"
            )


            pdf.setFont(
                "Helvetica-Bold",
                14
            )

            pdf.drawString(
                50,
                y,
                "Model Performance"
            )


            y -= 25


            pdf.setFont(
                "Helvetica",
                10
            )


            metrics = [
                (
                    "Accuracy",
                    performance["accuracy"]
                ),
                (
                    "Precision",
                    performance["precision"]
                ),
                (
                    "Recall",
                    performance["recall"]
                ),
                (
                    "F1-Score",
                    performance["f1_score"]
                )
            ]


            for name, value in metrics:

                pdf.drawString(
                    60,
                    y,
                    f"{name}: "
                    f"{value * 100:.2f}%"
                )

                y -= 18


        except Exception:

            pdf.drawString(
                60,
                y,
                "Performance data unavailable."
            )

            y -= 18


        y -= 15


        # ----------------------------------------------------
        # DISCLAIMER
        # ----------------------------------------------------

        pdf.setFont(
            "Helvetica-Bold",
            12
        )

        pdf.drawString(
            50,
            y,
            "Important Precautions"
        )


        y -= 20


        pdf.setFont(
            "Helvetica",
            9
        )


        precautions = [
            "This report provides decision support,",
            "not a guarantee of crop success.",
            "Soil testing is recommended before",
            "fertilizer decisions.",
            "Follow locally recommended agricultural",
            "practices and local expert guidance."
        ]


        for line in precautions:

            pdf.drawString(
                60,
                y,
                line
            )

            y -= 14


        pdf.save()


        return file_path


    report_file = create_farm_report()


    with open(
        report_file,
        "rb"
    ) as file:

        st.download_button(
            label="📥 Download Farm Report",
            data=file,
            file_name="smart_crop_farm_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )


    # ========================================================
    # SYSTEM PERFORMANCE
    # ========================================================

    st.divider()

    st.header(
        "📊 System Performance"
    )

    st.caption(
    "Components currently active in the Smart Crop AI pipeline."
    )


    perf1, perf2, perf3, perf4 = st.columns(4)


    with perf1:

        st.metric(
            "Machine Learning",
            "✓ Active"
        )


    with perf2:

        st.metric(
            "SHAP",
            "✓ Active"
        )


    with perf3:

        st.metric(
            "FAISS RAG",
            "✓ Active"
        )


    with perf4:

        st.metric(
            "Local Ollama",
            "✓ Active"
        )


    st.write(
        "Smart Crop AI combines Machine Learning, "
        "SHAP explainability, FAISS-based retrieval "
        "and a local Ollama language model."
    )


    # ========================================================
    # AI PIPELINE
    # ========================================================

    st.subheader(
        "🔄 AI Pipeline"
    )


    pipeline_steps = [
        "🌱 Farm Data",
        "🧠 ML Prediction",
        "📊 Top-3 Crops",
        "🔍 SHAP",
        "📚 FAISS RAG",
        "🤖 Local Ollama",
        "💬 AI Chat"
    ]


    st.write(
        " → ".join(
            pipeline_steps
        )
    )


    # ========================================================
    # IMPORTANT PRECAUTIONS
    # ========================================================

    st.divider()

    st.header(
        "⚠️ Important Precautions"
    )


    st.warning(
        "This system provides decision support, "
        "not a guarantee of crop success."
    )


    st.info(
        "🌱 Soil testing is recommended before "
        "fertilizer decisions."
    )


    st.info(
        "👨‍🌾 Follow locally recommended agricultural "
        "practices and consider local climate, soil "
        "and pest/disease conditions."
    )


    st.caption(
        "SHAP explains how input features influenced "
        "the ML model prediction. SHAP does not prove "
        "that a crop will definitely grow successfully."
    )


    # ========================================================
    # FOOTER
    # ========================================================

    st.divider()

    st.caption(
        "🌾 Smart Crop AI • Offline ML + RAG + "
        "SHAP + Local Ollama"
    )
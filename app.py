from pathlib import Path
import json

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "agrisentinel_rice_efficientnet.pth"
)

CLASSES_PATH = (
    PROJECT_DIR
    / "models"
    / "classes.json"
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AgriSentinel",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background: #f7faf7;
    }

    /* Header */
    .main-header {
        background: linear-gradient(
            135deg,
            #1f6b3a,
            #3d8b4d
        );
        padding: 28px 35px;
        border-radius: 18px;
        color: white;
        margin-bottom: 25px;
    }

    .main-header h1 {
        font-size: 42px;
        margin: 0;
        color: white;
    }

    .main-header p {
        font-size: 18px;
        margin-top: 8px;
        color: #edf8ef;
    }

    /* Section headers */
    .section-header {
        font-size: 25px;
        font-weight: 700;
        color: #245c35;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    /* Cards */
    .info-card {
        background: white;
        border: 1px solid #dce9df;
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 10px;
    }

    .info-card h4 {
        color: #245c35;
        margin-top: 0;
    }

    /* Result box */
    .result-box {
        background: #edf8ee;
        border: 2px solid #9acaa1;
        border-radius: 18px;
        padding: 25px;
        text-align: center;
    }

    .result-title {
        font-size: 30px;
        font-weight: 800;
        color: #245c35;
    }

    .result-subtitle {
        font-size: 17px;
        color: #4b6852;
        margin-top: 5px;
    }

    /* Evidence */
    .evidence-positive {
        background: #edf8ef;
        border-left: 5px solid #3c9b52;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 8px;
    }

    .evidence-pending {
        background: #fff8e8;
        border-left: 5px solid #e0a72f;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 8px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #718477;
        padding: 30px 0 10px 0;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    with open(
        CLASSES_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        classes = json.load(f)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = models.efficientnet_b0(
        weights=None
    )

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(classes)
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(
        state_dict
    )

    model = model.to(device)

    model.eval()

    return model, classes, device


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(
    model,
    classes,
    device,
    image
):

    transform = transforms.Compose([

        transforms.Resize(
            (224, 224)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ]
        )
    ])

    image = image.convert("RGB")

    image_tensor = transform(
        image
    ).unsqueeze(0)

    image_tensor = image_tensor.to(device)

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_index = torch.max(
            probabilities,
            dim=1
        )

    prediction = classes[
        predicted_index.item()
    ]

    confidence_value = (
        confidence.item() * 100
    )

    # Top predictions
    top_probabilities, top_indices = torch.topk(
        probabilities[0],
        min(3, len(classes))
    )

    top_predictions = []

    for probability, index in zip(
        top_probabilities,
        top_indices
    ):

        top_predictions.append(
            (
                classes[index.item()],
                probability.item() * 100
            )
        )

    return (
        prediction,
        confidence_value,
        top_predictions
    )

# ============================================================
# PROTOTYPE AGRICULTURAL EVIDENCE
# ============================================================

def get_demo_agricultural_data(prediction, location):

    return {

        # ----------------------------------------------------
        # WEATHER
        # ----------------------------------------------------

        "weather": {

            "rainfall": "High",
            "rainfall_7days": "68 mm",
            "humidity": "82%",
            "temperature": "29°C"
        },


        # ----------------------------------------------------
        # SOIL
        # ----------------------------------------------------

        "soil": {

            "moisture": "78%",
            "ph": "6.4",
            "condition": "Moist"
        },


        # ----------------------------------------------------
        # PREVIOUS HISTORY
        # ----------------------------------------------------

        "history": {

            "previous_cases": "2 observations",
            "previous_condition": prediction,
            "trend": "Symptoms previously observed"
        },


        # ----------------------------------------------------
        # LOCAL OUTBREAK
        # ----------------------------------------------------

        "local_outbreak": {

            "status": "Moderate",
            "nearby_cases": "Reported in the demo region"
        },


        # ----------------------------------------------------
        # SATELLITE
        # ----------------------------------------------------

        "satellite": {

            "status": "Moderate crop stress",
            "vegetation": (
                "Slight reduction in vegetation health"
            )
        },


        # ----------------------------------------------------
        # IoT SENSOR
        # ----------------------------------------------------

        "iot": {

            "soil_moisture": "78%",
            "leaf_wetness": "High",
            "sensor_status": "Online"
        },


        # ----------------------------------------------------
        # FARMER VOICE
        # ----------------------------------------------------

        "voice": {

            "language": "Tamil",

            "transcript": (
                "இலைகளில் பழுப்பு நிற புள்ளிகள் அதிகமாக "
                "காணப்படுகின்றன."
            )
        },


        # ----------------------------------------------------
        # GOVERNMENT ADVISORY
        # ----------------------------------------------------

        "government": {

            "status": "Available",

            "message": (
                "Follow validated rice disease management "
                "practices and monitor affected plants."
            )
        },


        # ----------------------------------------------------
        # AGRICULTURAL KNOWLEDGE BASE / RAG
        # ----------------------------------------------------

        "knowledge_base": {

            "status": "Retrieved",

            "source": (
                "Prototype agricultural knowledge base"
            ),

            "match": prediction
        },


        # ----------------------------------------------------
        # DEMO LABEL
        # ----------------------------------------------------

        "source_note": (
            "🧪 Prototype Demo Data"
        )
    }

# ============================================================
# DEMO AGRICULTURAL EVIDENCE
# ============================================================

def get_demo_agricultural_data(prediction, location):

    return {
        "weather": {
            "rainfall": "High",
            "rainfall_7days": "68 mm",
            "humidity": "82%",
            "temperature": "29°C"
        },

        "soil": {
            "moisture": "High",
            "ph": "6.4",
            "condition": "Moist"
        },

        "history": {
            "previous_cases": "2 observations",
            "previous_condition": prediction,
            "trend": "Symptoms previously observed"
        },

        "local_outbreak": {
            "status": "Moderate",
            "nearby_cases": "Reported in the demo region"
        },

        "government_advisory": {
            "status": "Available",
            "message": (
                "Follow validated rice disease management practices "
                "and monitor affected plants."
            )
        },

        "source_note": "🧪 Prototype Demo Data"
    }

# ============================================================
# HEADER
# ============================================================

st.title("🌾 AgriSentinel")

st.subheader(
    "Evidence-Grounded Crop Health & Verified Advisory Agent"
)

st.info(
    "💡 Advice is not the endpoint — outcome is."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🌱 Crop Context")

    crop = st.selectbox(
        "Crop",
        [
            "Rice"
        ]
    )

    growth_stage = st.selectbox(
        "Growth Stage",
        [
            "Seedling",
            "Vegetative",
            "Flowering",
            "Grain Filling"
        ]
    )

    location = st.text_input(
        "Location",
        value="Karaikal"
    )

    st.divider()

    st.markdown("### 🤖 AI System")

    st.success(
        "Model Ready"
    )

    st.caption(
        "EfficientNet-B0 based rice disease "
        "classification prototype."
    )

    st.divider()

    st.caption(
        "Prototype mode: weather, historical "
        "evidence and follow-up comparison "
        "are demonstrated concepts."
    )


# ============================================================
# CROP INVESTIGATION
# ============================================================

st.markdown(
    '<div class="section-header">📷 Crop Investigation</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(
    [1.2, 1]
)


# ------------------------------------------------------------
# IMAGE UPLOAD
# ------------------------------------------------------------

with col1:

    uploaded_file = st.file_uploader(
        "Upload a rice leaf image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        help="Upload a clear rice leaf image."
    )

    if uploaded_file:

        image = Image.open(
            uploaded_file
        )

        st.image(
            image,
            caption="Uploaded Rice Leaf",
            width=150
        )


# ------------------------------------------------------------
# CROP CONTEXT
# ------------------------------------------------------------

with col2:

    st.markdown(
        """
        <div class="info-card">

        <h4>📋 Investigation Context</h4>

        """,
        unsafe_allow_html=True
    )

    st.write(
        f"**Crop:** {crop}"
    )

    st.write(
        f"**Growth Stage:** {growth_stage}"
    )

    st.write(
        f"**Location:** {location}"
    )

    st.write(
        "**Input:** Rice leaf image"
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

if uploaded_file:

    analyze = st.button(
        "🔍  ANALYZE CROP",
        type="primary",
        use_container_width=True
    )

else:

    analyze = False


# ============================================================
# ANALYSIS
# ============================================================

if analyze and uploaded_file is not None:

    with st.spinner("AgriSentinel is investigating the crop..."):

        model, classes, device = load_model()

        (
            prediction,
            confidence,
            top_predictions
        ) = predict_image(
            model,
            classes,
            device,
            image
        )

        st.session_state["prediction"] = prediction
        st.session_state["confidence"] = confidence
        st.session_state["top_predictions"] = top_predictions

        # Reset demo workflow for a new crop analysis
        st.session_state["followup_stage"] = 1
        st.session_state["expert_review_requested"] = False
        
prediction = st.session_state.get("prediction", None)
confidence = st.session_state.get("confidence", None)
top_predictions = st.session_state.get("top_predictions", [])

if prediction is None or confidence is None:
    st.info("📷 Upload a rice leaf image and click ANALYZE CROP.")
    st.stop()


# ============================================================
# GET PROTOTYPE AGRICULTURAL EVIDENCE
# ============================================================

demo_data = get_demo_agricultural_data(
    prediction,
    location
)

# ============================================================
# ENSURE ALL PROTOTYPE EVIDENCE EXISTS
# ============================================================

demo_data["satellite"] = {
    "status": "Moderate crop stress",
    "vegetation": "Slight reduction in vegetation health"
}

demo_data["iot"] = {
    "soil_moisture": "78%",
    "leaf_wetness": "High",
    "sensor_status": "Online"
}

demo_data["voice"] = {
    "language": "Tamil",
    "transcript": (
        "இலைகளில் பழுப்பு நிற புள்ளிகள் அதிகமாக "
        "காணப்படுகின்றன.")
    }



demo_data["government"] = {
    "status": "Available",
    "message": (
        "Follow validated rice disease management "
        "practices and monitor affected plants."
    )
}

demo_data["knowledge_base"] = {
    "status": "Retrieved",
    "source": "Prototype agricultural knowledge base",
    "match": prediction
}

demo_data["source_note"] = "🧪 Prototype Demo Data"

# ========================================================
# AI RESULT
# ========================================================

st.markdown(
    '<div class="section-header">🔬 AI Investigation Result</div>',
    unsafe_allow_html=True
)

# Center AI Result
left_space, center_area, right_space = st.columns([1, 4, 1])

with center_area:

    result_col1, result_col2 = st.columns([2, 1])

    with result_col1:

        st.success(
            f"🌿 Detected Condition: {prediction}"
        )

        st.write(
            "Detected crop health condition"
        )

    with result_col2:

        st.metric(
            "Vision Confidence",
            f"{confidence:.2f}%"
        )

        if confidence >= 85:

            st.success(
                "High visual confidence"
            )

        elif confidence >= 60:

            st.warning(
                "Medium visual confidence"
            )

        else:

            st.error(
                "Low visual confidence"
            )

    # ========================================================
    # TOP PREDICTIONS
    # ========================================================

    with st.expander(
        "📊 View model prediction distribution"
    ):

        for class_name, probability in top_predictions:

            st.write(
                f"**{class_name}** — "
                f"{probability:.2f}%"
            )

            st.progress(
                min(
                    probability / 100,
                    1.0
                )
            )
    # ============================================================
# MULTI-SOURCE AGRICULTURAL EVIDENCE
# ============================================================

st.markdown(
    '<div class="section-header">🌐 Multi-Source Agricultural Evidence</div>',
    unsafe_allow_html=True
)

st.info(
    "🧪 Prototype Demo Data — AgriSentinel combines multiple "
    "evidence sources before generating its recommendation."
)


# ------------------------------------------------------------
# WEATHER
# ------------------------------------------------------------

st.markdown("### 🌧️ Weather Evidence")

weather1, weather2, weather3 = st.columns(3)

with weather1:
    st.metric(
        "7-Day Rainfall",
        demo_data["weather"]["rainfall_7days"]
    )

with weather2:
    st.metric(
        "Humidity",
        demo_data["weather"]["humidity"]
    )

with weather3:
    st.metric(
        "Temperature",
        demo_data["weather"]["temperature"]
    )


# ------------------------------------------------------------
# SOIL
# ------------------------------------------------------------

st.markdown("### 💧 Soil Evidence")

soil1, soil2, soil3 = st.columns(3)

with soil1:
    st.metric(
        "Soil Moisture",
        demo_data["soil"]["moisture"]
    )

with soil2:
    st.metric(
        "Soil pH",
        demo_data["soil"]["ph"]
    )

with soil3:
    st.write(
        f"**Condition:** {demo_data['soil']['condition']}"
    )


# ------------------------------------------------------------
# HISTORY + LOCAL RISK
# ------------------------------------------------------------

st.markdown("### 📚 Historical & Local Evidence")

history1, history2 = st.columns(2)

with history1:

    st.write(
        f"**Previous Cases:** "
        f"{demo_data['history']['previous_cases']}"
    )

    st.write(
        f"**Previous Condition:** "
        f"{demo_data['history']['previous_condition']}"
    )

    st.write(
        f"**Trend:** "
        f"{demo_data['history']['trend']}"
    )

with history2:

    st.write(
        f"**Local Disease Risk:** "
        f"{demo_data['local_outbreak']['status']}"
    )

    st.write(
        f"**Nearby Reports:** "
        f"{demo_data['local_outbreak']['nearby_cases']}"
    )


# ------------------------------------------------------------
# SATELLITE + IoT
# ------------------------------------------------------------

st.markdown("### 🛰️ Remote & Sensor Evidence")

sensor1, sensor2 = st.columns(2)

with sensor1:

    st.write(
        "**🛰️ Satellite Crop Stress**"
    )

    st.write(
        demo_data["satellite"]["status"]
    )

    st.write(
        demo_data["satellite"]["vegetation"]
    )

with sensor2:

    st.write(
        "**📡 IoT Sensor Snapshot**"
    )

    st.write(
        f"Soil Moisture: "
        f"{demo_data['iot']['soil_moisture']}"
    )

    st.write(
        f"Leaf Wetness: "
        f"{demo_data['iot']['leaf_wetness']}"
    )

    st.write(
        f"Sensor Status: "
        f"{demo_data['iot']['sensor_status']}"
    )


# ------------------------------------------------------------
# FARMER VOICE
# ------------------------------------------------------------

st.markdown("### 🗣️ Farmer Voice Input")

st.info(
    f"Language: {demo_data['voice']['language']}\n\n"
    f"Simulated Transcript: "
    f"{demo_data['voice']['transcript']}"
)

# ------------------------------------------------------------
# GOVERNMENT + KNOWLEDGE BASE
# ------------------------------------------------------------

st.markdown("### 📖 Verified Knowledge Sources")

knowledge1, knowledge2 = st.columns(2)

with knowledge1:

    st.write("🏛️ **Government Advisory**")

    st.write(
        f"Status: {demo_data['government']['status']}"
    )

    st.write(
        demo_data["government"]["message"]
    )


with knowledge2:

    st.write("📚 **Agricultural Knowledge Base / RAG**")

    st.write(
        f"Status: {demo_data['knowledge_base']['status']}"
    )

    st.write(
        f"Knowledge Match: "
        f"{demo_data['knowledge_base']['match']}"
    )


st.caption(
    demo_data["source_note"]
)


# ========================================================
# EVIDENCE LEDGER
# ========================================================

st.markdown(
    '<div class="section-header">📋 Evidence Ledger</div>',
    unsafe_allow_html=True
)

evidence1, evidence2 = st.columns(2)


with evidence1:

    st.markdown(
        f"""
        <div class="evidence-positive">

        🟢 <b>Vision Evidence</b><br>

        The uploaded image is classified as
        <b>{prediction}</b> by the trained
        vision model.

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="evidence-positive">

        🟢 <b>Crop Evidence</b><br>

        User identified the crop as rice.

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        f"""
        <div class="evidence-positive">

        🟢 <b>Growth Stage</b><br>

        Current stage: <b>{growth_stage}</b>

        </div>
        """,
        unsafe_allow_html=True
    )


with evidence2:

    st.markdown(
        f"""
        <div class="evidence-positive">

        🟢 <b>Weather Evidence</b><br>

        Rainfall: <b>{demo_data["weather"]["rainfall_7days"]}</b><br>
        Humidity: <b>{demo_data["weather"]["humidity"]}</b><br>
        Temperature: <b>{demo_data["weather"]["temperature"]}</b>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        f"""
        <div class="evidence-positive">

        🟢 <b>Historical Evidence</b><br>

        Previous observations:
        <b>{demo_data["history"]["previous_cases"]}</b><br>

        Previous condition:
        <b>{demo_data["history"]["previous_condition"]}</b>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        f"""
        <div class="evidence-positive">

        🟢 <b>Local Disease Evidence</b><br>

        Local disease risk:
        <b>{demo_data["local_outbreak"]["status"]}</b><br>

        Nearby reports:
        <b>{demo_data["local_outbreak"]["nearby_cases"]}</b>

        </div>
        """,
        unsafe_allow_html=True
    )
    # ========================================================
    # RISK ASSESSMENT
    # ========================================================

st.markdown(
        '<div class="section-header">⚠️ Risk Assessment</div>',
        unsafe_allow_html=True
    )

if confidence >= 85:

        st.success(
            f"🟢 HIGH VISUAL CONFIDENCE — "
            f"The image evidence strongly supports "
            f"{prediction}."
        )

        decision = "Proceed with evidence-grounded advisory"

elif confidence >= 60:

        st.warning(
            "🟡 MEDIUM CONFIDENCE — "
            "Additional evidence should be collected."
        )

        decision = "Ask for additional evidence"

else:

        st.error(
            "🔴 LOW CONFIDENCE — "
            "Expert verification is recommended."
        )

        decision = "Escalate for expert verification"


    # ========================================================
    # AGENT DECISION
    # ========================================================

st.markdown(
        '<div class="section-header">🤖 AgriAgent Decision</div>',
        unsafe_allow_html=True
    )

decision_col1, decision_col2 = st.columns(
        [1, 2]
    )

with decision_col1:

        st.info(
            "Decision State"
        )

with decision_col2:

        st.write(
            f"**{decision}**"
        )


    # ========================================================
# RECOMMENDED ACTION
# ========================================================

st.markdown(
    '<div class="section-header">💡 Verified Advisory</div>',
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="info-card">

    <h4>For detected condition: {prediction}</h4>

    <b>1️⃣ Immediate Action</b>

    <p>
    Inspect nearby rice plants for similar visible
    symptoms and identify the affected area.
    </p>

    <b>2️⃣ Environmental Action</b>

    <p>
    Review field moisture and environmental conditions
    that may support disease development.
    </p>

    <b>3️⃣ Monitoring Action</b>

    <p>
    Capture a new crop image after a monitoring period
    and compare the condition.
    </p>

    <b>4️⃣ Prevention</b>

    <p>
    Continue field observation and follow validated
    agricultural practices for the identified condition.
    </p>

    <b>🛡️ Safety</b>

    <p>
    Any chemical treatment must follow locally approved
    agricultural guidance and the product label.
    </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ========================================================
# FOLLOW-UP MONITORING
# ========================================================

st.markdown(
    '<div class="section-header">📈 Follow-up Monitoring</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------------
# FOLLOW-UP STATE
# --------------------------------------------------------

if "followup_stage" not in st.session_state:

    st.session_state["followup_stage"] = 1


followup_stage = st.session_state["followup_stage"]


# --------------------------------------------------------
# FOLLOW-UP MESSAGE
# --------------------------------------------------------

st.info(
    "Advice is not the endpoint. "
    "The next step is to verify whether the crop condition "
    "improves, remains stable or worsens."
)


# --------------------------------------------------------
# CURRENT STATUS
# --------------------------------------------------------

if followup_stage == 1:

    followup_status = "Pending"
    outcome_status = "Not Available"

elif followup_stage == 3:

    followup_status = "Day 3 Completed"
    outcome_status = "Improving"

else:

    followup_status = "Day 7 Completed"
    outcome_status = "Improving"


# --------------------------------------------------------
# FOLLOW-UP STATUS
# --------------------------------------------------------

follow1, follow2, follow3 = st.columns(3)


with follow1:

    st.metric(
        "Initial Assessment",
        "Day 1"
    )


with follow2:

    st.metric(
        "Follow-up",
        followup_status
    )


with follow3:

    st.metric(
        "Outcome",
        outcome_status
    )


# --------------------------------------------------------
# DEMO FOLLOW-UP ACTION
# --------------------------------------------------------

st.markdown("### 📸 Follow-up Simulation")


if followup_stage == 1:

    st.write(
        "Initial crop assessment recorded. "
        "A new image is required for comparison."
    )

    if st.button(
        "📸 Simulate Day 3 Follow-up",
        use_container_width=True
    ):

        st.session_state["followup_stage"] = 3

        st.rerun()


elif followup_stage == 3:

    st.success(
        "🟢 Day 3 follow-up completed. "
        "The simulated crop condition shows improvement."
    )

    st.write(
        "**Comparison:** Initial image → Day 3 image"
    )

    st.write(
        "**Observed trend:** Symptoms appear stable-to-improving."
    )

    if st.button(
        "📸 Simulate Day 7 Follow-up",
        use_container_width=True
    ):

        st.session_state["followup_stage"] = 7

        st.rerun()


else:

    st.success(
        "🟢 Day 7 follow-up completed. "
        "The simulated crop condition remains improved."
    )

    st.write(
        "**Initial Assessment:** Day 1"
    )

    st.write(
        "**Follow-up Assessments:** Day 3 → Day 7"
    )

    st.write(
        "**Final Demo Outcome:** Improving"
    )

    st.info(
        "AgriSentinel recommends continued observation "
        "and escalation if symptoms worsen again."
    )


st.caption(
    "🧪 Prototype Demo — Follow-up images and outcomes "
    "are simulated for hackathon demonstration."
)

# ========================================================
# EXPERT ESCALATION
# ========================================================
# ========================================================
# EXPERT ESCALATION
# ========================================================

st.markdown(
    '<div class="section-header">👨‍🌾 Expert Escalation</div>',
    unsafe_allow_html=True
)

# Demo expert-review state
if "expert_review_requested" not in st.session_state:
    st.session_state["expert_review_requested"] = False


if confidence < 60:

    st.error(
        "⚠️ AI confidence is low. Expert verification is recommended."
    )

else:

    st.info(
        "Expert review can be requested when the available "
        "evidence is insufficient or conflicting."
    )


# --------------------------------------------------------
# REQUEST EXPERT REVIEW
# --------------------------------------------------------

if st.button(
    "📩 Request Expert Review",
    use_container_width=True
):

    st.session_state["expert_review_requested"] = True


# --------------------------------------------------------
# DEMO EXPERT WORKFLOW
# --------------------------------------------------------

if st.session_state["expert_review_requested"]:

    st.success(
        "✅ Expert review request submitted successfully."
    )

    st.markdown("### 📋 Expert Case")

    case_col1, case_col2, case_col3 = st.columns(3)

    with case_col1:

        st.write("**Case ID**")
        st.write("AGR-DEMO-001")

    with case_col2:

        st.write("**Detected Condition**")
        st.write(prediction)

    with case_col3:

        st.write("**AI Confidence**")
        st.write(f"{confidence:.2f}%")


    st.markdown("### 🔎 Evidence Sent to Expert")

    evidence_col1, evidence_col2 = st.columns(2)

    with evidence_col1:

        st.write("🌿 **Crop:** Rice")

        st.write(
            f"🔬 **Vision Result:** {prediction}"
        )

        st.write(
            f"📊 **Vision Confidence:** {confidence:.2f}%"
        )

        st.write(
            f"📍 **Location:** {location}"
        )

    with evidence_col2:

        st.write("🌧️ **Weather:** High rainfall")

        st.write("💧 **Soil Moisture:** High")

        st.write("🦠 **Local Disease Risk:** Moderate")

        st.write("📚 **Knowledge Check:** Completed")


    st.markdown("### 👨‍🌾 Expert Review")

    st.info(
        "🧪 Demo Mode — This is a simulated expert response "
        "for hackathon demonstration."
    )

    st.success(
        f"Expert Assessment: The visual symptoms are consistent "
        f"with **{prediction}**. The farmer should continue "
        "monitoring the affected plants and follow the recommended "
        "crop-management practices."
    )


    st.markdown("### ✅ Expert Recommendation")

    recommendation_col1, recommendation_col2 = st.columns(2)

    with recommendation_col1:

        st.write(
            "🔹 Continue monitoring affected leaves"
        )

        st.write(
            "🔹 Avoid unnecessary chemical application"
        )

        st.write(
            "🔹 Follow locally approved agricultural practices"
        )

    with recommendation_col2:

        st.write(
            "🔹 Recheck the crop after 3 days"
        )

        st.write(
            "🔹 Capture a new image for comparison"
        )

        st.write(
            "🔹 Escalate again if symptoms spread")


    st.markdown("### 🔄 Case Status")

    status_col1, status_col2, status_col3 = st.columns(3)

    with status_col1:

        st.metric(
            "Request",
            "Submitted"
        )

    with status_col2:

        st.metric(
            "Expert Review",
            "Completed"
        )

    with status_col3:

        st.metric(
            "Next Step",
            "Follow-up"
        )

    st.success(
        "🟢 Expert verification completed. "
        "AgriSentinel now moves the farmer to follow-up monitoring."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

    🌾 <b>AgriSentinel Prototype</b><br>

    Evidence-Grounded Crop Health &
    Verified Advisory Agent<br><br>

    Prototype for AI for Sustainable Rural &
    Agricultural Development

    </div>
    """,
    unsafe_allow_html=True
)
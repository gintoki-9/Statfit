import streamlit as st

st.set_page_config(
    page_title="StatFit",
    page_icon="💪",
    layout="wide"
)

st.title("StatFit")
st.subheader(
    "Probability-Based Lifestyle Disease Risk Assessment "
    "and Personalized Health Recommendation System"
)

st.info(
    "Review 1 Prototype — This application demonstrates the proposed "
    "structure and workflow of the StatFit system."
)

# ---------------- SIDEBAR ----------------

st.sidebar.title("StatFit")

page = st.sidebar.radio(
    "Navigation",
    [
        "Health Assessment",
        "Risk Dashboard",
        "Recommendations",
        "About"
    ]
)

# ---------------- HEALTH ASSESSMENT ----------------

if page == "Health Assessment":

    st.header("Health & Lifestyle Assessment")

    st.write(
        "Enter the user's health, nutrition and lifestyle information."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Personal Information")

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=20
        )

        height = st.number_input(
            "Height (cm)",
            min_value=100.0,
            max_value=230.0,
            value=178.0
        )

        weight = st.number_input(
            "Weight (kg)",
            min_value=30.0,
            max_value=250.0,
            value=74.0
        )

        body_fat = st.number_input(
            "Body Fat (%)",
            min_value=2.0,
            max_value=60.0,
            value=15.0
        )

    with col2:

        st.subheader("Lifestyle Information")

        calories = st.number_input(
            "Average Daily Calories",
            min_value=500,
            max_value=6000,
            value=2200
        )

        protein = st.number_input(
            "Protein (g/day)",
            min_value=0.0,
            max_value=400.0,
            value=100.0
        )

        activity = st.selectbox(
            "Average Activity Level",
            [
                "Sedentary",
                "Lightly Active",
                "Moderately Active",
                "Very Active"
            ]
        )

        sleep = st.number_input(
            "Average Sleep (hours)",
            min_value=0.0,
            max_value=16.0,
            value=7.0
        )

    medical_history = st.multiselect(
        "Relevant Medical History",
        [
            "Diabetes",
            "Hypertension",
            "High Cholesterol",
            "Cardiovascular Disease",
            "None"
        ]
    )

    if st.button("Analyze Health Profile"):

        height_m = height / 100

        bmi = weight / (height_m ** 2)

        st.success("Health profile processed successfully.")

        st.divider()

        st.subheader("Calculated Health Metrics")

        m1, m2, m3, m4 = st.columns(4)

        m1.metric(
            "BMI",
            f"{bmi:.1f}"
        )

        m2.metric(
            "Body Fat",
            f"{body_fat:.1f}%"
        )

        m3.metric(
            "Calories",
            f"{calories:.0f} kcal"
        )

        m4.metric(
            "Sleep",
            f"{sleep:.1f} hrs"
        )

        st.session_state["bmi"] = bmi


# ---------------- RISK DASHBOARD ----------------

elif page == "Risk Dashboard":

    st.header("Probability-Based Risk Dashboard")

    st.write(
        "This section will contain the statistical disease-risk "
        "prediction module."
    )

    st.warning(
        "Risk probabilities are currently placeholders for the "
        "Review 1 prototype. A validated statistical model will "
        "be implemented in a later phase."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Diabetes Risk",
        "—"
    )

    col2.metric(
        "Hypertension Risk",
        "—"
    )

    col3.metric(
        "Cardiovascular Risk",
        "—"
    )

    st.subheader("Risk Factors")

    st.write(
        """
        The final model will consider factors such as:

        • Age  
        • BMI  
        • Body fat percentage  
        • Calorie intake  
        • Macronutrient distribution  
        • Physical activity  
        • Sleep  
        • Medical history
        """
    )


# ---------------- RECOMMENDATIONS ----------------

elif page == "Recommendations":

    st.header("Personalized Health Recommendations")

    st.write(
        "This module will generate recommendations based on "
        "the calculated health metrics and predicted risks."
    )

    st.subheader("Diet")

    st.write(
        "🥗 Personalized dietary recommendations will appear here."
    )

    st.subheader("Exercise")

    st.write(
        "🏃 Personalized exercise recommendations will appear here."
    )

    st.subheader("Lifestyle")

    st.write(
        "😴 Sleep, activity and lifestyle recommendations will appear here."
    )


# ---------------- ABOUT ----------------

elif page == "About":

    st.header("About StatFit")

    st.write(
        """
        StatFit is a proposed probability-based lifestyle disease
        risk assessment and personalized health recommendation system.

        The planned system pipeline is:

        User Data
        ↓
        Data Validation
        ↓
        Health Metric Calculation
        ↓
        Lifestyle Analysis
        ↓
        Statistical Risk Prediction
        ↓
        Personalized Recommendations
        ↓
        Dashboard & Health Report
        """
    )

    st.info(
        "StatFit is an academic prototype and is not intended "
        "to provide medical diagnosis or treatment."
    )
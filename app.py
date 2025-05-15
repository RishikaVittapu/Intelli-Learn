import streamlit as st
import pickle
import json
import pandas as pd
import base64

# Load model
model = pickle.load(open("AssessScore.pkl", "rb"))

# Load recommendation text with utf-8 encoding
with open("recommendations.json", "r", encoding="utf-8") as f:
    recommendations = json.load(f)

# Set background
def set_bg():
    with open("back.png", "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_bg()

# Tabs
tabs = st.tabs(["Intelligent Tutor", "Recommendations"])

# --- Intelligent Tutor Tab ---
with tabs[0]:
    st.title("Intelligent Tutoring System")
    st.markdown("Welcome to the AI-powered learning assistant!")

    st.header("Predict Student Performance")

    gender = st.selectbox("Gender", ["Male", "Female"])
    school_level = st.selectbox("School Level", ["Primary", "Middle", "High"])
    parental_support = st.selectbox("Parental Support", ["Yes", "No"])
    student_level = st.selectbox("Student Level", ["Beginner", "Intermediate", "Advanced"])
    focus_level = st.selectbox("Focus Level", ["Low", "Medium", "High"])
    recent_exam_score = st.slider("Recent Exam Score", 0, 100)
    attendance = st.slider("Attendance (%)", 0, 100)
    study_hours = st.slider("Study Hours Per Day", 0, 10)
    internet_access = st.selectbox("Internet Access", ["Yes", "No"])
    language_proficiency = st.selectbox("Language Proficiency", ["Low", "Medium", "High"])
    curriculum_type = st.selectbox("Curriculum Type", ["CBSE", "ICSE", "State Board"])

    if st.button("Predict"):
        # Manual encoding
        encoding = {
            "gender": {"Male": 0, "Female": 1},
            "school_level": {"Primary": 0, "Middle": 1, "High": 2},
            "parental_support": {"Yes": 1, "No": 0},
            "student_level": {"Beginner": 0, "Intermediate": 1, "Advanced": 2},
            "focus_level": {"Low": 0, "Medium": 1, "High": 2},
            "internet_access": {"Yes": 1, "No": 0},
            "language_proficiency": {"Low": 0, "Medium": 1, "High": 2},
            "curriculum_type": {"CBSE": 0, "ICSE": 1, "State Board": 2}
        }

        input_data = pd.DataFrame([[ 
            encoding["gender"][gender],
            encoding["school_level"][school_level],
            encoding["parental_support"][parental_support],
            encoding["student_level"][student_level],
            encoding["focus_level"][focus_level],
            recent_exam_score,
            attendance,
            study_hours,
            encoding["internet_access"][internet_access],
            encoding["language_proficiency"][language_proficiency],
            encoding["curriculum_type"][curriculum_type]
        ]], columns=['gender', 'school_level', 'parental_support', 'student_level', 'focus_level', 
                     'recent_exam_score', 'attendance', 'study_hours', 'internet_access', 
                     'language_proficiency', 'curriculum_type'])

        score = model.predict(input_data)[0]
        score = round(score, 2)

        # Categorize performance
        if score >= 80:
            category = "Excellent"
            color_func = st.success
            suggested_hours = "1-2 hours"
            attendance_target = "Above 90%"
        elif score >= 60:
            category = "Good"
            color_func = st.info
            suggested_hours = "2-3 hours"
            attendance_target = "85%+"
        elif score >= 40:
            category = "Average"
            color_func = st.warning
            suggested_hours = "3-4 hours"
            attendance_target = "80%+"
        else:
            category = "Needs Improvement"
            color_func = st.error
            suggested_hours = "4-5 hours"
            attendance_target = "75%+"

        color_func(f"✅ Predicted Performance Score: {score}")
        color_func(f"📊 Performance Category: {category}")

        st.subheader("📘 Study Recommendation:")
        st.markdown(f"- **Suggested Daily Study Time:** {suggested_hours}")
        st.markdown(f"- **Target Attendance:** {attendance_target}")
        st.markdown(f"- **AI Tip:** {recommendations.get(category, 'Keep learning and doing your best!')}")

# --- Recommendations Tab ---
with tabs[1]:
    st.header("Study Resources")

    with open("resources.json", "r") as f:
        data = json.load(f)

    class_level = st.selectbox("Class Level", list(data.keys()))
    subject = st.selectbox("Subject", list(data[class_level].keys()))

    topic_options = list(data[class_level][subject].keys())
    topic = st.selectbox("Topic", topic_options)

    if topic == "ALL TOPICS":
        link, label = data[class_level][subject]["ALL TOPICS"]
        st.markdown(f"### {label}")
        st.markdown(f"[Watch Video 📺]({link})")
    else:
        link, label = data[class_level][subject][topic]
        st.markdown(f"### {label}")
        st.markdown(f"[Watch Video 📺]({link})")

import streamlit as st
from src.predictor import predict_stress

st.set_page_config(page_title="Student Stress Risk Predictor", page_icon="📚", layout="centered")
st.title("📚 Student Stress Risk Predictor")
st.write("This app predicts a student's stress risk level based on daily habits and academic workload.")

sleep_hours = st.slider("Sleep Hours", 0, 12, 6)
study_hours = st.slider("Study Hours per Day", 0, 12, 5)
screen_time = st.slider("Screen Time per Day (hours)", 0, 14, 5)
exercise_days = st.slider("Exercise Days per Week", 0, 7, 2)
assignment_count = st.slider("Number of Assignments", 0, 10, 3)
exam_near = st.selectbox("Is an exam near?", ["No", "Yes"])
exam_value = 1 if exam_near == "Yes" else 0

if st.button("Predict Stress Risk"):
    prediction, probabilities = predict_stress(
        sleep_hours, study_hours, screen_time, exercise_days, assignment_count, exam_value
    )

    st.subheader("Prediction Result")
    if prediction == "High":
        st.error("High Stress Risk")
    elif prediction == "Medium":
        st.warning("Medium Stress Risk")
    else:
        st.success("Low Stress Risk")

    st.write("Prediction:", prediction)
    st.subheader("Recommendation")
    if prediction == "High":
        st.write("Your stress risk is high. Try to improve sleep, reduce screen time, take short breaks, and avoid overloading your study schedule.")
    elif prediction == "Medium":
        st.write("Your stress risk is moderate. Maintain balance between study, sleep, and physical activity to help reduce stress.")
    else:
        st.write("Your stress risk is low. Keep maintaining healthy habits.")

    st.subheader("Probabilities")
    st.write(probabilities)

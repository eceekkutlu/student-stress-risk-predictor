import os
import joblib
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'stress_model.pkl')


def load_model():
    data = joblib.load(os.path.abspath(MODEL_PATH))
    return data['model'], data['label_encoder']


def predict_stress(sleep_hours, study_hours, screen_time, exercise_days, assignment_count, exam_near):
    model, le = load_model()
    input_data = pd.DataFrame([{
        'sleep_hours': sleep_hours,
        'study_hours': study_hours,
        'screen_time': screen_time,
        'exercise_days': exercise_days,
        'assignment_count': assignment_count,
        'exam_near': exam_near,
    }])

    pred_enc = model.predict(input_data)[0]
    probs = model.predict_proba(input_data)[0]

    # Map probabilities to label names
    class_labels = le.inverse_transform(model.classes_)
    prob_dict = {label: float(probs[idx]) for idx, label in enumerate(class_labels)}

    prediction = le.inverse_transform([pred_enc])[0]
    return prediction, prob_dict

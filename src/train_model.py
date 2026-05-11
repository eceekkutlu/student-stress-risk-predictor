import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'student_stress_sample.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'stress_model.pkl')


def train_model():
    df = pd.read_csv(os.path.abspath(DATA_PATH))
    X = df.drop('stress_level', axis=1)
    y = df['stress_level']

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.25, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({'model': model, 'label_encoder': le}, os.path.abspath(MODEL_PATH))
    print(f"Model saved to {MODEL_PATH}")


if __name__ == '__main__':
    train_model()

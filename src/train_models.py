import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'student_stress_sample.csv')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')


def load_data():
    return pd.read_csv(os.path.abspath(DATA_PATH))


def build_models():
    return {
        'LogisticRegression': Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=2000))]),
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'KNeighbors': Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier())]),
        'SVC': Pipeline([('scaler', StandardScaler()), ('clf', SVC(probability=True))]),
    }


def evaluate_models(X_train, X_test, y_train, y_test, le):
    results = []
    models = build_models()
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for name, model in models.items():
        print(f'>> Training {name}...')
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True)

        # cross-validated score on full data for reference
        try:
            cv_scores = cross_val_score(model, pd.concat([X_train, X_test]), pd.concat([y_train, y_test]), cv=5)
            cv_mean = float(cv_scores.mean())
            cv_std = float(cv_scores.std())
        except Exception:
            cv_mean = None
            cv_std = None

        # Save model + label encoder together
        model_path = os.path.join(MODELS_DIR, f'{name.lower()}_model.pkl')
        joblib.dump({'model': model, 'label_encoder': le}, model_path)

        results.append({
            'model': name,
            'accuracy': float(acc),
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'model_path': model_path,
            'report': report,
        })

        print(f'  {name} accuracy: {acc:.4f} | saved to {model_path}')

    return results


def choose_best(results):
    # choose by highest accuracy on test set
    sorted_results = sorted(results, key=lambda r: r['accuracy'], reverse=True)
    return sorted_results[0]


def main():
    df = load_data()
    X = df.drop('stress_level', axis=1)
    y = df['stress_level']

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.25, random_state=42)

    results = evaluate_models(X_train, X_test, y_train, y_test, le)

    # save results summary
    results_df = pd.DataFrame([{
        'model': r['model'],
        'accuracy': r['accuracy'],
        'cv_mean': r['cv_mean'],
        'cv_std': r['cv_std'],
        'model_path': r['model_path']
    } for r in results])
    results_csv = os.path.join(OUTPUT_DIR, 'model_comparison.csv')
    results_df.to_csv(results_csv, index=False)
    print(f'Wrote results summary to {results_csv}')

    best = choose_best(results)
    print('\nBest model on test set:')
    print(f"  {best['model']} with accuracy {best['accuracy']:.4f}")
    print(f"Model artifact: {best['model_path']}")


if __name__ == '__main__':
    main()

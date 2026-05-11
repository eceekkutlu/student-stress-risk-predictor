import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'student_stress_sample.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'figures')


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    return pd.read_csv(os.path.abspath(DATA_PATH))


def plot_stress_distribution(df):
    plt.figure(figsize=(6,4))
    order = ['Low', 'Medium', 'High']
    sns.countplot(x='stress_level', data=df, order=order)
    plt.title('Stress Level Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'stress_distribution.png'))
    plt.close()


def plot_feature_distributions(df):
    numeric = ['sleep_hours', 'study_hours', 'screen_time', 'exercise_days', 'assignment_count']
    for col in numeric:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.tight_layout()
        fname = f'dist_{col}.png'
        plt.savefig(os.path.join(OUTPUT_DIR, fname))
        plt.close()


def plot_correlation_heatmap(df):
    plt.figure(figsize=(7,6))
    numeric = ['sleep_hours', 'study_hours', 'screen_time', 'exercise_days', 'assignment_count', 'exam_near']
    corr = df[numeric].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Feature Correlation')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_heatmap.png'))
    plt.close()


def plot_feature_importances(df):
    X = df.drop('stress_level', axis=1)
    y = LabelEncoder().fit_transform(df['stress_level'])
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

    plt.figure(figsize=(6,4))
    sns.barplot(x=importances.values, y=importances.index)
    plt.title('Feature Importances (RandomForest)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importances.png'))
    plt.close()


def main():
    ensure_output_dir()
    df = load_data()
    plot_stress_distribution(df)
    plot_feature_distributions(df)
    plot_correlation_heatmap(df)
    plot_feature_importances(df)
    print(f'Figures saved to {os.path.relpath(OUTPUT_DIR)}')


if __name__ == '__main__':
    main()

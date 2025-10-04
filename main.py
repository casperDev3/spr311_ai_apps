# Завдання: Класифікація діабету на основі dataset Pima Indians Diabetes
# 00, 01, 10, 11 - при 2 параметрах це 4 умови

# імпортуємо потрібні бібліотеки
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import asyncio

# Завантаження датасету діабету (Pima Indians)
from sklearn.datasets import load_diabetes
async def generate_synthetic_data(n_samples):
    return {
        'Вагітності': np.random.randint(0, 15, n_samples),
        'Глюкоза': np.random.normal(120, 30, n_samples),
        'Кров_тиск': np.random.normal(70, 15, n_samples),
        'Товщина_шкіри': np.random.normal(25, 10, n_samples),
        'Інсулін': np.random.normal(80, 100, n_samples),
        'ІМТ': np.random.normal(32, 7, n_samples),
        'Діабет_функція': np.random.uniform(0.1, 2.0, n_samples),
        'Вік': np.random.randint(21, 70, n_samples)
    }

async def generate_samples_diabetes(df, n_samples):
    diabetes_prob = (
            (df['Глюкоза'] > 140) * 0.3 +
            (df['ІМТ'] > 35) * 0.2 +
            (df['Вік'] > 50) * 0.15 +
            (df['Інсулін'] > 150) * 0.15 +
            np.random.random(n_samples) * 0.2
    )
    df['Діабет'] = (diabetes_prob > 0.5).astype(int)

def train_model(x_train, x_test, y_train, y_test):

    # Стандартизація даних
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(x_train)
    X_test_scaled = scaler.transform(x_test)

    # Навчання моделі (Логістична регресія)
    model_lr = LogisticRegression(max_iter=1000, random_state=42)
    model_lr.fit(X_train_scaled, y_train)

    # Навчання моделі (Випадковий ліс)
    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model_rf.fit(X_train_scaled, y_train)

    return model_lr, model_rf, X_train_scaled, X_test_scaled, scaler

def draw_visuals(df, y_test, y_pred_lr, y_pred_rf, y_pred_proba_lr, y_pred_proba_rf, feature_importance=None):
    # ВІЗУАЛІЗАЦІЯ
    fig = plt.figure(figsize=(16, 10))

    # 1. Розподіл класів
    ax1 = plt.subplot(2, 3, 1)
    df['Діабет'].value_counts().plot(kind='bar', color=['green', 'red'], alpha=0.7)
    plt.title('Розподіл класів', fontsize=12, fontweight='bold')
    plt.xlabel('Діабет (0 - Немає, 1 - Є)')
    plt.ylabel('Кількість пацієнтів')
    plt.xticks(rotation=0)

    # 2. Матриця помилок (Логістична регресія)
    ax2 = plt.subplot(2, 3, 2)
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Матриця помилок\n(Логістична регресія)', fontsize=12, fontweight='bold')
    plt.ylabel('Справжній клас')
    plt.xlabel('Передбачений клас')

    # 3. Матриця помилок (Випадковий ліс)
    ax3 = plt.subplot(2, 3, 3)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', cbar=False)
    plt.title('Матриця помилок\n(Випадковий ліс)', fontsize=12, fontweight='bold')
    plt.ylabel('Справжній клас')
    plt.xlabel('Передбачений клас')

    # 4. ROC крива
    ax4 = plt.subplot(2, 3, 4)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
    roc_auc_lr = auc(fpr_lr, tpr_lr)
    roc_auc_rf = auc(fpr_rf, tpr_rf)

    plt.plot(fpr_lr, tpr_lr, color='blue', lw=2, label=f'Логіст. регресія (AUC = {roc_auc_lr:.2f})')
    plt.plot(fpr_rf, tpr_rf, color='green', lw=2, label=f'Випадковий ліс (AUC = {roc_auc_rf:.2f})')
    plt.plot([0, 1], [0, 1], color='red', lw=1, linestyle='--', label='Випадковість')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC крива', fontsize=12, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    # 5. Важливість ознак
    ax5 = plt.subplot(2, 3, 5)
    feature_importance.plot(x='Ознака', y='Важливість', kind='barh', ax=ax5, legend=False, color='steelblue')
    plt.title('Важливість ознак\n(Випадковий ліс)', fontsize=12, fontweight='bold')
    plt.xlabel('Важливість')
    plt.ylabel('')

    plt.tight_layout()
    plt.show()

def run_diabetes_detect():
    # створення реалістичних даних
    np.random.seed(42)  # для відтворюваності
    n_samples = 768

    # Генерація ознак (симуляція реальних медичних показників)
    data = asyncio.run(generate_synthetic_data(n_samples))
    df = pd.DataFrame(data)

    # Target / Цільова змінна / Еталон
    asyncio.run(generate_samples_diabetes(df, n_samples))

    print("=== ІНФОРМАЦІЯ ПРО ДАТАСЕТ ===")
    print(f"Загальна кількість пацієнтів: {len(df)}")
    # TO-DO: Fix issue witch Promises
    # print("___", df['Діабет'])
    # print(f"З діабетом: {df['Діабет'].sum()} ({df['Діабет'].sum()/len(df)*100:.1f}%)")
    # print(f"Без діабету: {(1-df['Діабет']).sum()} ({(1-df['Діabет']).sum()/len(df)*100:.1f}%)")
    print("\nПерші 5 записів:")
    print(df.head())
    # Розділення на ознаки та цільову змінну
    X = df.drop('Діабет', axis=1)
    y = df['Діабет']

    # Розділення на тренувальний та тестовий набори
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    (model_lr, model_rf, X_train_scaled, X_test_scaled, scaler) = train_model(X_train, X_test, y_train, y_test)


    # Прогнозування
    y_pred_lr = model_lr.predict(X_test_scaled)
    y_pred_rf = model_rf.predict(X_test_scaled)
    y_pred_proba_lr = model_lr.predict_proba(X_test_scaled)[:, 1]
    y_pred_proba_rf = model_rf.predict_proba(X_test_scaled)[:, 1]


    # Оцінка моделей
    print("\n=== РЕЗУЛЬТАТИ ЛОГІСТИЧНОЇ РЕГРЕСІЇ ===")
    print(f"Точність: {accuracy_score(y_test, y_pred_lr):.3f}")
    print(f"\nМатриця помилок:\n{confusion_matrix(y_test, y_pred_lr)}")
    print(f"\n{classification_report(y_test, y_pred_lr, target_names=['Немає діабету', 'Є діабет'])}")

    print("\n=== РЕЗУЛЬТАТИ ВИПАДКОВОГО ЛІСУ ===")
    print(f"Точність: {accuracy_score(y_test, y_pred_rf):.3f}")
    print(f"\nМатриця помилок:\n{confusion_matrix(y_test, y_pred_rf)}")

    # Важливість ознак
    feature_importance = pd.DataFrame({
        'Ознака': X.columns,
        'Важливість': model_rf.feature_importances_
    }).sort_values('Важливість', ascending=False)
    print("\n=== ВАЖЛИВІСТЬ ОЗНАК ===")
    print(feature_importance)

    # draw_visuals(df, y_test, y_pred_lr, y_pred_rf, y_pred_proba_lr, y_pred_proba_rf, feature_importance)



    print("\n=== ПРИКЛАД ПРОГНОЗУВАННЯ ===")
    new_patient = pd.DataFrame({
        'Вагітності': [6],
        'Глюкоза': [148],
        'Кров_тиск': [72],
        'Товщина_шкіри': [35],
        'Інсулін': [125],
        'ІМТ': [33.6],
        'Діабет_функція': [0.627],
        'Вік': [50]
    })

    new_patient_scaled = scaler.transform(new_patient)
    prediction = model_rf.predict(new_patient_scaled)[0]
    probability = model_rf.predict_proba(new_patient_scaled)[0]

    print("Дані пацієнта:")
    print(new_patient.to_string(index=False))
    print(f"\nПрогноз: {'ДІАБЕТ ВИЯВЛЕНО' if prediction == 1 else 'Діабету немає'}")
    print(f"Ймовірність діабету: {probability[1] * 100:.1f}%")
    print(f"Ймовірність відсутності: {probability[0] * 100:.1f}%")

def main():
    run_diabetes_detect()


if __name__ == '__main__':
    main()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Carichiamo il dataset
df = pd.read_csv('../datasets/games-data.csv')

# Pulizia dei dati: rimuoviamo eventuali valori mancanti nelle colonne selezionate
df.dropna(subset=['user score', 'genre', 'critics', 'users'], inplace=True)

# Gestione dei valori non numerici nella colonna 'user score'
df['user score'] = df['user score'].replace('tbd', np.nan).astype(float)
df['user score'] = df['user score'].fillna(df['user score'].mean())

# Definiamo i criteri per il successo
def is_success(row):
    if row['score'] >= 70 and row['user score'] >= 7.0 and (row['critics'] + row['users']) >= 200:
        return 1
    else:
        return 0

# Aggiungiamo la colonna 'success' basata sui criteri definiti
df['success'] = df.apply(is_success, axis=1)

# Salva i risultati nel database indicato
df.to_csv('../datasets/games-data-with-success.csv', index=False)

# Codifica delle variabili categoriali
label_encoders = {}
for column in ['genre']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Separiamo le caratteristiche (features) dall'etichetta (target)
X = df[['user score', 'genre', 'critics', 'users']]
y = df['success']

# Dividiamo i dati in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalizziamo i dati
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Funzione per addestrare e valutare il modello
def train_and_evaluate_model(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["gioco non di successo", "gioco di successo"])
    return accuracy, report, model, y_pred, y_pred_prob

# Funzione per stampare i risultati del modello
def print_results(model_name, accuracy, report, model, X_test, y_test, y_pred, y_pred_prob):
    print(f"\nRisultati del modello {model_name}:")
    print(f"\nAccuracy: {accuracy}")
    print("\nClassification Report:")
    print(report)

    # Precisione sul set di addestramento
    train_accuracy = model.score(X_train, y_train)
    print(f"Training Accuracy: {train_accuracy}")

    # Precisione sul set di test
    test_accuracy = model.score(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy}")

    # Importanza delle caratteristiche, solo se disponibile
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        features = X.columns
        importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
        print("\nFeature Importances:")
        print(importance_df)
    else:
        print("\nIl modello selezionato non supporta l'estrazione delle importanze delle caratteristiche.")

    # Matrice di confusione
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Non di successo", "Di successo"], yticklabels=["Non di successo", "Di successo"])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()

    # Curva ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    print(f"\nArea Under the ROC Curve (AUC): {roc_auc}")
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve for {model_name}')
    plt.legend(loc="lower right")
    plt.show()

# Menu per scegliere il modello
def main():
    while True:
        printMenu()
        choice = input("Inserisci il numero del modello scelto: ")

        if choice == '1':
            model = KNeighborsClassifier(n_neighbors=3)
            model_name = "K-Nearest Neighbors (KNN)"
        elif choice == '2':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model_name = "Random Forest"
        elif choice == '3':
            model = SVC(kernel='linear', probability=True)
            model_name = "Support Vector Machine (SVM)"
        elif choice == '4':
            print("Uscita dal programma.")
            break
        else:
            print("\nSCELTA NON VALIDA. RIPROVA")
            continue

        accuracy, report, trained_model, y_pred, y_pred_prob = train_and_evaluate_model(model)
        print_results(model_name, accuracy, report, trained_model, X_test, y_test, y_pred, y_pred_prob)

# Metodo per stampare le voci del menù
def printMenu():
    print("\nScegli il modello da utilizzare:")
    print("1. K-Nearest Neighbors (KNN)")
    print("2. Random Forest")
    print("3. Support Vector Machine (SVM)")
    print("4. Per uscire")

# Esegui il menu principale all'avvio
if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, learning_curve, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, average_precision_score
from scipy.stats import gmean
import matplotlib.pyplot as plt
import seaborn as sns


# Carichiamo il dataset
df = pd.read_csv('../datasets/games-data_KB.csv')

# Pulizia dei dati: rimuoviamo eventuali valori mancanti nelle colonne selezionate
df.dropna(subset=['user score', 'genre', 'critics', 'users'], inplace=True)

# Codifica delle variabili categoriali
label_encoders = {}
for column in ['genre']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Separiamo le caratteristiche (features) dall'etichetta (target)
X = df[['user score', 'critics', 'score', 'users']]
y = df['success']

# Dividiamo i dati in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalizziamo i dati
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Funzione per tracciare la curva di apprendimento
def plot_learning_curve(model, model_name, X, y):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 7))
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')

    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='r', alpha=0.1)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color='g', alpha=0.1)

    plt.title(f'Learning Curve for {model_name}')
    plt.xlabel('Training size')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(f'../img/Apprendimento_supervisionato/{model_name} - learningCurve.png')
    plt.show()

# Funzione per calcolare la GMAP
def calculate_gmap(y_true, y_pred_prob):
    # Calcola l'Average Precision per ogni classe
    ap_per_class = []
    for i in range(y_pred_prob.shape[1]):
        ap = average_precision_score(y_true == i, y_pred_prob[:, i])
        ap_per_class.append(ap)
    # Calcola la GMAP come media geometrica delle AP per classe
    gmap = gmean(ap_per_class)
    return gmap

# Funzione per addestrare e valutare il modello con cross-validation e GridSearchCV
def train_and_evaluate_model_grid_search(model, param_grid, X_train, y_train, X_test, y_test, model_name):
    # Definiamo il metodo di cross-validation
    cv_method = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # GridSearchCV per ricerca dei parametri ottimali
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv_method, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Calcola la GMAP per ciascun fold della cross-validation
    gmap_scores = []
    for train_index, test_index in cv_method.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[test_index]  # Accedi direttamente usando gli indici
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]  # Assicurati di mantenere y_train come DataFrame

        model.fit(X_train_fold, y_train_fold)
        y_pred_prob = model.predict_proba(X_val_fold)
        gmap = calculate_gmap(y_val_fold, y_pred_prob)
        gmap_scores.append(gmap)

    mean_gmap = np.mean(gmap_scores)
    std_gmap = np.std(gmap_scores)

    # Valutazione del modello ottimizzato sul set di test
    y_pred = grid_search.predict(X_test)
    y_pred_prob = grid_search.predict_proba(X_test)
    cv_scores = cross_val_score(grid_search.best_estimator_, X_train, y_train, cv=cv_method, scoring='accuracy')

    # Salva le predizioni nel dataset originale
    # df['predicted_success'] = grid_search.predict(scaler.transform(X))  # Usa tutto il dataset per predire
    # df.to_csv(f'../predictions/{model_name}_predictions.csv', index=False)

    # Stampa dei risultati
    print_results_cv(model_name, cv_scores, accuracy_score(y_test, y_pred), classification_report(y_test, y_pred, target_names=["gioco non di successo", "gioco di successo"]), grid_search.best_estimator_, X_test, y_test, y_pred, y_pred_prob, mean_gmap, std_gmap)

    # Tracciamento della curva di apprendimento
    plot_learning_curve(grid_search.best_estimator_, model_name, X_train, y_train)

    # Salva il grafico della curva ROC
    plt.figure()
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve for {model_name}')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(f'../img/Apprendimento_supervisionato/{model_name} - ROC.png')
    plt.show()


# Stampiamo i parametri ottimizzati
    print(f"\nParametri ottimizzati per {model_name}:")
    print(grid_search.best_params_)

# Funzione per stampare i risultati del modello
def print_results_cv(model_name, cv_scores, accuracy, report, model, X_test, y_test, y_pred, y_pred_prob, mean_gmap, std_gmap):
    print(f"\nRisultati del modello {model_name}:")
    print("\nClassification Report:")
    print(report)
    print("Risultati della Cross-Validation:")
    print(f"\nCross-Validation Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean()}")  # Media delle accuratezze
    print(f"Standard Deviation CV Accuracy: {cv_scores.std()}")
    print(f"\nAccuracy on Test Set: {accuracy}")

    # Precisione sul set di addestramento
    train_accuracy = model.score(X_train, y_train)
    print(f"Training Accuracy: {train_accuracy}")

    # Precisione sul set di test
    test_accuracy = model.score(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy}")

    # Stampiamo la GMAP media e deviazione standard
    print(f"\nGeometric Mean Average Precision (GMAP): {mean_gmap} (± {std_gmap})")

    if model_name == 'Support Vector Machine' or model_name == 'Gradient Boosting Classifier' or DecisionTreeClassifier:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            features = X.columns
            importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
            print("\nFeature Importances:")
            print(importance_df)

    # Matrice di confusione
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Non di successo", "Di successo"], yticklabels=["Non di successo", "Di successo"])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.savefig(f'../img/Apprendimento_supervisionato/{model_name} - confusionMatrix.png')
    plt.show()

def main():

    try:
        while True:
            printMenu()
            choice = input("Inserisci il numero del modello scelto: ")

            if choice == '1':
                # Random Forest con GridSearchCV
                rf_param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                rf_model = RandomForestClassifier(random_state=42)
                train_and_evaluate_model_grid_search(rf_model, rf_param_grid, X_train, y_train, X_test, y_test, "Random Forest")

            elif choice == '2':
                # Support Vector Machine con GridSearchCV
                svc_param_grid = {
                    'C': [0.1, 1, 10, 100],
                    'gamma': [1, 0.1, 0.01, 0.001],
                    'kernel': ['rbf']
                }
                svc_model = SVC(probability=True, random_state=42)
                train_and_evaluate_model_grid_search(svc_model, svc_param_grid, X_train, y_train, X_test, y_test, "Support Vector Machine")

            elif choice == '3':
                # Decision Tree con GridSearchCV
                dt_param_grid = {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                dt_model = DecisionTreeClassifier(random_state=42)
                train_and_evaluate_model_grid_search(dt_model, dt_param_grid, X_train, y_train, X_test, y_test, "Decision Tree")

            elif choice == '4':
                # K-Nearest Neighbors con GridSearchCV
                knn_param_grid = {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                }
                knn_model = KNeighborsClassifier()
                train_and_evaluate_model_grid_search(knn_model, knn_param_grid, X_train, y_train, X_test, y_test, "K-Nearest Neighbors")

            elif choice == '5':
                # Gradient Boosting Classifier con GridSearchCV
                gb_param_grid = {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.05],
                    'max_depth': [3, 4, 5]
                }
                gb_model = GradientBoostingClassifier(random_state=42)
                train_and_evaluate_model_grid_search(gb_model, gb_param_grid, X_train, y_train, X_test, y_test, "Gradient Boosting Classifier")

            elif choice == '0':
                print("Uscita...")
                break

            else:
                print("Scelta non valida. Per favore, riprova.")
    except KeyboardInterrupt:
        print("\nEsecuzione interrotta.")


def printMenu():

    print("\nModelli di Machine Learning:")
    print("1. Random Forest")
    print("2. Support Vector Machine")
    print("3. Decision Tree")
    print("4. K-Nearest Neighbors")
    print("5. Gradient Boosting Classifier")
    print("0. Exit")


if __name__ == "__main__":
    main()

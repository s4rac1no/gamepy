import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Caricamento del dataset
data = pd.read_csv('../datasets/games-data.csv')

# Verifica e gestione dei valori mancanti
data = data.dropna()

# Rimozione del testo non numerico e conversione in float per la colonna "players"
data['players'] = data['players'].str.replace(r'\D', '', regex=True)
data['players'] = pd.to_numeric(data['players'], errors='coerce')

# Eliminazione delle righe con valori mancanti nella colonna 'players'
data = data.dropna(subset=['players'])

# Unifica le piattaforme Xbox One e Xbox 360 sotto la categoria Xbox
data['platform'] = data['platform'].replace(['Xbox One', 'Xbox 360'], 'Xbox')

# Unifica le piattaforme PlayStation 2, PlayStation 3 e PlayStation 4 sotto la categoria PlayStation
data['platform'] = data['platform'].replace(['PlayStation 2', 'PlayStation 3', 'PlayStation 4'], 'PlayStation')

# Unifica le piattaforme Wii, Wii U, Nintendo DS e Nintendo 3DS sotto la categoria Nintendo
data['platform'] = data['platform'].replace(['Wii', 'Wii U', 'Nintendo DS', 'Nintendo 3DS'], 'Nintendo')

# Seleziona solo le righe relative alle piattaforme specificate
allowed_platforms = ['PC', 'Xbox', 'PlayStation', 'Nintendo']
data = data[data['platform'].isin(allowed_platforms)]

# Selezione delle colonne da utilizzare
data = data[['platform', 'score', 'players']]

# Separazione delle caratteristiche numeriche e categoriche
numerical_features = ['score', 'players']
categorical_features = ['platform']

# Pipeline per la trasformazione delle variabili categoriche
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Pipeline per la standardizzazione delle variabili numeriche
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Composizione della ColumnTransformer per applicare le trasformazioni appropriate
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Lista per memorizzare i valori di inertia per ciascun K
inertia = []

# Range di K da 1 a 10
k_range = range(1, 11)

for k in k_range:
    # Creazione della pipeline con K-means
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('kmeans', KMeans(n_clusters=k, random_state=42))
    ])

    # Adattamento della pipeline ai dati
    pipeline.fit(data)

    # Recupero del valore di inertia e aggiunta alla lista
    inertia.append(pipeline.named_steps['kmeans'].inertia_)

# Visualizzazione della curva a gomito
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, marker='o')
plt.xlabel('Numero di cluster (K)')
plt.ylabel('Inertia')
plt.title('Curva a Gomito per la determinazione del numero ottimale di cluster')
plt.show()

# Determinazione del numero ottimale di cluster (esempio: 3 cluster)
optimal_k = 3

# Creazione della pipeline con il numero ottimale di cluster
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('kmeans', KMeans(n_clusters=optimal_k, random_state=42))
])

# Adattamento della pipeline ai dati
pipeline.fit(data)

# Predizione dei cluster
data['cluster'] = pipeline['kmeans'].labels_

# Visualizzazione della distribuzione dei cluster
print(data['cluster'].value_counts())

# Plotting
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='score', y='players', hue='cluster', style='platform', palette='Set1')
plt.title('K-means Clustering dei Videogiochi (score vs players)')
plt.xlabel('Score')
plt.ylabel('Players')
plt.legend(title='Platform')
plt.show()
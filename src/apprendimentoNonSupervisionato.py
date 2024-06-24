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
data = data[['name', 'platform', 'score', 'users']]

# Prendi un campione casuale del 30% del dataset
#data_sample = data.sample(frac=0.3, random_state=42)

# Separazione delle caratteristiche numeriche e categoriche
numerical_features = ['score', 'users']
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
    pipeline.fit(data[['platform', 'score', 'users']])

    # Recupero del valore di inertia e aggiunta alla lista
    inertia.append(pipeline.named_steps['kmeans'].inertia_)

# Visualizzazione della curva a gomito
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, marker='o')
plt.xlabel('Numero di cluster (K)')
plt.ylabel('Inertia')
plt.title('Curva a Gomito per la determinazione del numero ottimale di cluster')
plt.savefig('../img/Apprendimento_non_supervisionato/gomito_curva.png')
plt.show()

# Determinazione del numero ottimale di cluster ricavati dalla regola del gomito
optimal_k = 3

# Creazione della pipeline con il numero ottimale di cluster
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('kmeans', KMeans(n_clusters=optimal_k, random_state=42))
])

# Adattamento della pipeline ai dati
pipeline.fit(data[['platform', 'score', 'users']])

# Predizione dei cluster
data['cluster'] = pipeline['kmeans'].labels_

# Calcolo delle distanze dal centro del cluster
# Recupero dei centri dei cluster
centroids = pipeline.named_steps['kmeans'].cluster_centers_

# Trasformazione dei dati
transformed_data = pipeline.named_steps['preprocessor'].transform(data[['platform', 'score', 'users']])

# Calcolo della distanza euclidea da ciascun centro di cluster
distances = np.min(np.linalg.norm(transformed_data[:, np.newaxis] - centroids, axis=2), axis=1)

# Aggiunta delle distanze al dataframe
data['distance_from_centroid'] = distances

# Determinazione della soglia per le anomalie (95° percentile)
threshold = np.percentile(distances, 95)

# Identificazione delle anomalie
anomalies = data[distances > threshold]

# Salvataggio delle anomalie in un file CSV
anomalies.to_csv('../results/anomalies.csv', index=False)

# Visualizzazione della distribuzione dei cluster
print(data['cluster'].value_counts())

# Plotting migliorato con scala logaritmica sull'asse x
plt.figure(figsize=(12, 8))
sns.scatterplot(data=data, x='users', y='score', hue='cluster', style='platform', palette='Set1', s=100)
plt.xscale('log')
plt.title('K-means Clustering dei Videogiochi (users vs score)', fontsize=16)
plt.xlabel('Users', fontsize=14)
plt.ylabel('Score', fontsize=14)
plt.legend(title='Platform')
plt.grid(True, which="both", ls="--")
plt.savefig('../img/Apprendimento_non_supervisionato/K-means_cluster.png')
plt.show()

print(f'Numero di anomalie identificate: {len(anomalies)}')
print('Le anomalie sono state salvate nel file anomalies.csv nella directory results.')

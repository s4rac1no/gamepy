import pandas as pd
import random

# Lettura del dataset di giochi
def leggi_dataset(file):
    return pd.read_csv(file)

# Controllo dei vincoli
def verifica_vincoli(game_with_constraints):
    # Vincolo: esattamente 10 giochi
    if len(game_with_constraints) != 10:
        return False
    # Vincolo: numero totale di critici massimo di 300
    critici_totale = game_with_constraints['critics'].sum()
    if critici_totale > 300:
        return False
    # Vincolo: media del punteggio degli utenti almeno 7.0
    user_score_medio = game_with_constraints['user score'].mean()
    if user_score_medio < 7.0:
        return False
    # Vincolo: non più di due giochi dello stesso genere
    generi = {}
    for generi_lista in game_with_constraints['genre']:
        for genere in generi_lista.split(','):
            genere = genere.strip()
            if genere not in generi:
                generi[genere] = 0
            generi[genere] += 1
            if generi[genere] > 2:
                return False
    return True

# Funzione per verificare se i vincoli sono stati soddisfatti
def verifica_vincoli_soddisfatti(game_with_constraints):
    if verifica_vincoli(game_with_constraints):
        print("I vincoli sono stati soddisfatti.")
    else:
        print("I vincoli non sono stati soddisfatti.")

# Funzione di valutazione dei vincoli
def valuta_vincoli(game_with_constraints):
    violazioni = 0
    if len(game_with_constraints) != 10:
        violazioni += abs(10 - len(game_with_constraints))
    critici_totale = game_with_constraints['critics'].sum()
    if critici_totale > 300:
        violazioni += critici_totale - 300
    user_score_medio = game_with_constraints['user score'].mean()
    if user_score_medio < 7.0:
        violazioni += (7.0 - user_score_medio) * len(game_with_constraints)
    generi = {}
    for generi_lista in game_with_constraints['genre']:
        for genere in generi_lista.split(','):
            genere = genere.strip()
            if genere not in generi:
                generi[genere] = 0
            generi[genere] += 1
            if generi[genere] > 2:
                violazioni += generi[genere] - 2
    return violazioni

# Algoritmo di random walk
def random_walk(dataset, max_iter=1000):
    best_game_with_constraints = None
    best_violazioni = float('inf')
    for _ in range(max_iter):
        game_with_constraints = dataset.sample(n=10)
        violazioni = valuta_vincoli(game_with_constraints)
        if violazioni < best_violazioni:
            best_game_with_constraints = game_with_constraints
            best_violazioni = violazioni
        if violazioni == 0:
            break
    return best_game_with_constraints

# Scrittura del dataset ottimizzato in un file CSV
def scrivi_game_with_constraints(file, game_with_constraints):
    game_with_constraints.to_csv(file, index=False)

# Esecuzione del processo
dataset = leggi_dataset('../datasets/games-data-with-success.csv')
game_with_constraints_ottimizzato = random_walk(dataset)
scrivi_game_with_constraints('../results/game_with_constraints.csv', game_with_constraints_ottimizzato)

# Verifica se i vincoli sono stati soddisfatti
verifica_vincoli_soddisfatti(game_with_constraints_ottimizzato)


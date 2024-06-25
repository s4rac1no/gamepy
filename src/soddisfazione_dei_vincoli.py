import pandas as pd
import random
import numpy as np
import time

# Lettura del dataset di giochi
def leggi_dataset(file):
    return pd.read_csv(file)

# Controllo dei vincoli
def verifica_vincoli(game_with_constraints):
    if len(game_with_constraints) != 10:
        return False
    critici_totale = game_with_constraints['critics'].sum()
    if critici_totale > 300:
        return False
    user_score_medio = game_with_constraints['user score'].mean()
    if user_score_medio < 7.0:
        return False
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

# Algoritmo di Random Walk
def random_walk(dataset, max_iter=1000):
    # Inizializzazione delle migliori soluzioni e delle violazioni
    best_game_with_constraints = None
    best_violazioni = float('inf')

    # Iterazione per un massimo di 'max_iter' volte
    for _ in range(max_iter):
        # Generazione di una nuova soluzione casuale campionando 10 elementi dal dataset
        game_with_constraints = dataset.sample(n=10)

        # Valutazione delle violazioni dei vincoli nella soluzione corrente
        violazioni = valuta_vincoli(game_with_constraints)

        # Confronto con la migliore soluzione trovata finora
        if violazioni < best_violazioni:
            best_game_with_constraints = game_with_constraints
            best_violazioni = violazioni

        # Se non ci sono violazioni, interrompi il ciclo
        if violazioni == 0:
            break

    # Restituzione della migliore soluzione trovata
    return best_game_with_constraints


# Algoritmo di Simulated Annealing
def simulated_annealing(dataset, max_iter=1000, temp=1000, alpha=0.99):
    # Inizializzazione della soluzione corrente campionando 10 elementi dal dataset senza duplicati
    current_solution = dataset.sample(n=10).drop_duplicates()
    # Verifica che la soluzione corrente contenga esattamente 10 elementi
    while len(current_solution) < 10:
        current_solution = dataset.sample(n=10).drop_duplicates()

    # Valutazione delle violazioni dei vincoli nella soluzione corrente
    current_violations = valuta_vincoli(current_solution)
    best_solution = current_solution
    best_violations = current_violations

    for i in range(max_iter):
        # Riduzione della temperatura ad ogni iterazione
        temp *= alpha
        if temp <= 0:
            break

        # Generazione di una nuova soluzione campionando 10 elementi dal dataset senza duplicati
        new_solution = dataset.sample(n=10).drop_duplicates()
        while len(new_solution) < 10:
            new_solution = dataset.sample(n=10).drop_duplicates()

        # Valutazione delle violazioni dei vincoli nella nuova soluzione
        new_violations = valuta_vincoli(new_solution)

        # Calcolo della variazione nelle violazioni
        delta = new_violations - current_violations

        # Condizioni di accettazione della nuova soluzione:
        # Se la nuova soluzione è migliore (quindi ha meno violazioni), accettala
        # Altrimenti, accettala con una probabilità che tiene conto della temperatura
        if delta < 0 or np.exp(-delta / temp) > random.random():
            current_solution = new_solution
            current_violations = new_violations

            # Aggiornamento della migliore soluzione trovata
            if new_violations < best_violations:
                best_solution = new_solution
                best_violations = new_violations

        # Se non ci sono violazioni, termina l'algoritmo
        if best_violations == 0:
            break

    return best_solution



# Scrittura del dataset ottimizzato in un file CSV
def scrivi_game_with_constraints(file, game_with_constraints):
    game_with_constraints.to_csv(file, index=False)

# Scrittura del dataset ottimizzato da Simulated Annealing in un file CSV separato
def scrivi_simulated_annealing_results(file, game_with_constraints):
    game_with_constraints.to_csv(file, index=False)

# Funzione principale con menu e test delle prestazioni
def main():
    dataset = leggi_dataset('../datasets/games-data_KB.csv')
    try:
        while True:
            print("Scegli un metodo di ottimizzazione:")
            print("1. Random Walk")
            print("2. Simulated Annealing")
            print("3. Esci")

            scelta = input("Inserisci il numero della tua scelta: ")

            if scelta == '3':
                print("Uscita dal programma.")
                break

            risultati = []
            tempi = []

            for _ in range(10):  # Esegui ogni algoritmo 10 volte per ottenere una media delle prestazioni
                start_time = time.time()

                if scelta == '1':
                    game_with_constraints_ottimizzato = random_walk(dataset)
                    scrivi_game_with_constraints('../results/random_walk_game_with_constraints.csv', game_with_constraints_ottimizzato)
                elif scelta == '2':
                    game_with_constraints_ottimizzato = simulated_annealing(dataset)
                    scrivi_simulated_annealing_results('../results/simulated_annealing_game_with_constraints.csv', game_with_constraints_ottimizzato)
                else:
                    print("Scelta non valida. Riprova.")
                    continue

                end_time = time.time()
                durata = end_time - start_time
                tempi.append(durata)
                violazioni = valuta_vincoli(game_with_constraints_ottimizzato)
                risultati.append(violazioni)

            media_tempo = np.mean(tempi)
            media_violazioni = np.mean(risultati)

            print(f"Media del tempo di esecuzione: {media_tempo:.4f} secondi")
            print(f"Media delle violazioni dei vincoli: {media_violazioni:.2f}")

            verifica_vincoli_soddisfatti(game_with_constraints_ottimizzato)

    except KeyboardInterrupt as e:
        print("\n\nEsecuzione interrotta.", e)

# Esecuzione del programma principale
if __name__ == "__main__":
    main()

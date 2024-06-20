import os
import pandas as pd
from pyswip import Prolog
import random
import re
from createKB import normalize_developer_name

# Funzione per normalizzare il nome del gioco
def normalize_game_name(name):
    name = name.replace("'", "_")
    name = re.sub(r'[.:]', '-', name)
    return name

# Funzione per pulire il genere
def clean_genre(genre):
    if "'" in genre:
        genre = genre.replace("'", "")
    return genre.strip()

# Funzione per estrarre l'anno dalla data di rilascio
def extract_year_from_r_date(r_date):
    match = re.search(r'\d{4}$', r_date)
    return int(match.group()) if match else None

# Funzione per assegnare pesi ai generi
def assign_genre_weight(genre):
    genre_weights = {
        'action': 1,
        'adventure': 1,
        'action adventure': 2,
        'strategy': 0.50,
        'simulation': 0.40,
        'role-playing': 0.70,
        'sports': 0.90,
        'racing': 0.50
    }
    return genre_weights.get(genre.lower(), 0.20)

# Funzione per verificare l'installazione di SWI-Prolog
def check_swi_prolog_installation():
    print("Verifica dell'installazione di SWI-Prolog...")
    try:
        os.system("swipl --version")
    except Exception as e:
        print(f"Errore durante la verifica di SWI-Prolog: {e}")
        return False
    return True

# Funzione per ottenere i generi disponibili nella base di conoscenza
def get_available_genres(prolog):
    genre_query = "distinct(Genre, gioco_generi(_, Genre))"
    results = list(prolog.query(genre_query))
    return [result["Genre"] for result in results]

# Funzione per ottenere gli sviluppatori disponibili nella base di conoscenza
def get_available_developers(prolog):
    developer_query = "distinct(Developer, gioco_developer(_, Developer))"
    results = list(prolog.query(developer_query))
    return [result["Developer"] for result in results]

# Funzione per cercare i giochi in base al genere scelto dall'utente
def query_genre(prolog):
    genres = get_available_genres(prolog)

    print("Generi disponibili:")
    for genre in genres:
        print(genre)

    genre_choice = input("Inserisci il genere desiderato: ").strip().lower()

    normalized_genres = {genre.lower(): genre for genre in genres}

    if genre_choice not in normalized_genres:
        print(f"Genere '{genre_choice}' non disponibile.")
        return

    normalized_genre_choice = normalized_genres[genre_choice]
    genre_query = f"gioco_generi(Nome, '{normalized_genre_choice}')"
    print(f"Giochi del genere '{normalized_genre_choice}':")

    results = list(prolog.query(genre_query))

    if not results:
        print(f"Nessun gioco trovato per il genere '{normalized_genre_choice}'.")
        return

    random_results = random.sample(results, min(10, len(results)))

    for result in random_results:
        print(result["Nome"])

# Funzione per cercare i giochi con maggiore successo a partire da un certo anno
def query_top_games_by_year(prolog):
    try:
        year = int(input("Inserisci un anno (1996-2020): "))
        while year < 1996 or year >= 2021:
            print("Anno non valido. Inserisci un anno compreso tra 1996 e 2020.")
            year = int(input())

        year_query = f"gioco_top_score(Nome, Anno), Anno >= {year}"
        print(f"Giochi con maggiore successo e anno di uscita >= {year} :")

        results = list(prolog.query(year_query))

        if not results:
            print("Nessun gioco trovato per questo anno.")
            return

        random_results = random.sample(results, min(10, len(results)))

        for result in random_results:
            print(result["Nome"])

    except ValueError:
        print("Inserisci un numero valido.")

# Funzione per cercare i migliori giochi di un developer scelto dall'utente
def query_top_games_by_developer(prolog):
    developers = get_available_developers(prolog)

    print("Sviluppatori disponibili:")
    for dev in developers:
        print(dev)

    developer_choice = input("Inserisci il developer desiderato: ").strip().lower()

    normalized_developers = {dev.lower(): dev for dev in developers}

    if developer_choice not in normalized_developers:
        print(f"Developer '{developer_choice}' non disponibile.")
        return

    normalized_dev_choice = normalized_developers[developer_choice]
    dev_query = f"gioco_developer(Nome, '{normalized_dev_choice}')"
    print(f"Ecco i 5 migliori giochi del developer '{normalized_dev_choice}':")

    results = list(prolog.query(dev_query))

    if not results:
        print(f"Nessun gioco trovato per il developer '{normalized_dev_choice}'.")
        return

    print("Elenco giochi:")
    for result in results:
        print(result["Nome"])

# Funzione per cercare giochi per tipo di modalità di gioco
def query_games_by_mode(prolog):
    modes = ["singleplayer", "multiplayer", "co-op mode", "no mode"]

    print("Modalità di gioco disponibili:")
    for mode in modes:
        print(mode)

    mode_choice = input("Inserisci la modalità di gioco desiderata: ").strip().lower()

    if mode_choice not in modes:
        print(f"Modalità di gioco '{mode_choice}' non disponibile.")
        return

    mode_query = f"gioco_modalita(Nome, '{mode_choice}')"
    print(f"Giochi di tipo '{mode_choice}':")

    results = list(prolog.query(mode_query))

    if not results:
        print(f"Nessun gioco trovato per la modalità '{mode_choice}'.")
        return

    random_results = random.sample(results, min(10, len(results)))

    for result in random_results:
        print(result["Nome"])

# Funzione per cercare giochi che potrebbero essere di tendenza basati sui pesi dei generi
def query_games_by_genre_weight(prolog, dataset_filename):
    try:
        trend_query = "gioco_peso_generi(Nome, Peso), Peso >= 0.5"
        print("Giochi che potrebbero essere di tendenza:")

        results = list(prolog.query(trend_query))

        if not results:
            print("Nessun gioco trovato che potrebbe essere di tendenza.")
            return

        random_results = random.sample(results, min(5, len(results)))

        for result in random_results:
            print(result["Nome"])

    except Exception as e:
        print(f"Errore durante la query dei giochi di tendenza: {e}")

# Funzione per ottenere i giochi di tendenza di un developer dalla base di conoscenza
def get_trending_games_by_developer(prolog, developer_name):
    dev_query = f"game_developer_fact(Nome, '{developer_name}')"
    results = list(prolog.query(dev_query))
    return [result["Nome"] for result in results]

# Funzione per eseguire la query 6
def top_game_developers(prolog):
    try:
        trending_developers_df = pd.read_csv('../datasets/trending_developers_playlist.csv')

        top_trending_developers = trending_developers_df.head(5)['Developer'].tolist()

        print("Ecco i 5 developer di tendenza e i loro giochi di tendenza:")
        for developer in top_trending_developers:
            print(f"\nDeveloper: {developer}")
            trending_games = get_trending_games_by_developer(prolog, developer)

            if not trending_games:
                print("Nessun gioco di tendenza trovato per questo developer.")
            else:
                print("Giochi di tendenza:")
                for game in trending_games[:5]:
                    print(game)

    except Exception as e:
        print(f"Errore durante l'esecuzione della query: {e}")

# Funzione principale per gestire il menu e le query
def query_kb():
    if not check_swi_prolog_installation():
        print("SWI-Prolog non è installato correttamente.")
        return

    prolog = Prolog()
    prolog.consult("games_kb.pl")

    dataset_filename = '../datasets/games-data_KB.csv'

    while True:
        print("\nMenu:")
        print("1. Mostra 10 giochi di un genere scelto")
        print("2. Mostra 10 giochi con maggiore successo a partire da un certo anno")
        print("3. Mostra 5 migliori giochi di un developer scelto")
        print("4. Mostra 10 giochi di una modalità di gioco scelta")
        print("5. Mostra 5 giochi che potrebbero essere di tendenza")
        print("6. Mostra alcuni giochi di un developer di tendenza")
        print("7. Esci")

        choice = input("Scegli un'opzione: ")

        if choice == '1':
            query_genre(prolog)
        elif choice == '2':
            query_top_games_by_year(prolog)
        elif choice == '3':
            query_top_games_by_developer(prolog)
        elif choice == '4':
            query_games_by_mode(prolog)
        elif choice == '5':
            query_games_by_genre_weight(prolog, dataset_filename)
        elif choice == '6':
            top_game_developers(prolog)
        elif choice == '7':
            print("Uscita...")
            break
        else:
            print("Scelta non valida. Riprova.")

if __name__ == '__main__':
    query_kb()

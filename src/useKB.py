import os
from pyswip import Prolog
import random

def check_swi_prolog_installation():
    print("Verifica dell'installazione di SWI-Prolog...")
    try:
        os.system("swipl --version")
    except Exception as e:
        print(f"Errore durante la verifica di SWI-Prolog: {e}")
        return False
    return True

def get_available_genres(prolog):
    genre_query = "distinct(Genre, gioco_generi(_, Genre))"
    results = list(prolog.query(genre_query))
    return [result["Genre"] for result in results]

def get_available_developers(prolog):
    developer_query = "distinct(Developer, gioco_developer(_, Developer))"
    results = list(prolog.query(developer_query))
    return [result["Developer"] for result in results]

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

    # Recupera tutti i risultati
    results = list(prolog.query(genre_query))

    # Se non ci sono risultati, stampa un messaggio e ritorna
    if not results:
        print(f"Nessun gioco trovato per il genere '{normalized_genre_choice}'.")
        return

    # Ottiene fino a 10 risultati casuali
    random_results = random.sample(results, min(10, len(results)))

    for result in random_results:
        print(result["Nome"])

def query_top_games_by_year(prolog):
    try:
        year = int(input("Inserisci un anno (1996-2020): "))
        while year < 1996 or year >= 2021:
            print("Anno non valido. Inserisci un anno compreso tra 1996 e 2020.")
            year = int(input())

        year_query = f"gioco_top_score(Nome, Anno), Anno >= {year}"
        print(f"Giochi con maggiore successo e anno di uscita >= {year} :")

        # Recupera tutti i risultati
        results = list(prolog.query(year_query))

        # Se non ci sono risultati, stampa un messaggio e ritorna
        if not results:
            print("Nessun gioco trovato per questo anno.")
            return

        # Ottiene fino a 10 risultati casuali
        random_results = random.sample(results, min(10, len(results)))

        for result in random_results:
            print(result["Nome"])

    except ValueError:
        print("Inserisci un numero valido.")

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
    print(f"Giochi del developer '{normalized_dev_choice}':")

    # Recupera tutti i risultati
    results = list(prolog.query(dev_query))

    # Se non ci sono risultati, stampa un messaggio e ritorna
    if not results:
        print(f"Nessun gioco trovato per il developer '{normalized_dev_choice}'.")
        return

    # Ottiene fino a 5 risultati casuali
    random_results = random.sample(results, min(5, len(results)))

    for result in random_results:
        print(result["Nome"])

def query_kb():
    if not check_swi_prolog_installation():
        print("SWI-Prolog non è installato correttamente.")
        return

    prolog = Prolog()
    prolog.consult("games_kb.pl")

    while True:
        print("\nMenu:")
        print("1. Mostra 10 giochi di un genere scelto")
        print("2. Mostra 10 giochi con maggiore successo a partire da un certo anno")
        print("3. Mostra 5 migliori giochi di un developer")
        print("4. Esci")

        choice = input("Scegli un'opzione: ")

        if choice == '1':
            query_genre(prolog)
        elif choice == '2':
            query_top_games_by_year(prolog)
        elif choice == '3':
            query_top_games_by_developer(prolog)
        elif choice == '4':
            print("Uscita...")
            break
        else:
            print("Scelta non valida. Riprova.")

if __name__ == '__main__':
    query_kb()
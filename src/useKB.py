import os
from pyswip import Prolog
import random

# Funzione per verificare se SWI-Prolog è installato correttamente
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

    # Normalizzazione degli input per evitare problemi di maiuscole/minuscole
    normalized_genres = {genre.lower(): genre for genre in genres}

    if genre_choice not in normalized_genres:
        print(f"Genere '{genre_choice}' non disponibile.")
        return

    normalized_genre_choice = normalized_genres[genre_choice]
    genre_query = f"gioco_generi(Nome, '{normalized_genre_choice}')"
    print(f"Giochi del genere '{normalized_genre_choice}':")

    # Recupera tutti i risultati della query
    results = list(prolog.query(genre_query))

    # Se non ci sono risultati, stampa un messaggio e ritorna al menù
    if not results:
        print(f"Nessun gioco trovato per il genere '{normalized_genre_choice}'.")
        return

    # Messaggio di avviso se meno di 10 giochi
    if len(results) < 10:
        print(f"Attenzione: trovati solo {len(results)} giochi per il genere '{normalized_genre_choice}'.")

    # Ottiene fino a 10 risultati casuali
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

        # Recupera tutti i risultati della query
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

# Funzione per cercare i migliori giochi di un developer scelto dall'utente
def query_top_games_by_developer(prolog):
    developers = get_available_developers(prolog)

    print("Sviluppatori disponibili:")
    for dev in developers:
        print(dev)

    developer_choice = input("Inserisci il developer desiderato: ").strip().lower()

    # Normalizzazione degli input per evitare problemi di maiuscole/minuscole
    normalized_developers = {dev.lower(): dev for dev in developers}

    if developer_choice not in normalized_developers:
        print(f"Developer '{developer_choice}' non disponibile.")
        return

    normalized_dev_choice = normalized_developers[developer_choice]
    dev_query = f"gioco_developer(Nome, '{normalized_dev_choice}')"
    print(f"Ecco i 5 migliori giochi del developer '{normalized_dev_choice}':")

    # Recupera tutti i risultati della query
    results = list(prolog.query(dev_query))

    # Se non ci sono risultati, stampa un messaggio e ritorna
    if not results:
        print(f"Nessun gioco trovato per il developer '{normalized_dev_choice}'.")
        return

    # Messaggio di avviso se meno di 5 giochi
    if len(results) < 5:
        print(f"Attenzione: trovati solo {len(results)} top giochi per il developer '{normalized_dev_choice}'.")

    # Mostra tutti i giochi trovati per quel developer
    print("Elenco giochi:")
    for result in results:
        print(result["Nome"])

# Funzione principale per gestire il menu e le query
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
        print("3. Mostra 5 migliori giochi di un developer scelto")
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

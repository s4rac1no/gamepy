import pandas as pd
import re
from mapping_modes import classify_game_mode

# Funzione per normalizzare il nome del gioco
def normalize_game_name(name):
    # Sostituisci gli apostrofi con underscore
    name = name.replace("'", "_")
    # Sostituisci . e : con -
    name = re.sub(r'[.:]', '-', name)
    return name

# Funzione per pulire il genere
def clean_genre(genre):
    # Rimuovi gli apostrofi singoli nel genere, se presenti
    if "'" in genre:
        genre = genre.replace("'", "")
    return genre.strip()

# Funzione per estrarre l'anno dalla data di rilascio
def extract_year_from_r_date(r_date):
    # Trova una sequenza di quattro cifre alla fine della stringa
    match = re.search(r'\d{4}$', r_date)
    if match:
        return int(match.group())
    else:
        return None

# Funzione principale per convertire un CSV in un file Prolog
def csv_to_prolog(csv_filename, prolog_filename):
    # Leggi il dataset CSV usando pandas
    df = pd.read_csv(csv_filename)

    # Insiemi per tracciare i fatti già scritti
    facts_written = set()
    developers_written = set()
    modes_written = set()

    # Apri il file Prolog in modalità scrittura
    with open(prolog_filename, 'w', encoding='utf-8') as prologfile:
        # Scrivi l'intestazione nel file Prolog
        prologfile.write('% Fatti per nome gioco e genere\n\n')

        # Itera su ogni riga del dataframe
        for index, row in df.iterrows():
            nome_gioco = normalize_game_name(row['name'])
            generi = row['genre'].split(',')  # Dividi i generi usando la virgola come separatore
            generi = [clean_genre(genere) for genere in generi]  # Pulisci i generi

            # Scrivi i fatti Prolog per ogni genere del gioco se non è già stato scritto
            for genere in generi:
                fact = f"gioco_generi('{nome_gioco.strip()}', '{genere}')"
                if fact not in facts_written and genere != 'No info':
                    prologfile.write(f"{fact}.\n")
                    facts_written.add(fact)

        # Filtra i giochi con score >= 90 e almeno 100 critiche
        high_score_games = df[(df['score'] >= 90) & (df['users'] >= 100)]

        # Scrivi i fatti per i giochi con score >= 90 e almeno 100 critiche
        prologfile.write('\n% Giochi con score maggiore o uguale a 90 e almeno 100 critics\n\n')
        for index, row in high_score_games.iterrows():
            nome_gioco = normalize_game_name(row['name'])
            r_date = row['r-date']
            year = extract_year_from_r_date(r_date)
            if year:
                fact = f"gioco_top_score('{nome_gioco.strip()}', {year})"
                if fact not in facts_written:
                    prologfile.write(f"{fact}.\n")
                    facts_written.add(fact)

        # Scrivi i fatti per il developer dei giochi con score >= 90 e almeno 100 critiche
        prologfile.write('\n% Developer dei giochi con score maggiore o uguale a 90 e almeno 100 critics\n\n')
        for index, row in high_score_games.iterrows():
            nome_gioco = normalize_game_name(row['name'])
            developer = row['developer'].replace("'", "")
            fact = f"gioco_developer('{nome_gioco.strip()}', '{developer}')"
            if fact not in developers_written:
                prologfile.write(f"{fact}.\n")
                developers_written.add(fact)

        # Scrivi i fatti per le modalità di gioco
        prologfile.write('\n% Modalità di gioco\n\n')
        for index, row in df.iterrows():
            nome_gioco = normalize_game_name(row['name'])
            mode = row['players']
            if pd.notna(mode):  # Verifica se la modalità non è NaN
                mode = mode.strip()  # Pulisci la modalità di gioco
                classified_mode = classify_game_mode(mode)
                fact = f"gioco_modalita('{nome_gioco.strip()}', '{classified_mode}')"
                if fact not in modes_written:
                    prologfile.write(f"{fact}.\n")
                    modes_written.add(fact)

# Esegui la funzione principale se il file è eseguito come script principale
if __name__ == '__main__':
    csv_to_prolog('../datasets/games-data.csv', 'games_kb.pl')

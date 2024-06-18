import pandas as pd
import re
from mapping_modes import classify_game_mode
def normalize_game_name(name):
    # Sostituisci gli apostrofi con underscore
    name = name.replace("'", "_")
    # Sostituisci . e : con -
    name = re.sub(r'[.:]', '-', name)
    return name

def clean_genre(genre):
    # Rimuovi virgolette singole solo se sono contenute nel genere
    if "'" in genre:
        genre = genre.replace("'", "")
    return genre.strip()

def extract_year_from_r_date(r_date):
    # Estrarre l'anno dall'ultima voce di r-date
    match = re.search(r'\d{4}$', r_date)
    if match:
        return int(match.group())
    else:
        return None

def csv_to_prolog(csv_filename, prolog_filename):
    # Leggi il dataset CSV usando pandas
    df = pd.read_csv(csv_filename)

    # Set per tracciare i fatti già scritti
    facts_written = set()
    developers_written = set()  # Set per tracciare i developer già scritti
    modes_written = set()  # Set per tracciare le modalità già scritte

    # Apri il file Prolog in modalità scrittura
    with open(prolog_filename, 'w', encoding='utf-8') as prologfile:
        # Aggiungi la scritta iniziale nel file Prolog
        prologfile.write('% Fatti per nome gioco e genere\n\n')

        # Itera su ogni riga del dataframe
        for index, row in df.iterrows():
            # Estrai nome del gioco e generi associati
            nome_gioco = normalize_game_name(row['name'])
            generi = row['genre'].split(',')  # Split sui generi usando la virgola come separatore

            # Rimuovi spazi vuoti e pulisci i generi
            generi = [clean_genre(genere) for genere in generi]

            # Scrivi i fatti Prolog per ogni genere del gioco se non è già stato scritto
            for genere in generi:
                fact = (nome_gioco.strip(), genere)
                if fact not in facts_written:
                    prologfile.write(f"gioco_generi('{fact[0]}', '{fact[1]}').\n")
                    facts_written.add(fact)

        # Filtra i giochi con score >= 90 e almeno 100 critics
        high_score_games = df[(df['score'] >= 90) & (df['users'] >= 100)]

        # Scrivi i fatti per i giochi con score >= 90 e almeno 100 critics
        prologfile.write('\n% Giochi con score maggiore o uguale a 90 e almeno 100 critics\n\n')
        for index, row in high_score_games.iterrows():
            nome_gioco = normalize_game_name(row['name'])
            r_date = row['r-date']
            year = extract_year_from_r_date(r_date)
            if year:
                prologfile.write(f"gioco_top_score('{nome_gioco.strip()}', {year}).\n")

        # Scrivi i fatti per il developer dei giochi con score >= 90 e almeno 100 critics
        prologfile.write('\n% Developer dei giochi con score maggiore o uguale a 90 e almeno 100 critics\n\n')
        for index, row in high_score_games.iterrows():
            nome_gioco = normalize_game_name(row['name'])
            developer = row['developer'].replace("'", "")  # Rimuovi apostrofi nel developer
            # Verifica se il fatto è già stato scritto
            fact = (nome_gioco.strip(), developer)
            if fact not in developers_written:
                prologfile.write(f"gioco_developer('{fact[0]}', '{fact[1]}').\n")
                developers_written.add(fact)

         # Scrivi i fatti per le modalità di gioco
        prologfile.write('\n% Modalità di gioco\n\n')
        for index, row in df.iterrows():
            nome_gioco = normalize_game_name(row['name'])
            mode = row['players']  # Estrai la modalità di gioco
            if pd.notna(mode):  # Verifica se la modalità non è NaN
                mode = mode.strip()  # Pulisci la modalità di gioco
                classified_mode = classify_game_mode(mode)
                fact = (nome_gioco.strip(), classified_mode)
                if fact not in modes_written:
                    prologfile.write(f"gioco_modalita('{fact[0]}', '{fact[1]}').\n")
                    modes_written.add(fact)

if __name__ == '__main__':
    csv_to_prolog('../datasets/games-data.csv', 'games_kb.pl')
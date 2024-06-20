import pandas as pd
import re
from collections import Counter
from mapping_modes import classify_game_mode

# Funzione per normalizzare il nome del gioco
def normalize_name(name):
    # Rendi tutto minuscolo per uniformità
    name = name.lower()
    # Rimuovi "tom clancy's" se presente
    name = re.sub(r"tom clancy['’]s", "", name)
    # Sostituisci apostrofi con underscore
    name = name.replace("'", "_").replace("’", "_")
    # Sostituisci . e : con -
    name = re.sub(r'[.:]', '-', name)
    # Sostituisci spazi multipli o trattini con singolo underscore
    name = re.sub(r'[\s-]+', '_', name)
    # Rimuovi eventuali underscore iniziali o finali
    name = name.strip('_')
    return name

# Funzione per normalizzare il nome del developer
def normalize_developer_name(name):
    # Rimuovi apostrofi e altri caratteri speciali
    normalized_name = re.sub(r"[^a-zA-Z0-9]", "", name)
    return normalized_name.strip()

# Funzione per pulire il genere
def clean_genre(genre):
    # Rimuovi apostrofi singoli nel genere, se presenti
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
    return genre_weights.get(genre.lower(), 0.20)  # Restanti generi con peso 0.20

# Funzione principale per convertire un CSV in un file Prolog
def csv_to_prolog(csv_filename, prolog_filename, output_csv_filename, output_developer_playlist):
    # Leggi il dataset CSV usando pandas
    df = pd.read_csv(csv_filename)

    # Insiemi per tracciare i fatti già scritti
    facts_written = set()
    developers_written = set()
    modes_written = set()
    game_developer_facts_written = set()

    # Dizionario per tenere traccia dei pesi dei generi per ogni gioco
    game_genre_weights = {}

    # Apri il file Prolog in modalità scrittura
    with open(prolog_filename, 'w', encoding='utf-8') as prologfile:
        # Scrivi l'intestazione nel file Prolog
        prologfile.write('% Fatti per nome gioco e genere e il loro peso\n\n')

        # Itera su ogni riga del dataframe
        for index, row in df.iterrows():
            nome_gioco = normalize_name(row['name'])
            genres = row['genre']
            if pd.notna(genres):
                genres_list = genres.split(',')
                clean_genres_list = [clean_genre(genere) for genere in genres_list]

                unique_genres = set(clean_genres_list)

                # Calcola il peso totale dei generi per il gioco e aggiorna il dizionario
                total_weight = sum(assign_genre_weight(genere.lower()) for genere in unique_genres)
                game_genre_weights[nome_gioco] = total_weight

                # Scrivi i fatti Prolog per ogni genere unico del gioco se non è già stato scritto
                for genere in unique_genres:
                    fact = f"gioco_generi('{nome_gioco.strip()}', '{genere.strip()}')"
                    if fact not in facts_written and genere != 'No info':
                        prologfile.write(f"{fact}.\n")
                        facts_written.add(fact)

        # Scrivi i fatti per i giochi con score >= 90 e almeno 100 critiche
        high_score_games = df[(df['score'] >= 90) & (df['users'] >= 100)]

        prologfile.write('\n% Giochi con score maggiore o uguale a 90 e almeno 100 critiche\n\n')
        prologfile.write(":- discontiguous gioco_developer/2.\n")
        prologfile.write(":- discontiguous gioco_top_score/2.\n")
        for index, row in high_score_games.iterrows():
            nome_gioco = normalize_name(row['name'])
            r_date = row['r-date']
            year = extract_year_from_r_date(r_date)
            if year:
                fact = f"gioco_top_score('{nome_gioco.strip()}', {year})"
                if fact not in facts_written:
                    prologfile.write(f"{fact}.\n")
                    facts_written.add(fact)

            # Normalizza il nome del developer
            developer = normalize_developer_name(row['developer'])

            # Scrivi i fatti per il developer dei giochi con score >= 90 e almeno 100 critiche
            fact = f"gioco_developer('{nome_gioco.strip()}', '{developer}')"
            if fact not in developers_written:
                prologfile.write(f"{fact}.\n")
                developers_written.add(fact)

        # Scrivi i fatti per le modalità di gioco
        prologfile.write('\n% Modalità di gioco\n\n')
        for index, row in df.iterrows():
            nome_gioco = normalize_name(row['name'])
            mode = row['players']
            if pd.notna(mode):  # Verifica se la modalità non è NaN
                mode = mode.strip()  # Pulisci la modalità di gioco
                classified_mode = classify_game_mode(mode)
                fact = f"gioco_modalita('{nome_gioco.strip()}', '{classified_mode}')"
                if fact not in modes_written:
                    prologfile.write(f"{fact}.\n")
                    modes_written.add(fact)

        # Scrivi i fatti per il nome gioco e il developer senza ripetizioni
        prologfile.write('\n% Nome gioco e developer\n\n')
        for index, row in df.iterrows():
            nome_gioco = normalize_name(row['name'])
            developer = normalize_developer_name(row['developer'])

            # Costruisci il fatto nel formato che include il nome del gioco e il nome del developer
            fact = f"game_developer_fact('{nome_gioco.strip()}', '{developer}')"

            # Scrivi il fatto solo se non è già stato scritto
            if fact not in game_developer_facts_written:
                prologfile.write(f"{fact}.\n")
                game_developer_facts_written.add(fact)

    # Aggiungi la regola per i pesi dei generi alla fine del file Prolog
    with open(prolog_filename, 'a', encoding='utf-8') as prologfile:
        prologfile.write('\n% Regole per i pesi dei generi\n\n')
        for game, weight in game_genre_weights.items():
            if weight >= 2.40:
                fact = f"gioco_peso_generi('{game.strip().lower()}', {weight:.2f})"
                if fact not in facts_written:
                    prologfile.write(f"{fact}.\n")
                    facts_written.add(fact)

    # Aggiungi la colonna trending_game al DataFrame originale e salva il CSV aggiornato
    df['trending_game'] = df['name'].apply(lambda x: 1 if game_genre_weights.get(normalize_name(x), 0) >= 2 else 0)

    # Filtra i developer dei giochi di tendenza e normalizza i nomi
    trending_developers = df[df['trending_game'] == 1]['developer'].apply(normalize_developer_name).tolist()

    # Conta la frequenza dei developer
    developer_counter = Counter(trending_developers)

    # Ordina i developer per frequenza (da più a meno)
    sorted_developers = sorted(developer_counter.items(), key=lambda x: x[1], reverse=True)

    # Crea il DataFrame per la playlist dei developer
    developer_playlist_df = pd.DataFrame(sorted_developers, columns=['Developer', 'Trending Games Count'])

    # Salva il DataFrame della playlist dei developer in un CSV
    developer_playlist_df.to_csv(output_developer_playlist, index=False)

    print("\n\nLa playlist dei top developer è stata salvata in /datasets/trending_developers_playlist.csv ")

# Esegui la funzione principale se il file è eseguito come script principale
if __name__ == '__main__':
    csv_to_prolog('../datasets/games-data.csv', 'games_kb.pl', '../datasets/games-data_KB.csv', '../datasets/trending_developers_playlist.csv')

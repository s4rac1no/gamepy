import pandas as pd

# Lista delle modalità
modes = [
    '1 Player', '1-2 ', 'No Online Multiplayer', 'Up to 32 ', 'Up to 30 ',
    'Up to 16 ', 'No info', '1-4 ', 'Online Multiplayer', 'Up to 6 ', 'Up to 4 ',
    '1-16 ', 'Up to 8 ', '2 ', 'Up to 18 ', '1-8 ', '8  Online', '2  Online',
    '4  Online', '1-32 ', 'Up to 10 ', 'Up to 40 ', 'Massively Multiplayer',
    'Up to 12 ', 'Up to 5 ', '64  Online', 'Up to 20 ', '6  Online',
    '16  Online', '24  Online', '1-6 ', '1-12 ', '64+ ', 'Up to 22 ', 'Up to 60 ',
    'Up to 64 ', '1-10 ', 'Up to 24 ', 'Up to 3 ', '12  Online', '10  Online',
    '32  Online', '1-64 ', 'Up to more than 64 ', '14  Online', '44  Online',
    '1-5 ', '64+  Online', '1-3 ', '5  Online', 'Up to 14 ', 'Up to 9 ',
    'Up to 36 ', '1-24 ', '3  Online'
]


def classify_game_mode(mode):
    singleplayer_modes = ['1 Player', '1-2 ', 'No online Multiplayer']
    co_op_modes = ['1-2']
    game_no_mode = ['No info']
    if mode in singleplayer_modes:
        return 'singleplayer'
    elif mode in co_op_modes:
        return 'co-op mode'
    elif mode in game_no_mode:
        return 'no mode'
    else:
        return 'multiplayer'


# funzione utile per stampare le varie modalità dei giochi e creare la lista modes
# (utile per lo sviluppo)
def extract_and_print_game_modes(csv_filename):
    # Leggi il dataset CSV usando pandas
    df = pd.read_csv(csv_filename)

    # Estrai i valori unici dalla colonna 'players'
    unique_modes = df['players'].dropna().unique()

    # Stampa i vari tipi di modalità di gioco
    print("Tipi di modalità di gioco:")
    for mode in unique_modes:
        print(mode)

if __name__ == '__main__':
    extract_and_print_game_modes('../datasets/games-data.csv')
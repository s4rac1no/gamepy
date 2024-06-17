## Indice

1. [Introduzione](#1-introduzione)
2. [Struttura del progetto](#2-struttura-del-progetto)
3. [Requisiti per eseguire il progetto](#3-requisiti-per-eseguire-il-progetto)

## 1. Introduzione

Il progetto è stato creato da:

- Saracino Lorenzo
- Rosmarino Fabrizio

## 2. Struttura del progetto


## 3. Requisiti per eseguire il progetto

Per eseguire il progetto è necessario installare le seguenti componenti:

- `Python 3.9.*` -> https://www.python.org/downloads/
- `Prolog 8.2.4` -> https://www.swi-prolog.org/download/stable?show=all

Si è scelto la versione 8.2.4 di Prolog perché questa è risultata la più stabile per l'utilizzo in python con la libreria `pyswip`, inoltre è risultata efficiente per i dispositivi Windows e macOS. 

Per l'avvio del progetto seguire i seguenti passaggi:

- Aprire il terminale
- Spostarsi nel path(directory) del progetto.

Digitare infine i seguenti comandi:

- `$ python -m venv venv`
  
- Attivare l'ambiente virtuale:
  
  |  Windows                    | Macos                       |
  |-----------------------------|-----------------------------|
  | `$ venv\Scripts\activate`   | `$ source venv/bin/activate`|

- `$ pip install -r requirements.txt`
  
**N.B.** Per gli utenti che utilizzano **macOS**, per risolvere il problema presentato all'avvio del file sorgente useKB.py, 
`atom_chars/2, problem caused by pyswip`, procedere con l'installazione manuale della libreria _pyswip_ digitanto questo
comando nel terminale, nella direcotory del progetto:

- `pip install git+https://github.com/yuce/pyswip@master#egg=pyswip`

_Questo installerà l'ultima versione di pyswip che ha risolto il problema su Mac._









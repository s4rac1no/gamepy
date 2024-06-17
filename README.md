Il progetto è stato creato da:

- Saracino Lorenzo
- Rosmarino Fabrizio

**Struttura del progetto**


**Requisiti per eseguire il progetto:**

Per eseguire il progetto è necessario installare le seguenti componenti:

- <p> Python 3.9.* </p>
- <p> Prolog 8.2.4 </p>

Si è scelto la versione 8.2.4 di Prolog perché questa è risultata la più stabile per l'utilizzo in python con la librerie <p> pyswip </p>, 
inoltre è risultata efficiente per i dispositivi Windows e macOS. 

Per l'avvio del progetto seguire i seguenti passaggi:

- Aprire il terminale
- Spostarsi nel path(directory) del progetto.

Digitare infine i seguenti comandi:

- $ <p> python -m venv venv </P>       

- |  Windows                                 | Macos                                    |
  |------------------------------------------|------------------------------------------|
  | $ <p> venv\Scripts\activate </p>         | $ <p> source venv/bin/activate </p>      |

- $ pip install -r requirements.txt
  
**N.B.** Per gli utenti che utilizzano **macOS**, per risolvere il problema presentato all'avvio del file sorgente useKB.py, 
<p> atom_chars/2, problem caused by pyswip </p>, procedere con l'installazione manuale della libreria _pyswip_ digitanto questo
comando nel terminale, nella direcotory del progetto:

- <p> pip install git+https://github.com/yuce/pyswip@master#egg=pyswip </p>

_Questo installerà l'ultima versione di pyswip che ha risolto il problema su Mac._









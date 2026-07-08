# Taxi-KI

# Aufgabenstellung und Gesamtsituation klären

## Pakete Installieren (Später geht das weg)
pip install -r requirements.txt

## Hauptziel
Wie gut lässt sich die Fahrtdauer einer Taxifahrt in New York City anhand der in 2019 erhobenen Daten vorhersagen?

## Nebenziele
Mögliche Frage: Wo soll das Taxi sich aufgrund der Vergangenheit hinbewegen, damit es schnell den nächsten kunden kriegt

## Qualitätsmetriken bestimmen
Durchschnittlicher Fehler (Abweichung) der tatsächlichen Fahrtzeit in Minuten.

## MVP definieren
Die Benutzer des Models bekommen auf Basis der Daten eine möglichst genaue Vorhersage wie lange eine Fahrt dauert.
Dadurch soll durch die Daten unser Modell die geplante Fahrtzeit ausgeben. 

## ML-Aufgabe
Überwachtes Lernen mit Regression

# Datenakquise

Datenakquise kann hauptsächlich übersprungen werden, da dies schon durch der ausgewählten Datenmenge passiert ist.

Die Ausnahme ist die Aufteilung der Testdaten, was in [train_test_split.py](train_test_split.py) gemacht wird.

# Datenexploration

Datenexploration wird hauptsächlich in [Datenexploration.ipynb](Datenexploration.ipynb) ausgeführt.

# Datenaufbereitung

## KI Pipeline

### Entfernte Spalten

- **vendorid** – Anbieter für Daten, nicht wichtig wie schnell gefahren wird
- **passenger_count** – Passagieranzahl hat kein oder minimal Einfluss in einer Innenstadt
- **ratecodeid** – Bezahlungsrate, nicht relevant für Fahrtdauer
- **store_and_fwd_flag** – Zwischenspeicher bei Internetproblemen, irrelevant für Fahrtdauer
- **payment_type** – Bezahlungsart, irrelevant für Fahrtdauer
- **fare_amount** – Bezahlungsmenge, irrelevant für Fahrtdauer
- **extra** – verschiedene Extrakosten, irrelevant für Fahrtdauer
- **mta_tax** – Steuer, irrelevant für Fahrtdauer
- **tip_amount** – Trinkgeldmenge, irrelevant für Fahrtdauer
- **tolls_amount** – Mautgebührenmenge, irrelevant für Fahrtdauer
- **improvement_surcharge** – Extra Gebühr zur Verbesserung, irrelevant für Fahrtdauer
- **total_amount** – Summe der Bezahlung, irrelevant für Fahrtdauer
- **congestion_surcharge** – Extrakosten für Stau, irrelevant für Fahrtdauer

### Behandlung von falschen Daten

- Falsche Daten in entfernten Spalten werden ignoriert
- Sonst allgemeine Filterung von unglaubwürdigen Daten

### Spalten und was transformiert werden muss

- **pickup_time**
  - Innerhalb von 2019
  - Kein Null
- **dropoff_time**
  - Innerhalb von 2019
  - Kein Null
- **trip_distance**
  - Keine negativen Werte
  - Kein Null
  - Min. 0.1 Meilen (160 Meter, darunter ist eine unrealistische Fahrt)
  - Max. 40 Meilen (New York City ist 37 Meilen lang)
- **pulocationid**
  - Kein Null
  - 1–263 (inklusive; darunter ist falsch, darüber ist außerhalb New York City (~1% der Daten))
- **dolocationid**
  - Kein Null
  - 1–263 (inklusive; darunter ist falsch, darüber ist außerhalb New York City (~1% der Daten))

### Feature Engineering

- **duration** = dropoff_time − pickup_time (in Sekunden)
  - Min. Duration: 10 Sekunden (minimale Distanz ist 0.1 Meilen = 160 Meter, schneller als 10 Sekunden ist unrealistisch)
  - Max. Duration: 10800 Sekunden (3 Stunden)
- **speed (average)** = trip_distance / (duration / 3600)
  - Min. Speed: 0.3 (0.3 mph ≈ 0.48 km/h)
  - Max. Speed: 60 (60 mph ≈ 96 km/h)

Datenaufbereitung wird hauptsächlich in [pipeline_data_cleaning.py](pipeline_data_cleaning.py) gemacht. Während Skalierungen und Encodierungen in [creating_encoder_scaler.py](creating_encoder_scaler.py) passieren.


# Modelltraining

Modelltraining wird in [pipeline_model_trainer.py](pipeline_model_trainer.py) ausgeführt.

# Modellbewertung

Die Bewertung des Modells passiert in [pipeline_model_evaluator](pipeline_model_evaluator.py).
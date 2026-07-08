# Taxi-KI

## Installation
Run the following line to install the requirements:
```
pip install -r requirements.txt
```

## Creating the model
Put the databases of the NYC taxi trips of 2019 into the <b>databases/raw</b> folder next to
the <b>RAW DATABASES HERE</b> file and run
```
python3 train_test_split.py
```
to split the dataset into a training and testing data.

Then run
```
python3 pipeline_data_cleaning.py
```
to clean and transform both the train.sqlite and test.sqlite.

To train the model, run
```
python3 pipeline_model_trainer.py
```
Note that this requires about 14 GB of RAM.

Finally, to evaluate the model, run
```
python3 pipeline_model_evaluator.py
```
Note that this requires even more RAM.
Around 23 GB of RAM, likely caused by a load overhead caused by pandas.read_sql_query.

### TODO
Write a script / pipeline that accepts input, transforms it, feeds it to the model and shows the predicted result.

## Dokumentation
Eine Zusammenfassung unserer Dokumentation ist in dem [LabNotes.md](LabNotes.md) Dokument, dies beinhaltet dann entweder die relevanten Entscheidungen/Ideen oder verweist dann auf die dazugehörigen Dateien mit Kontext.

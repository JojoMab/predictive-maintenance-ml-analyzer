# Predictive Maintenance ML Analyzer

![Python CI](https://github.com/JojoMab/predictive-maintenance-ml-analyzer/actions/workflows/python-ci.yml/badge.svg)

Predictive Maintenance ML Analyzer ist ein Bewerberprojekt für duale Studiengänge in Informatik, Wirtschaftsinformatik, Data Science und KI-nahen Themen. Das Projekt trainiert ein einfaches Klassifikationsmodell mit scikit-learn auf synthetischen Sensordaten und bewertet, ob eine Wartung wahrscheinlich notwendig ist.

## Bewerbungskontext

Das Projekt zeigt Grundlagen von Data Science, Feature Engineering, Modelltraining und Evaluation. Es eignet sich besonders für Bewerbungen bei MTU, Siemens Energy, Infineon, MAN, Rohde & Schwarz, ASMPT, Atos und weiteren technischen oder datengetriebenen Studienangeboten.

## Tech Stack

- Python 3
- pandas
- numpy
- scikit-learn
- Logistic Regression
- Train/Test Split
- Accuracy, Precision, Recall, F1
- Confusion Matrix
- pytest
- GitHub Actions CI

## Funktionen

- synthetische Sensordaten erzeugen
- Temperatur, Vibration, Druck, Laufzeit und Fehleranzahl verarbeiten
- neue Features wie `temp_vib_ratio` und `runtime_error_factor` berechnen
- Daten in Trainings- und Testdaten aufteilen
- einfaches Klassifikationsmodell trainieren
- Modell als Datei speichern
- Accuracy, Precision, Recall, F1 und Confusion Matrix berechnen
- Evaluationsreport als Textdatei erzeugen
- zentrale Funktionen mit Tests prüfen

## Projektstruktur

```text
predictive-maintenance-ml-analyzer/
├── main.py
├── data_generator.py
├── feature_engineering.py
├── model_trainer.py
├── model_evaluator.py
├── requirements.txt
├── data/
│   └── sensor_data.csv
├── reports/
│   └── evaluation_report.txt
├── tests/
│   ├── test_features.py
│   └── test_model.py
├── docs/
│   ├── application_fit.md
│   └── recruiter_summary_de.md
└── .github/workflows/python-ci.yml
```

## Schnellstart

```bash
python3 -m pip install -r requirements.txt
python3 main.py
```

## Tests

```bash
python3 -m pytest tests/ -v
```

## Beispielausgabe

```text
Predictive maintenance pipeline completed.
Accuracy: 0.82
Precision: 0.78
Recall: 0.74
Report: reports/evaluation_report.txt
```

## Hinweis auf synthetische Daten

Alle Daten werden künstlich erzeugt und dienen nur zur nachvollziehbaren Demonstration. Das Projekt bildet kein echtes Unternehmenssystem ab und behauptet keine echte Praxiserfahrung.

## English Summary

This applicant project demonstrates a small predictive maintenance workflow with synthetic sensor data. It uses scikit-learn for a simple classification model and reports accuracy, precision, recall, F1 and a confusion matrix. The project is intentionally scoped for dual study applications and avoids exaggerated AI claims.

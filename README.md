![Python CI](https://github.com/JojoMab/predictive-maintenance-ml-analyzer/actions/workflows/python-ci.yml/badge.svg)

# Predictive Maintenance ML Analyzer

Dieses Bewerberprojekt zeigt ein einfaches Klassifikationsmodell mit scikit-learn auf synthetischen Sensordaten. Es simuliert eine kleine Predictive-Maintenance-Aufgabe mit Temperatur, Vibration, Druck, Laufzeit, Fehleranzahl und der Zielvariable `maintenance_needed`.

## Bewerbungskontext

Das Projekt passt zu dualen Studiengängen in Data Science, KI, Informatik und technischer Informatik. Es ist besonders relevant für MTU, Siemens Energy, Infineon, MAN und Rohde & Schwarz, weil technische Sensordaten, Feature Engineering und Modellbewertung sichtbar werden.

## Tech Stack

- Python 3.11
- pandas
- numpy
- scikit-learn
- matplotlib
- pytest/unittest-kompatible Tests
- GitHub Actions

## Funktionen

- 500 bis 1000 synthetische Sensordatenpunkte erzeugen
- Feature Engineering mit zusätzlichen Merkmalen
- Logistic Regression trainieren
- Train/Test Split 80/20 verwenden
- Accuracy, Precision, Recall, F1 und Confusion Matrix berechnen
- Evaluation Report als Textdatei erzeugen

## Projektstruktur

```txt
predictive-maintenance-ml-analyzer/
├── main.py
├── data_generator.py
├── feature_engineering.py
├── model_trainer.py
├── model_evaluator.py
├── requirements.txt
├── data/
├── reports/
├── tests/
└── docs/
```

## Schnellstart

```bash
python -m pip install -r requirements.txt
python main.py
```

## Tests ausführen

```bash
python -m pytest tests/ -v
```

## Beispielausgabe

```txt
Predictive-Maintenance-Pipeline abgeschlossen.
Accuracy: 0.850
Precision: 0.780
Recall: 0.720
Report: reports/evaluation_report.txt
```

## Hinweis auf synthetische Daten

Alle Daten sind synthetisch und dienen ausschließlich der Demonstration.

## English Summary

This applicant project demonstrates a small machine learning pipeline for predictive maintenance with synthetic sensor data. It uses feature engineering, train/test split, logistic regression and standard evaluation metrics. The project is designed to show data science fundamentals without overstating the scope.

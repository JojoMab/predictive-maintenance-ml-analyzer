# Predictive Maintenance ML Analyzer

Python-Projekt fuer synthetische Maschinendaten mit Feature Engineering, einfacher Risikoklassifikation und Modell-Evaluation auf Bewerberniveau.

## Kurzprofil fuer Recruiter

Dieses Repository ist ein Bewerberprojekt fuer duale Studiengaenge in Wirtschaftsinformatik, Informatik, Data Science und KI-nahen Themen. Es nutzt synthetische Daten, eine bewusst einfache Projektstruktur und nachvollziehbare Reports, damit fachliche und technische Grundlagen schnell erkennbar sind.

## Bewerbungsbezug

Das Projekt zeigt Data-Science-Grundlagen, technische Sensordaten, Health Score, Wartungsrisiko und Evaluation ohne uebertriebene KI-Claims.

## Passende Zielunternehmen

MTU, SIXT, ASMPT, Atos, Infineon, Siemens Energy, Truma und Data-Science-/KI-nahe duale Studiengaenge.

## Tech Stack

- Python 3
- CSV-Verarbeitung
- regelbasierte Analyse und einfache Kennzahlen
- Unit Tests mit unittest
- GitHub Actions CI

## Bewerbungsbezug: SIXT

Das Projekt kann fuer SIXT als datenbasierte Entscheidungsunterstuetzung im Mobility- und Flottenumfeld erklaert werden. Es baut kein echtes SIXT-System nach, sondern zeigt auf Bewerberniveau, wie technische Zustandsdaten einer Fahrzeug- oder Maschinenflotte bewertet und fuer Wartungsentscheidungen genutzt werden koennen.

## Funktionen

- synthetische CSV-Daten laden
- Eingabedaten validieren
- Feature Engineering fuer Temperatur, Vibration, Druck und Laufzeit
- einfachen Health Score berechnen
- Wartungsrisiko einstufen
- einfaches Modell trainieren
- Accuracy, Precision, Recall und Confusion Matrix ausgeben
- Report als Text ausgeben
- Beispielausgabe versionieren

## Projektstruktur

```txt
predictive-maintenance-ml-analyzer/
├── main.py
├── data/
├── docs/
│   ├── application_fit.md
│   └── recruiter_summary_de.md
├── examples/
│   └── terminal_output.txt
├── tests/
└── .github/workflows/python-ci.yml
```

## Schnellstart

```bash
python3 main.py
```

## Tests

```bash
python3 -m unittest discover -s tests
```

## Hinweis

Alle Daten sind synthetisch. Das Projekt bildet kein echtes Unternehmenssystem ab und behauptet keine echte Praxiserfahrung.

## English Summary

Applicant portfolio project using synthetic data to demonstrate basic software structure, data processing, reporting and business/IT understanding.

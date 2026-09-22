# ResQPH ML/ETL

Data engineering and machine learning pipeline for **ResQPH**, a flood-aware emergency rescue coordination system.

This repository contains the **ETL pipeline**, **feature engineering**, **model training**, and **model export** for the ResQPH academic prototype. It does **not** contain frontend or backend application code. Trained models and processed artifacts are exported for deployment in the main ResQPH backend repository.

> **Scope:** This repository focuses on preparing geospatial data and training predictive models. It is not an official emergency-response or flood-forecasting platform.

---

## Table of Contents

- [Overview](#overview)
- [Scope](#scope)
- [Repository Structure](#repository-structure)
- [Data Sources](#data-sources)
- [ETL Pipeline](#etl-pipeline)
- [Machine Learning Models](#machine-learning-models)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Model Export and Deployment](#model-export-and-deployment)
- [Development Workflow](#development-workflow)
- [Team Responsibilities](#team-responsibilities)
- [License](#license)
- [Disclaimer](#disclaimer)

---

## Overview

ResQPH connects stranded individuals with rescue teams during severe flooding. The system combines rescue request management, interactive mapping, flood-aware routing, limited machine learning, and offline synchronization.

This repository provides the **ML/ETL component** of ResQPH:

- **ETL pipeline** for OpenStreetMap roads, flood-hazard layers, elevation data, and ResQPH-generated operational data.
- **NLP-based incident triage** to extract priority signals from free-text citizen reports.
- **Road-risk/passability modeling** to produce a probabilistic score used as a routing penalty.
- **Model export** for integration into the main ResQPH backend.

The central engineering principle is that AI assists the system while deterministic routing, transparent data, and human rescue decisions remain responsible for the operational workflow.

---

## Scope

This repository covers:

- Data extraction, cleaning, transformation, and geospatial alignment.
- Feature engineering for road-risk and incident-triage models.
- Training and evaluation of explainable baseline models.
- Exporting trained models and metadata for backend deployment.

This repository does **not** cover:

- Frontend interfaces.
- Backend API implementation.
- MongoDB production setup.
- Full hydrological modeling or real-time flood forecasting.
- Mesh, satellite, or radio communication.

---

## Repository Structure

```text
resqph-ml-etl/
├── data/
│   ├── raw/                  # Original, immutable datasets
│   ├── interim/              # Intermediate processed data
│   └── processed/            # Final datasets for modeling
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
├── src/
│   ├── etl/
│   │   ├── extract.py        # Data extraction from sources
│   │   ├── transform.py      # Cleaning, alignment, feature engineering
│   │   ├── load.py           # Save processed data
│   │   └── pipeline.py       # Full ETL pipeline runner
│   ├── ml/
│   │   ├── nlp_triage.py     # NLP-based incident triage
│   │   ├── road_risk.py      # Road-risk/passability model
│   │   └── evaluate.py       # Model evaluation utilities
│   └── utils/
│       ├── geospatial.py     # Geospatial helper functions
│       └── config.py         # Configuration management
├── models/                   # Saved model artifacts
├── tests/                    # Unit and integration tests
├── requirements.txt
├── README.md
└── .gitignore

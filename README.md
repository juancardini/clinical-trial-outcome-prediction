# Prediction of Clinical Trial Outcomes Using Graph Metrics Integrated into Machine Learning Models

This repository contains the code developed for the Master's Thesis in Data Science:

**Prediction of Clinical Trial Outcomes Using Graph Metrics Integrated into Machine Learning Models**

---

# Technologies

- Azure Databricks
- PySpark
- Spark SQL
- Neo4j Desktop
- Cypher
- Neo4j Graph Data Science (GDS)
- Python

---

# Repository Structure

```text
clinical-trial-outcome-prediction-graph-ml
│
├── notebooks
│   ├── 01_Bronze_Layer.ipynb
│   ├── 02_Silver_Layer.ipynb
│   ├── 03_Gold_Layer.ipynb
│   ├── 04_Exploratory_Data_Analysis.ipynb
│   ├── 05_Export_Data.ipynb
│   ├── 09_Feature_Engineering.ipynb
│   └── 10_Machine_Learning_Models.ipynb
│
├── neo4j
│   ├── 06_graph_creation.cypher
│   ├── 07_graph_metrics.cypher
│   └── 08_export_graph_metrics.cypher
└── README.md
```

---

# Project Description

The objective of this project is to develop a Machine Learning model capable of predicting clinical trial outcomes using data from **ClinicalTrials.gov (AACT)** enriched with graph structural metrics extracted from a knowledge graph built in **Neo4j**.

The project implements a complete pipeline consisting of:

- Data extraction and transformation using Azure Databricks and PySpark.
- Construction of a knowledge graph in Neo4j.
- Graph analytics using Neo4j Graph Data Science.
- Feature Engineering.
- Machine Learning model training and evaluation.

---

# ELT Pipeline

The notebooks in this section implement the **Extract, Load and Transform (ELT)** process used to prepare ClinicalTrials.gov (AACT) data before building the knowledge graph and training the Machine Learning models.

---

## Notebook 01: `01_Bronze_Layer.ipynb`

This notebook implements the **Extract & Load** stage.

### Functionality

- Connect to the AACT public database using JDBC.
- Read tables from the `ctgov` schema using PySpark.
- Store raw data as Delta tables in the Bronze layer.

### Tables Loaded

- bronze.raw_studies
- bronze.raw_conditions
- bronze.raw_countries
- bronze.raw_interventions
- bronze.raw_sponsors
- bronze.raw_designs
- bronze.raw_facilities

---

## Notebook 02: `02_Silver_Layer.ipynb`

This notebook performs the transformation stage.

### Functionality

- Filter diabetes clinical trials.
- Select interventional studies.
- Clean and normalize data.
- Handle missing values.
- Standardize countries, conditions and interventions.
- Create the target variable (`approved`).

### Output Tables

- silver.studies_diabetes
- silver.conditions
- silver.countries
- silver.interventions
- silver.sponsors
- silver.designs
- silver.facilities

---

## Notebook 03: `03_Gold_Layer.ipynb`

This notebook prepares the data required to build the knowledge graph in Neo4j.

### Functionality

- Generate node tables.
- Generate relationship tables.
- Export CSV files for Neo4j.

### Nodes

- Studies
- Conditions
- Countries
- Interventions
- Sponsors
- Facilities
- Designs

### Relationships

- HAS_CONDITION
- HAS_INTERVENTION
- HAS_SPONSOR
- HAS_DESIGN
- HAS_FACILITY
- DEVELOPED_IN_COUNTRY

---

## Notebook 04: `04_Exploratory_Data_Analysis.ipynb`

This notebook performs exploratory data analysis on the Gold layer before importing the data into Neo4j.

### Objective

Validate the quality and consistency of the generated entities and relationships before graph construction.

---

## Notebook 05: `05_Export_Data.ipynb`

Exports the Gold layer tables to CSV files that will be imported into Neo4j.

### Functionality

- Export node tables.
- Export relationship tables.
- Save CSV files into an Azure Databricks Volume.

### Generated Files

#### Nodes

- nodes_studies.csv
- nodes_conditions.csv
- nodes_countries.csv
- nodes_interventions.csv
- nodes_sponsors.csv
- nodes_designs.csv
- nodes_facilities.csv

#### Relationships

- rels_study_condition.csv
- rels_study_country.csv
- rels_study_intervention.csv
- rels_study_sponsor.csv
- rels_study_design.csv
- rels_study_facility.csv

The generated CSV files are copied into the **Neo4j import** directory and loaded using the `LOAD CSV` Cypher command.

---

# Neo4j

## Script: `06_graph_creation.cypher`

Builds the complete knowledge graph.

### Functionality

- Create constraints.
- Import nodes.
- Import relationships.
- Validate graph construction.

### Node Labels

- Study
- Condition
- Country
- Intervention
- Sponsor
- Design
- Facility

### Relationship Types

- HAS_CONDITION
- HAS_INTERVENTION
- HAS_SPONSOR
- HAS_DESIGN
- HAS_FACILITY
- DEVELOPED_IN_COUNTRY

---

## Script: `07_graph_metrics.cypher`

Projects the graph using Neo4j Graph Data Science and computes graph metrics.

### Functionality

- Graph projection.
- Degree Centrality.
- Closeness Centrality.
- Betweenness Centrality.
- PageRank.
- Louvain Community Detection.
- Metric normalization.

The calculated metrics are written back to the Study nodes.

---

## Script: `08_export_graph_metrics.cypher`

Exports graph metrics from Neo4j into a CSV file.

### Exported Variables

- Degree Centrality
- Closeness Centrality
- Betweenness Centrality
- PageRank
- Louvain Community

The generated CSV file is later used during the Feature Engineering stage.

---

## Notebook 09: `09_Feature_Engineering.ipynb`

Builds the final Machine Learning dataset.

### Functionality

- Read graph metrics exported from Neo4j.
- Read Gold layer tables.
- Merge graph metrics with clinical trial information.
- Compute aggregated variables.
- Generate the final Machine Learning dataset.

### Output

- `gold.dataset_ml`

This dataset combines clinical variables, graph metrics and engineered features.

---

## Notebook 10: `10_Machine_Learning_Models.ipynb`

Implements the training and evaluation of the Machine Learning models.

### Functionality

- Read the final dataset.
- Prepare predictor and target variables.
- Split the data into training and testing sets.
- Preprocess numerical, categorical and textual features.
- Train and optimize Machine Learning models.
- Evaluate model performance.
- Compare different feature configurations.

### Models

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- XGBoost

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Average Precision

The models are compared to evaluate the contribution of graph-derived features to clinical trial outcome prediction.

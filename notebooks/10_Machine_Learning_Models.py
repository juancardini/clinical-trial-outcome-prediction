# Databricks notebook source
# MAGIC %md Import Dataset

# COMMAND ----------

# DBTITLE 1,Untitled
df_spark = spark.table("gold.dataset_ml")
df = df_spark.toPandas()
df = df[df["approved"].notna()].copy()
df["approved"] = df["approved"].astype(int)
# Remove embeddings
emb_features = [c for c in df.columns if c.startswith("emb_")]
df = df.drop(columns=emb_features)

print("Rows:", len(df))
print("\nTarget distribution:")
print(df["approved"].value_counts())
print("\nTarget proportion:")
print(df["approved"].value_counts(normalize=True))
display(df)

# COMMAND ----------

# MAGIC %md Define fields groups

# COMMAND ----------

# Community Louvain Feature
df["community_louvain"] = df["community_louvain"].astype(int)
community_counts = df["community_louvain"].value_counts()
df["community_size"] = df["community_louvain"].map(community_counts).fillna(0)

# Graph Features
graph_features = [
    "degree_total",
    "degree_norm",
    "pagerank_norm",
    "betweenness_norm",
    "closeness_norm",
    "community_size"
]



# Variables BASE
numeric_features_zero = ["n_facilities", "n_countries"]

numeric_features_other = [
    "n_interventions",
    "n_sponsor_lead",
    "n_sponsor_collaborator",
    "n_sponsors_total",
    "n_conditions"
]

categorical_features = [
    "allocation",
    "primary_purpose",
    "masking",
    "intervention_model",
    "is_blinded"
]

text_feature = "brief_title"

df[categorical_features] = df[categorical_features].fillna("UNKNOWN").astype(str)
df[text_feature] = df[text_feature].fillna("").astype(str)

numeric_features_full = (
    numeric_features_zero +
    numeric_features_other +
    graph_features 
)




# COMMAND ----------

# MAGIC %md Define X and y

# COMMAND ----------

id_cols = ["study_id"]

y = df["approved"].astype(int)
#Base
X_base = df.drop(
    columns=graph_features  + [text_feature, "approved", "community_louvain"] + id_cols,
    errors="ignore"
)
# Tabular + Text 
X_text = df.drop(
    columns=graph_features  + ["approved", "community_louvain"] + id_cols,
    errors="ignore"
)

# Tabular + Graph
X_graph = df.drop(
    columns=  [text_feature, "approved", "community_louvain"] + id_cols,
    errors="ignore"
)



# Tabular + Text + Graph 
X_graph_text = df.drop(
    columns=["approved", "community_louvain"] + id_cols,
    errors="ignore"
)
print("BASE:", list(X_base.columns))
print("TEXT:", list(X_text.columns))
print("GRAPH:", list(X_graph.columns))

print("FULL:", list(X_graph_text.columns))

# COMMAND ----------

# MAGIC %md Train/Test

# COMMAND ----------

from sklearn.model_selection import train_test_split

# Split data
idx_train, idx_test = train_test_split(
    df.index,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# BASE
X_base_train = X_base.loc[idx_train]
X_base_test  = X_base.loc[idx_test]

# TABULAR + TEXT
X_text_train = X_text.loc[idx_train]
X_text_test  = X_text.loc[idx_test]

# TABULAR + GRAPH

X_graph_train = X_graph.loc[idx_train]
X_graph_test  = X_graph.loc[idx_test]



# FULL
# Tabular + texto + métricas 
X_graph_text_train = X_graph_text.loc[idx_train]
X_graph_text_test  = X_graph_text.loc[idx_test]

# TARGET
y_train = y.loc[idx_train]
y_test  = y.loc[idx_test]

# Trazability
study_train = df.loc[idx_train, "study_id"]
study_test  = df.loc[idx_test, "study_id"]

print("Train size:", len(y_train))
print("Test size:", len(y_test))

print("\nTrain target distribution:")
print(y_train.value_counts(normalize=True))

print("\nTest target distribution:")
print(y_test.value_counts(normalize=True))

# COMMAND ----------

# MAGIC %md Preprocess

# COMMAND ----------

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

text_transformer = Pipeline(steps=[
    ("to_1d", FunctionTransformer(
        lambda x: (
            x.iloc[:, 0]
            if isinstance(x, pd.DataFrame)
            else (x.ravel() if isinstance(x, np.ndarray) else x)
        ).fillna("").astype(str),
        validate=False
    )),
    ("tfidf", TfidfVectorizer(
        max_features=5000,        
        ngram_range=(1, 2),
        min_df=5,                
        max_df=0.8,              
        stop_words="english"
    ))
])

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# TREE: RF / XGB

preprocess_tree_base = ColumnTransformer(
    transformers=[
        ("num_zero", SimpleImputer(strategy="constant", fill_value=0), numeric_features_zero),
        ("num_other", SimpleImputer(strategy="median"), numeric_features_other),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features)
    ],
    remainder="drop",
    sparse_threshold=1.0
)

preprocess_tree_text = ColumnTransformer(
    transformers=[
        ("num_zero", SimpleImputer(strategy="constant", fill_value=0), numeric_features_zero),
        ("num_other", SimpleImputer(strategy="median"), numeric_features_other),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features),
        ("text", text_transformer, [text_feature])
    ],
    remainder="drop",
    sparse_threshold=1.0
)

preprocess_tree_graph = ColumnTransformer(
    transformers=[
        ("num_zero", SimpleImputer(strategy="constant", fill_value=0), numeric_features_zero),
        ("num_other", SimpleImputer(strategy="median"),
         numeric_features_other + graph_features ),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features)
    ],
    remainder="drop",
    sparse_threshold=1.0
)

preprocess_tree_graph_text = ColumnTransformer(
    transformers=[
        ("num_zero", SimpleImputer(strategy="constant", fill_value=0), numeric_features_zero),
        ("num_other", SimpleImputer(strategy="median"),
         numeric_features_other + graph_features ),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features),
        ("text", text_transformer, [text_feature])
    ],
    remainder="drop",
    sparse_threshold=1.0
)

# LINEAR: Logistic Regression / SVM

preprocess_linear_base = ColumnTransformer(
    transformers=[
        ("num_zero", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler(with_mean=False))
        ]), numeric_features_zero),
        ("num_other", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False))
        ]), numeric_features_other),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features)
    ],
    remainder="drop",
    sparse_threshold=1.0
)

preprocess_linear_text = ColumnTransformer(
    transformers=[
        ("num_zero", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler(with_mean=False))
        ]), numeric_features_zero),
        ("num_other", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False))
        ]), numeric_features_other),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features),
        ("text", text_transformer, [text_feature])
    ],
    remainder="drop",
    sparse_threshold=1.0
)

preprocess_linear_graph = ColumnTransformer(
    transformers=[
        ("num_zero", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler(with_mean=False))
        ]), numeric_features_zero),
        ("num_other", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False))
        ]), numeric_features_other + graph_features ),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features)
    ],
    remainder="drop",
    sparse_threshold=1.0
)

preprocess_linear_graph_text = ColumnTransformer(
    transformers=[
        ("num_zero", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler(with_mean=False))
        ]), numeric_features_zero),
        ("num_other", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False))
        ]), numeric_features_other + graph_features ),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features),
        ("text", text_transformer, [text_feature])
    ],
    remainder="drop",
    sparse_threshold=1.0
)


# COMMAND ----------

# MAGIC %md ###Logistic Regresion

# COMMAND ----------

# MAGIC %md Logistic Regression Base

# COMMAND ----------

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import pandas as pd

# CV
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# PIPELINE BASE (solo tabular)
logreg_base_pipe = Pipeline(steps=[
    ("preprocess", preprocess_linear_base),  # 👈 importante: sin texto ni grafo
    ("clf", LogisticRegression(
        max_iter=10000,
        solver="liblinear",
        class_weight="balanced",
        random_state=42
    ))
])

# GRID
param_grid_logreg = [
    {
        "clf__penalty": ["l1", "l2"],
        "clf__C": [0.001, 0.01, 0.1, 1.0, 10.0]
    }
]

grid_logreg_base = GridSearchCV(
    estimator=logreg_base_pipe,
    param_grid=param_grid_logreg,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

# TRAIN (solo 70%)
grid_logreg_base.fit(X_base_train, y_train)

best_logreg_base = grid_logreg_base.best_estimator_

print("Best Logistic Regression BASE params:", grid_logreg_base.best_params_)
print("Best CV f1_macro:", grid_logreg_base.best_score_)

# TEST 
proba_base = best_logreg_base.predict_proba(X_base_test)[:, 1]
pred_base = (proba_base >= 0.5).astype(int)

print("\nLogistic Regression BASE | Test independiente | threshold=0.5")
print(classification_report(y_test, pred_base))
print("Confusion:\n", confusion_matrix(y_test, pred_base))
print("AUC:", roc_auc_score(y_test, proba_base))

# RESULTADOS
logreg_base_test_results = {
    "model": "Logistic Regression",
    "strategy": "BASE (TABULAR ONLY)",
    "best_cv_f1_macro": grid_logreg_base.best_score_,
    "accuracy": accuracy_score(y_test, pred_base),
    "precision_macro": precision_score(y_test, pred_base, average="macro", zero_division=0),
    "recall_macro": recall_score(y_test, pred_base, average="macro", zero_division=0),
    "f1_macro": f1_score(y_test, pred_base, average="macro", zero_division=0),
    "roc_auc": roc_auc_score(y_test, proba_base),
    "best_params": grid_logreg_base.best_params_
}

pd.DataFrame([logreg_base_test_results])

# COMMAND ----------

# MAGIC %md
# MAGIC Tabular + Text

# COMMAND ----------

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import pandas as pd

# CV
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# PIPELINE (TABULAR + TEXT)
logreg_text_pipe = Pipeline(steps=[
    ("preprocess", preprocess_linear_text),
    ("clf", LogisticRegression(
        max_iter=10000,
        solver="liblinear",
        class_weight="balanced",
        random_state=42
    ))
])

# GRID
param_grid_logreg = {
    "clf__penalty": ["l1", "l2"],
    "clf__C": [0.001, 0.01, 0.1, 1.0, 10.0]
}

grid_logreg_text = GridSearchCV(
    estimator=logreg_text_pipe,
    param_grid=param_grid_logreg,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

# TRAIN 
grid_logreg_text.fit(X_text_train, y_train)

best_logreg_text = grid_logreg_text.best_estimator_

print("Best Logistic Regression TABULAR + TEXT params:", grid_logreg_text.best_params_)
print("Best CV f1_macro:", grid_logreg_text.best_score_)

# TEST 
proba_text = best_logreg_text.predict_proba(X_text_test)[:, 1]
pred_text = (proba_text >= 0.5).astype(int)

print("\nLogistic Regression TABULAR + TEXT | Test independiente | threshold=0.5")
print(classification_report(y_test, pred_text))
print("Confusion:\n", confusion_matrix(y_test, pred_text))
print("AUC:", roc_auc_score(y_test, proba_text))

# RESULTADOS
logreg_text_test_results = {
    "model": "Logistic Regression",
    "strategy": "TABULAR + TEXT",
    "best_cv_f1_macro": grid_logreg_text.best_score_,
    "accuracy": accuracy_score(y_test, pred_text),
    "precision_macro": precision_score(y_test, pred_text, average="macro", zero_division=0),
    "recall_macro": recall_score(y_test, pred_text, average="macro", zero_division=0),
    "f1_macro": f1_score(y_test, pred_text, average="macro", zero_division=0),
    "roc_auc": roc_auc_score(y_test, proba_text),
    "best_params": grid_logreg_text.best_params_
}

pd.DataFrame([logreg_text_test_results])

# COMMAND ----------

# MAGIC %md
# MAGIC Graph

# COMMAND ----------

# Logistic Regression - TABULAR + GRAPH

logreg_graph_pipe = Pipeline(steps=[
    ("preprocess", preprocess_linear_graph),
    ("clf", LogisticRegression(
        max_iter=10000,
        solver="liblinear",
        class_weight="balanced",
        random_state=42
    ))
])

param_grid_logreg = {
    "clf__penalty": ["l1", "l2"],
    "clf__C": [0.001, 0.01, 0.1, 1.0, 10.0]
}

grid_logreg_graph = GridSearchCV(
    estimator=logreg_graph_pipe,
    param_grid=param_grid_logreg,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

# TRAIN
grid_logreg_graph.fit(X_graph_train, y_train)

best_logreg_graph = grid_logreg_graph.best_estimator_

print("Best Logistic Regression TABULAR + GRAPH params:", grid_logreg_graph.best_params_)
print("Best CV f1_macro:", grid_logreg_graph.best_score_)

# TEST
proba_graph = best_logreg_graph.predict_proba(X_graph_test)[:, 1]
pred_graph = (proba_graph >= 0.5).astype(int)

print("\nLogistic Regression TABULAR + GRAPH | Test independiente | threshold=0.5")
print(classification_report(y_test, pred_graph))
print("Confusion:\n", confusion_matrix(y_test, pred_graph))
print("AUC:", roc_auc_score(y_test, proba_graph))

# RESULTADOS
logreg_graph_test_results = {
    "model": "Logistic Regression",
    "strategy": "TABULAR + GRAPH",
    "best_cv_f1_macro": grid_logreg_graph.best_score_,
    "accuracy": accuracy_score(y_test, pred_graph),
    "precision_macro": precision_score(y_test, pred_graph, average="macro", zero_division=0),
    "recall_macro": recall_score(y_test, pred_graph, average="macro", zero_division=0),
    "f1_macro": f1_score(y_test, pred_graph, average="macro", zero_division=0),
    "roc_auc": roc_auc_score(y_test, proba_graph),
    "best_params": grid_logreg_graph.best_params_
}

# COMMAND ----------

# MAGIC %md Logistic Regression FULL

# COMMAND ----------

# Logistic Regression - TABULAR + TEXT + GRAPH


logreg_graph_text_pipe = Pipeline(steps=[
    ("preprocess", preprocess_linear_graph_text),
    ("clf", LogisticRegression(
        max_iter=10000,
        solver="liblinear",
        class_weight="balanced",
        random_state=42
    ))
])

param_grid_logreg = {
    "clf__penalty": ["l1", "l2"],
    "clf__C": [0.001, 0.01, 0.1, 1.0, 10.0]
}

grid_logreg_graph_text = GridSearchCV(
    estimator=logreg_graph_text_pipe,
    param_grid=param_grid_logreg,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

# CV solo sobre train
grid_logreg_graph_text.fit(X_graph_text_train, y_train)

best_logreg_graph_text = grid_logreg_graph_text.best_estimator_

print("Best Logistic Regression TABULAR + TEXT + GRAPH params:", grid_logreg_graph_text.best_params_)
print("Best CV f1_macro:", grid_logreg_graph_text.best_score_)

# Test independiente
proba_graph_text = best_logreg_graph_text.predict_proba(X_graph_text_test)[:, 1]
pred_graph_text = (proba_graph_text >= 0.5).astype(int)

print("\nLogistic Regression TABULAR + TEXT + GRAPH | Test independiente | threshold=0.5")
print(classification_report(y_test, pred_graph_text))
print("Confusion:\n", confusion_matrix(y_test, pred_graph_text))
print("AUC:", roc_auc_score(y_test, proba_graph_text))

logreg_graph_text_test_results = {
    "model": "Logistic Regression",
    "strategy": "TABULAR + TEXT + GRAPH",
    "best_cv_f1_macro": grid_logreg_graph_text.best_score_,
    "accuracy": accuracy_score(y_test, pred_graph_text),
    "precision_macro": precision_score(y_test, pred_graph_text, average="macro", zero_division=0),
    "recall_macro": recall_score(y_test, pred_graph_text, average="macro", zero_division=0),
    "f1_macro": f1_score(y_test, pred_graph_text, average="macro", zero_division=0),
    "roc_auc": roc_auc_score(y_test, proba_graph_text),
    "best_params": grid_logreg_graph_text.best_params_
}

pd.DataFrame([logreg_graph_text_test_results])

# COMMAND ----------

df_lr_comparison = pd.DataFrame([
    logreg_base_test_results,
    logreg_text_test_results,
    logreg_graph_test_results,
    logreg_graph_text_test_results
])

df_lr_comparison.sort_values(by="f1_macro", ascending=False)

# COMMAND ----------

# MAGIC %md SVM Base

# COMMAND ----------

from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import pandas as pd

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Linear SVM - BASE (TABULAR ONLY)

svm_base_pipe = Pipeline(steps=[
    ("preprocess", preprocess_linear_base),
    ("clf", LinearSVC(
        class_weight="balanced",
        random_state=42,
        max_iter=50000
    ))
])

param_grid_svm = {
    "clf__C": [0.01, 0.1, 1.0, 5.0]
}

grid_svm_base = GridSearchCV(
    estimator=svm_base_pipe,
    param_grid=param_grid_svm,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

# CV solo sobre train
grid_svm_base.fit(X_base_train, y_train)

best_svm_base = grid_svm_base.best_estimator_

print("Best SVM BASE params:", grid_svm_base.best_params_)
print("Best CV f1_macro:", grid_svm_base.best_score_)

# Test independiente
scores_base = best_svm_base.decision_function(X_base_test)
pred_base = (scores_base >= 0).astype(int)

print("\nSVM BASE | Test independiente | threshold=0")
print(classification_report(y_test, pred_base))
print("Confusion:\n", confusion_matrix(y_test, pred_base))
print("AUC:", roc_auc_score(y_test, scores_base))

svm_base_test_results = {
    "model": "Linear SVM",
    "strategy": "BASE (TABULAR ONLY)",
    "best_cv_f1_macro": grid_svm_base.best_score_,
    "accuracy": accuracy_score(y_test, pred_base),
    "precision_macro": precision_score(y_test, pred_base, average="macro", zero_division=0),
    "recall_macro": recall_score(y_test, pred_base, average="macro", zero_division=0),
    "f1_macro": f1_score(y_test, pred_base, average="macro", zero_division=0),
    "roc_auc": roc_auc_score(y_test, scores_base),
    "best_params": grid_svm_base.best_params_
}

pd.DataFrame([svm_base_test_results])

# COMMAND ----------

# MAGIC %md
# MAGIC Text

# COMMAND ----------

from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import pandas as pd

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

svm_text_pipe = Pipeline(steps=[
    ("preprocess", preprocess_linear_text),
    ("clf", LinearSVC(
        class_weight="balanced",
        random_state=42,
        max_iter=50000
    ))
])

param_grid_svm = {
    "clf__C": [0.01, 0.1, 1.0, 5.0]
}

grid_svm_text = GridSearchCV(
    estimator=svm_text_pipe,
    param_grid=param_grid_svm,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

grid_svm_text.fit(X_text_train, y_train)

best_svm_text = grid_svm_text.best_estimator_

print("Best SVM TABULAR + TEXT params:", grid_svm_text.best_params_)
print("Best CV f1_macro:", grid_svm_text.best_score_)

scores_text = best_svm_text.decision_function(X_text_test)
pred_text = (scores_text >= 0).astype(int)

print("\nSVM TABULAR + TEXT | Test independiente | threshold=0")
print(classification_report(y_test, pred_text))
print("Confusion:\n", confusion_matrix(y_test, pred_text))
print("AUC:", roc_auc_score(y_test, scores_text))

svm_text_test_results = {
    "model": "Linear SVM",
    "strategy": "TABULAR + TEXT",
    "best_cv_f1_macro": grid_svm_text.best_score_,
    "accuracy": accuracy_score(y_test, pred_text),
    "precision_macro": precision_score(y_test, pred_text, average="macro", zero_division=0),
    "recall_macro": recall_score(y_test, pred_text, average="macro", zero_division=0),
    "f1_macro": f1_score(y_test, pred_text, average="macro", zero_division=0),
    "roc_auc": roc_auc_score(y_test, scores_text),
    "best_params": grid_svm_text.best_params_
}

pd.DataFrame([svm_text_test_results])

# COMMAND ----------

# MAGIC %md
# MAGIC Graph

# COMMAND ----------

from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import pandas as pd

# Linear SVM - TABULAR + GRAPH

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

svm_graph_pipe = Pipeline(steps=[
    ("preprocess", preprocess_linear_graph),
    ("clf", LinearSVC(
        class_weight="balanced",
        random_state=42,
        max_iter=50000
    ))
])

param_grid_svm = {
    "clf__C": [0.01, 0.1, 1.0, 5.0]
}

grid_svm_graph = GridSearchCV(
    estimator=svm_graph_pipe,
    param_grid=param_grid_svm,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

# CV solo sobre train
grid_svm_graph.fit(X_graph_train, y_train)

best_svm_graph = grid_svm_graph.best_estimator_

print("Best SVM TABULAR + GRAPH params:", grid_svm_graph.best_params_)
print("Best SVM TABULAR + GRAPH CV f1_macro:", grid_svm_graph.best_score_)

# Evaluación en test independiente
scores_svm_graph = best_svm_graph.decision_function(X_graph_test)
pred_svm_graph = (scores_svm_graph >= 0).astype(int)

print("\nSVM TABULAR + GRAPH | Test independiente | threshold=0")
print(classification_report(y_test, pred_svm_graph))
print("Confusion:\n", confusion_matrix(y_test, pred_svm_graph))
print("AUC:", roc_auc_score(y_test, scores_svm_graph))

svm_graph_test_results = {
    "model": "Linear SVM",
    "strategy": "TABULAR + GRAPH",
    "best_cv_f1_macro": grid_svm_graph.best_score_,
    "accuracy": accuracy_score(y_test, pred_svm_graph),
    "precision_macro": precision_score(y_test, pred_svm_graph, average="macro", zero_division=0),
    "recall_macro": recall_score(y_test, pred_svm_graph, average="macro", zero_division=0),
    "f1_macro": f1_score(y_test, pred_svm_graph, average="macro", zero_division=0),
    "roc_auc": roc_auc_score(y_test, scores_svm_graph),
    "best_params": grid_svm_graph.best_params_
}

pd.DataFrame([svm_graph_test_results])

# COMMAND ----------

# MAGIC %md
# MAGIC Full

# COMMAND ----------

# Linear SVM - TABULAR + TEXT + GRAPH
# CV sobre train / evaluación final sobre test independiente

svm_graph_text_pipe = Pipeline(steps=[
    ("preprocess", preprocess_linear_graph_text),
    ("clf", LinearSVC(
        class_weight="balanced",
        random_state=42,
        max_iter=50000
    ))
])

grid_svm_graph_text = GridSearchCV(
    estimator=svm_graph_text_pipe,
    param_grid=param_grid_svm,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

# CV solo sobre train
grid_svm_graph_text.fit(X_graph_text_train, y_train)

best_svm_graph_text = grid_svm_graph_text.best_estimator_

print("Best SVM TABULAR + TEXT + GRAPH params:", grid_svm_graph_text.best_params_)
print("Best CV f1_macro:", grid_svm_graph_text.best_score_)

# Evaluación final sobre test independiente
scores_graph_text = best_svm_graph_text.decision_function(X_graph_text_test)
pred_graph_text = (scores_graph_text >= 0).astype(int)

print("\nSVM TABULAR + TEXT + GRAPH | Test independiente | threshold=0")
print(classification_report(y_test, pred_graph_text))
print("Confusion:\n", confusion_matrix(y_test, pred_graph_text))
print("AUC:", roc_auc_score(y_test, scores_graph_text))

svm_graph_text_test_results = {
    "model": "Linear SVM",
    "strategy": "TABULAR + TEXT + GRAPH",
    "best_cv_f1_macro": grid_svm_graph_text.best_score_,
    "accuracy": accuracy_score(y_test, pred_graph_text),
    "precision_macro": precision_score(y_test, pred_graph_text, average="macro", zero_division=0),
    "recall_macro": recall_score(y_test, pred_graph_text, average="macro", zero_division=0),
    "f1_macro": f1_score(y_test, pred_graph_text, average="macro", zero_division=0),
    "roc_auc": roc_auc_score(y_test, scores_graph_text),
    "best_params": grid_svm_graph_text.best_params_
}

pd.DataFrame([svm_graph_text_test_results])

# COMMAND ----------

df_svm_comparison = pd.DataFrame([
    svm_base_test_results,
    svm_text_test_results,
    svm_graph_test_results,
    svm_graph_text_test_results
])

df_svm_comparison.sort_values(by="f1_macro", ascending=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Rnadom Forest

# COMMAND ----------

# MAGIC %md
# MAGIC Base

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import pandas as pd


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


rf_base_pipe = Pipeline(steps=[
    ("preprocess", preprocess_tree_base),  
    ("clf", RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    ))
])
param_grid_rf = {
    "clf__n_estimators": [300,500],
    "clf__max_depth": [10, 20,None],
    "clf__min_samples_leaf": [1, 3,5],
    "clf__max_features": ["sqrt", 0.3],
}

grid_rf_base = GridSearchCV(
    estimator=rf_base_pipe,
    param_grid=param_grid_rf,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=1
)


grid_rf_base.fit(X_base_train, y_train)

best_rf_base = grid_rf_base.best_estimator_

print("Best RF BASE params:", grid_rf_base.best_params_)
print("Best CV f1_macro:", grid_rf_base.best_score_)


proba_base = best_rf_base.predict_proba(X_base_test)[:, 1]
pred_base = (proba_base >= 0.5).astype(int)

print("\nRandom Forest BASE | Test independiente | threshold=0.5")
print(classification_report(y_test, pred_base))
print("Confusion:\n", confusion_matrix(y_test, pred_base))
print("AUC:", roc_auc_score(y_test, proba_base))


rf_base_test_results = {
    "model": "Random Forest",
    "strategy": "BASE (TABULAR ONLY)",
    "best_cv_f1_macro": grid_rf_base.best_score_,
    "accuracy": accuracy_score(y_test, pred_base),
    "precision_macro": precision_score(y_test, pred_base, average="macro", zero_division=0),
    "recall_macro": recall_score(y_test, pred_base, average="macro", zero_division=0),
    "f1_macro": f1_score(y_test, pred_base, average="macro", zero_division=0),
    "roc_auc": roc_auc_score(y_test, proba_base),
    "best_params": grid_rf_base.best_params_
}

pd.DataFrame([rf_base_test_results])

# COMMAND ----------

# MAGIC %md
# MAGIC Text

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import pandas as pd

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf_text_pipe = Pipeline(steps=[
    ("preprocess", preprocess_tree_text),
    ("clf", RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    ))
])

param_grid_rf_text = {
    "clf__n_estimators": [200, 300, 400],
    "clf__max_depth": [None],
    "clf__min_samples_leaf": [2, 3, 4],
    "clf__max_features": ["sqrt"],
}

grid_rf_text = GridSearchCV(
    estimator=rf_text_pipe,
    param_grid=param_grid_rf_text,
    scoring="f1_macro",
    cv=cv5,
    n_jobs=1,
    verbose=1
)

grid_rf_text.fit(X_text_train, y_train)

best_rf_text = grid_rf_text.best_estimator_

print("Best RF TABULAR + TEXT params:", grid_rf_text.best_params_)
print("Best RF TABULAR + TEXT CV f1_macro:", grid_rf_text.best_score_)

proba_text = best_rf_text.predict_proba(X_text_test)[:, 1]
pred_text = (proba_text >= 0.5).astype(int)

print("\nRandom Forest TABULAR + TEXT | Test independiente | threshold=0.5")
print(classification_report(y_test, pred_text))
print("Confusion:\n", confusion_matrix(y_test, pred_text))
print("AUC:", roc_auc_score(y_test, proba_text))

rf_text_test_results = {
    "model": "Random Forest",
    "strategy": "TABULAR + TEXT",
    "best_cv_f1_macro": grid_rf_text.best_score_,
    "accuracy": accuracy_score(y_test, pred_text),
    "precision_macro": precision_score(y_test, pred_text, average="macro", zero_division=0),
    "recall_macro": recall_score(y_test, pred_text, average="macro", zero_division=0),
    "f1_macro": f1_score(y_test, pred_text, average="macro", zero_division=0),
    "roc_auc": roc_auc_score(y_test, proba_text),
    "best_params": grid_rf_text.best_params_
}

pd.DataFrame([rf_text_test_results])

# COMMAND ----------

# MAGIC %md
# MAGIC Graph 

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import pandas as pd

# Random Forest - TABULAR + GRAPH

cv5 = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

rf_graph_pipe = Pipeline(steps=[
    ("preprocess", preprocess_tree_graph),
    ("clf", RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    ))
])

param_grid_rf_graph = {
    "clf__n_estimators": [300, 500],
    "clf__max_depth": [8, 10, 15, None],
    "clf__min_samples_leaf": [1, 3, 5],
    "clf__max_features": ["sqrt", 0.5],
}

grid_rf_graph = GridSearchCV(
    estimator=rf_graph_pipe,
    param_grid=param_grid_rf_graph,
    scoring="f1_macro",
    cv=cv5,
    n_jobs=1,
    verbose=1
)

# CV solo sobre train
grid_rf_graph.fit(X_graph_train, y_train)

best_rf_graph = grid_rf_graph.best_estimator_

print("Best RF TABULAR + GRAPH params:", grid_rf_graph.best_params_)
print("Best RF TABULAR + GRAPH CV f1_macro:", grid_rf_graph.best_score_)

# Evaluación final en test independiente
proba_graph = best_rf_graph.predict_proba(X_graph_test)[:, 1]
pred_graph = (proba_graph >= 0.5).astype(int)

print("\nRandom Forest TABULAR + GRAPH | Test independiente | threshold=0.5")
print(classification_report(y_test, pred_graph))
print("Confusion:\n", confusion_matrix(y_test, pred_graph))
print("AUC:", roc_auc_score(y_test, proba_graph))

rf_graph_test_results = {
    "model": "Random Forest",
    "strategy": "TABULAR + GRAPH",
    "best_cv_f1_macro": grid_rf_graph.best_score_,
    "accuracy": accuracy_score(y_test, pred_graph),
    "precision_macro": precision_score(y_test, pred_graph, average="macro", zero_division=0),
    "recall_macro": recall_score(y_test, pred_graph, average="macro", zero_division=0),
    "f1_macro": f1_score(y_test, pred_graph, average="macro", zero_division=0),
    "roc_auc": roc_auc_score(y_test, proba_graph),
    "best_params": grid_rf_graph.best_params_
}

pd.DataFrame([rf_graph_test_results])

# COMMAND ----------

# MAGIC %md
# MAGIC Full

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import pandas as pd

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf_graph_text_pipe = Pipeline(steps=[
    ("preprocess", preprocess_tree_graph_text),
    ("clf", RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample"
    ))
])

param_grid_rf_graph_text = {
    "clf__n_estimators": [300, 500],
    "clf__max_depth": [8, 10, 15],
    "clf__min_samples_leaf": [3, 5, 10],
    "clf__max_features": ["sqrt", 0.5],
}

grid_rf_graph_text = GridSearchCV(
    estimator=rf_graph_text_pipe,
    param_grid=param_grid_rf_graph_text,
    scoring="f1_macro",
    cv=cv5,
    n_jobs=1,
    verbose=1
)

# CV solo sobre train
grid_rf_graph_text.fit(X_graph_text_train, y_train)

best_rf_graph_text = grid_rf_graph_text.best_estimator_

print("Best RF TABULAR + TEXT + GRAPH params:", grid_rf_graph_text.best_params_)
print("Best RF TABULAR + TEXT + GRAPH CV f1_macro:", grid_rf_graph_text.best_score_)

# Test independiente
proba_graph_text = best_rf_graph_text.predict_proba(X_graph_text_test)[:, 1]
pred_graph_text = (proba_graph_text >= 0.5).astype(int)

print("\nRandom Forest TABULAR + TEXT + GRAPH | Test independiente | threshold=0.5")
print(classification_report(y_test, pred_graph_text))
print("Confusion:\n", confusion_matrix(y_test, pred_graph_text))
print("AUC:", roc_auc_score(y_test, proba_graph_text))

rf_graph_text_test_results = {
    "model": "Random Forest",
    "strategy": "TABULAR + TEXT + GRAPH",
    "best_cv_f1_macro": grid_rf_graph_text.best_score_,
    "accuracy": accuracy_score(y_test, pred_graph_text),
    "precision_macro": precision_score(y_test, pred_graph_text, average="macro", zero_division=0),
    "recall_macro": recall_score(y_test, pred_graph_text, average="macro", zero_division=0),
    "f1_macro": f1_score(y_test, pred_graph_text, average="macro", zero_division=0),
    "roc_auc": roc_auc_score(y_test, proba_graph_text),
    "best_params": grid_rf_graph_text.best_params_
}

pd.DataFrame([rf_graph_text_test_results])

# COMMAND ----------

df_rf_comparison = pd.DataFrame([
    rf_base_test_results,
    rf_text_test_results,
    rf_graph_test_results,
    rf_graph_text_test_results
])

df_rf_comparison.sort_values(by="f1_macro", ascending=False)

# COMMAND ----------

# MAGIC %md
# MAGIC XGBOOST

# COMMAND ----------

# MAGIC %pip install xgboost

# COMMAND ----------

# MAGIC %md
# MAGIC Base

# COMMAND ----------

from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from scipy.stats import randint, uniform
import pandas as pd

# scale_pos_weight (solo train)
pos = float((y_train == 1).sum())
neg = float((y_train == 0).sum())
spw_base = neg / max(pos, 1.0)

# PIPELINE BASE
xgb_base_pipe = Pipeline(steps=[
    ("preprocess", preprocess_tree_base),  
    ("clf", XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    ))
])

# RANDOM SEARCH
param_dist_xgb_base = {
    "clf__n_estimators": randint(200, 600),
    "clf__max_depth": randint(3, 7),
    "clf__learning_rate": uniform(0.02, 0.10),
    "clf__subsample": uniform(0.65, 0.35),
    "clf__colsample_bytree": uniform(0.65, 0.35),
    "clf__min_child_weight": randint(1, 10),
    "clf__gamma": uniform(0.0, 2.0),
    "clf__reg_lambda": uniform(0.5, 15.0),
    "clf__reg_alpha": uniform(0.0, 2.0),
    "clf__scale_pos_weight": [1.0, spw_base],
}

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rand_xgb_base = RandomizedSearchCV(
    estimator=xgb_base_pipe,
    param_distributions=param_dist_xgb_base,
    n_iter=40,
    scoring="f1_macro",
    cv=cv5,
    verbose=1,
    n_jobs=1,
    random_state=42
)

# TRAIN
rand_xgb_base.fit(X_base_train, y_train)

best_xgb_base = rand_xgb_base.best_estimator_

print("Best XGB BASE params:", rand_xgb_base.best_params_)
print("Best XGB BASE CV f1_macro:", rand_xgb_base.best_score_)

# TEST
proba_base = best_xgb_base.predict_proba(X_base_test)[:, 1]
pred_base = (proba_base >= 0.5).astype(int)

print("\nXGBoost BASE | Test independiente | threshold=0.5")
print(classification_report(y_test, pred_base))
print("Confusion:\n", confusion_matrix(y_test, pred_base))
print("AUC:", roc_auc_score(y_test, proba_base))

xgb_base_test_results = {
    "model": "XGBoost",
    "strategy": "BASE (TABULAR ONLY)",
    "best_cv_f1_macro": rand_xgb_base.best_score_,
    "accuracy": accuracy_score(y_test, pred_base),
    "precision_macro": precision_score(y_test, pred_base, average="macro", zero_division=0),
    "recall_macro": recall_score(y_test, pred_base, average="macro", zero_division=0),
    "f1_macro": f1_score(y_test, pred_base, average="macro", zero_division=0),
    "roc_auc": roc_auc_score(y_test, proba_base),
    "best_params": rand_xgb_base.best_params_
}

pd.DataFrame([xgb_base_test_results])

# COMMAND ----------

# MAGIC %md
# MAGIC Text

# COMMAND ----------

from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from scipy.stats import randint, uniform
import pandas as pd

# scale_pos_weight 
pos = float((y_train == 1).sum())
neg = float((y_train == 0).sum())
spw_text = neg / max(pos, 1.0)

# PIPELINE
xgb_text_pipe = Pipeline(steps=[
    ("preprocess", preprocess_tree_text),
    ("clf", XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    ))
])


# RANDOM SEARCH
param_dist_xgb_text = {
    "clf__n_estimators": randint(200, 600),
    "clf__max_depth": randint(3, 7),
    "clf__learning_rate": uniform(0.02, 0.10),
    "clf__subsample": uniform(0.65, 0.35),
    "clf__colsample_bytree": uniform(0.65, 0.35),
    "clf__min_child_weight": randint(1, 10),
    "clf__gamma": uniform(0.0, 2.0),
    "clf__reg_lambda": uniform(0.5, 15.0),
    "clf__reg_alpha": uniform(0.0, 2.0),
    "clf__scale_pos_weight": [1.0, spw_text],
}

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rand_xgb_text = RandomizedSearchCV(
    estimator=xgb_text_pipe,
    param_distributions=param_dist_xgb_text,
    n_iter=40,
    scoring="f1_macro",
    cv=cv5,
    verbose=1,
    n_jobs=1,
    random_state=42
)

# TRAIN
rand_xgb_text.fit(X_text_train, y_train)

best_xgb_text = rand_xgb_text.best_estimator_

print("Best XGB TABULAR + TEXT params:", rand_xgb_text.best_params_)
print("Best CV f1_macro:", rand_xgb_text.best_score_)

# TEST
proba_text = best_xgb_text.predict_proba(X_text_test)[:, 1]
pred_text = (proba_text >= 0.5).astype(int)

print("\nXGBoost TABULAR + TEXT | Test independiente | threshold=0.5")
print(classification_report(y_test, pred_text))
print("Confusion:\n", confusion_matrix(y_test, pred_text))
print("AUC:", roc_auc_score(y_test, proba_text))

# RESULTADOS
xgb_text_test_results = {
    "model": "XGBoost",
    "strategy": "TABULAR + TEXT",
    "best_cv_f1_macro": rand_xgb_text.best_score_,
    "accuracy": accuracy_score(y_test, pred_text),
    "precision_macro": precision_score(y_test, pred_text, average="macro", zero_division=0),
    "recall_macro": recall_score(y_test, pred_text, average="macro", zero_division=0),
    "f1_macro": f1_score(y_test, pred_text, average="macro", zero_division=0),
    "roc_auc": roc_auc_score(y_test, proba_text),
    "best_params": rand_xgb_text.best_params_
}

pd.DataFrame([xgb_text_test_results])

# COMMAND ----------

# MAGIC %md
# MAGIC Graph

# COMMAND ----------

from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from scipy.stats import randint, uniform
import pandas as pd

# scale_pos_weight usando solo train
pos = float((y_train == 1).sum())
neg = float((y_train == 0).sum())
spw_graph = neg / max(pos, 1.0)

# XGBoost - TABULAR + GRAPH

xgb_graph_pipe = Pipeline(steps=[
    ("preprocess", preprocess_tree_graph),
    ("clf", XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    ))
])

param_dist_xgb_graph = {
    "clf__n_estimators": randint(200, 600),
    "clf__max_depth": randint(3, 7),
    "clf__learning_rate": uniform(0.02, 0.10),
    "clf__subsample": uniform(0.65, 0.35),
    "clf__colsample_bytree": uniform(0.65, 0.35),
    "clf__min_child_weight": randint(1, 10),
    "clf__gamma": uniform(0.0, 2.0),
    "clf__reg_lambda": uniform(0.5, 15.0),
    "clf__reg_alpha": uniform(0.0, 2.0),
    "clf__scale_pos_weight": [1.0, spw_graph],
}

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rand_xgb_graph = RandomizedSearchCV(
    estimator=xgb_graph_pipe,
    param_distributions=param_dist_xgb_graph,
    n_iter=30,
    scoring="f1_macro",
    cv=cv5,
    verbose=1,
    n_jobs=1,
    random_state=42
)

# CV solo sobre train
rand_xgb_graph.fit(X_graph_train, y_train)

best_xgb_graph = rand_xgb_graph.best_estimator_

print("Best XGB TABULAR + GRAPH params:", rand_xgb_graph.best_params_)
print("Best XGB TABULAR + GRAPH CV f1_macro:", rand_xgb_graph.best_score_)

# Evaluación final sobre test independiente
proba_xgb_graph = best_xgb_graph.predict_proba(X_graph_test)[:, 1]
pred_xgb_graph = (proba_xgb_graph >= 0.5).astype(int)

print("\nXGBoost TABULAR + GRAPH | Test independiente | threshold=0.5")
print(classification_report(y_test, pred_xgb_graph))
print("Confusion:\n", confusion_matrix(y_test, pred_xgb_graph))
print("AUC:", roc_auc_score(y_test, proba_xgb_graph))

xgb_graph_test_results = {
    "model": "XGBoost",
    "strategy": "TABULAR + GRAPH",
    "best_cv_f1_macro": rand_xgb_graph.best_score_,
    "accuracy": accuracy_score(y_test, pred_xgb_graph),
    "precision_macro": precision_score(y_test, pred_xgb_graph, average="macro", zero_division=0),
    "recall_macro": recall_score(y_test, pred_xgb_graph, average="macro", zero_division=0),
    "f1_macro": f1_score(y_test, pred_xgb_graph, average="macro", zero_division=0),
    "roc_auc": roc_auc_score(y_test, proba_xgb_graph),
    "best_params": rand_xgb_graph.best_params_
}

pd.DataFrame([xgb_graph_test_results])

# COMMAND ----------

# MAGIC %md FULL

# COMMAND ----------

from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from scipy.stats import randint, uniform
import pandas as pd

# scale_pos_weight SOLO 
pos = float((y_train == 1).sum())
neg = float((y_train == 0).sum())
spw_full = neg / max(pos, 1.0)


# PIPELINE FULL
xgb_full_pipe = Pipeline(steps=[
    ("preprocess", preprocess_tree_graph_text),  
    ("clf", XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    ))
])

# RANDOM SEARCH
param_dist_xgb_full = {
    "clf__n_estimators": randint(450, 800),
    "clf__max_depth": randint(5, 9),
    "clf__learning_rate": uniform(0.01, 0.06),
    "clf__subsample": uniform(0.75, 0.25),
    "clf__colsample_bytree": uniform(0.75, 0.25),
    "clf__min_child_weight": randint(1, 6),
    "clf__gamma": uniform(0.5, 1.8),
    "clf__reg_lambda": uniform(5.0, 15.0),
    "clf__reg_alpha": uniform(0.0, 1.5),
    "clf__scale_pos_weight": [1.0, spw_full],
    "clf__max_delta_step": [0, 1, 5]
}

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rand_xgb_full = RandomizedSearchCV(
    estimator=xgb_full_pipe,
    param_distributions=param_dist_xgb_full,
    n_iter=50,
    scoring="f1_macro",
    cv=cv5,
    verbose=1,
    n_jobs=1,
    random_state=42
)

# TRAIN
rand_xgb_full.fit(X_graph_text_train, y_train)

best_xgb_full = rand_xgb_full.best_estimator_

print("Best XGB TABULAR + TEXT + GRAPH params:", rand_xgb_full.best_params_)
print("Best CV f1_macro:", rand_xgb_full.best_score_)

# TEST
proba_xgb_full = best_xgb_full.predict_proba(X_graph_text_test)[:, 1]
pred_xgb_full = (proba_xgb_full >= 0.5).astype(int)

print("\nXGBoost TABULAR + TEXT + GRAPH | Test independiente | threshold=0.5")
print(classification_report(y_test, pred_xgb_full))
print("Confusion:\n", confusion_matrix(y_test, pred_xgb_full))
print("AUC:", roc_auc_score(y_test, proba_xgb_full))

xgb_full_test_results = {
    "model": "XGBoost",
    "strategy": "TABULAR + TEXT + GRAPH",
    "best_cv_f1_macro": rand_xgb_full.best_score_,
    "accuracy": accuracy_score(y_test, pred_xgb_full),
    "precision_macro": precision_score(y_test, pred_xgb_full, average="macro", zero_division=0),
    "recall_macro": recall_score(y_test, pred_xgb_full, average="macro", zero_division=0),
    "f1_macro": f1_score(y_test, pred_xgb_full, average="macro", zero_division=0),
    "roc_auc": roc_auc_score(y_test, proba_xgb_full),
    "best_params": rand_xgb_full.best_params_
}

pd.DataFrame([xgb_full_test_results])

# COMMAND ----------

import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    balanced_accuracy_score, accuracy_score
)

def eval_model(name, y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)

    return {
        "model": name,
        "auc_roc": roc_auc_score(y_true, y_score),
        "auc_pr": average_precision_score(y_true, y_score),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "recall_0": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_1": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "precision_0": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "precision_1": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "threshold": threshold
    }

results = []
curves = []
y_true = y_test.values

# LOGISTIC REGRESSION

proba_lr_base = best_logreg_base.predict_proba(X_base_test)[:, 1]
results.append(eval_model("Regresión Logística Base", y_true, proba_lr_base, 0.5))
curves.append(("Regresión Logística Base", y_true, proba_lr_base))

proba_lr_text = best_logreg_text.predict_proba(X_text_test)[:, 1]
results.append(eval_model("Regresión Logística Texto", y_true, proba_lr_text, 0.5))
curves.append(("Regresión Logística Texto", y_true, proba_lr_text))

proba_lr_graph = best_logreg_graph.predict_proba(X_graph_test)[:, 1]
results.append(eval_model("Regresión Logística Grafo", y_true, proba_lr_graph, 0.5))
curves.append(("Regresión Logística Grafo", y_true, proba_lr_graph))

proba_lr_full = best_logreg_graph_text.predict_proba(X_graph_text_test)[:, 1]
results.append(eval_model("Regresión Logística Completo", y_true, proba_lr_full, 0.5))
curves.append(("Regresión Logística Completo", y_true, proba_lr_full))


# SVM

scores_svm_base = best_svm_base.decision_function(X_base_test)
results.append(eval_model("SVM Base", y_true, scores_svm_base, 0.0))
curves.append(("SVM Base", y_true, scores_svm_base))

scores_svm_text = best_svm_text.decision_function(X_text_test)
results.append(eval_model("SVM Texto", y_true, scores_svm_text, 0.0))
curves.append(("SVM Texto", y_true, scores_svm_text))

scores_svm_graph = best_svm_graph.decision_function(X_graph_test)
results.append(eval_model("SVM Grafo", y_true, scores_svm_graph, 0.0))
curves.append(("SVM Grafo", y_true, scores_svm_graph))

scores_svm_full = best_svm_graph_text.decision_function(X_graph_text_test)
results.append(eval_model("SVM Completo", y_true, scores_svm_full, 0.0))
curves.append(("SVM Completo", y_true, scores_svm_full))


# RANDOM FOREST

proba_rf_base = best_rf_base.predict_proba(X_base_test)[:, 1]
results.append(eval_model("Random Forest Base", y_true, proba_rf_base, 0.5))
curves.append(("Random Forest Base", y_true, proba_rf_base))

proba_rf_text = best_rf_text.predict_proba(X_text_test)[:, 1]
results.append(eval_model("Random Forest Texto", y_true, proba_rf_text, 0.5))
curves.append(("Random Forest Texto", y_true, proba_rf_text))

proba_rf_graph = best_rf_graph.predict_proba(X_graph_test)[:, 1]
results.append(eval_model("Random Forest Grafo", y_true, proba_rf_graph, 0.5))
curves.append(("Random Forest Grafo", y_true, proba_rf_graph))

proba_rf_full = best_rf_graph_text.predict_proba(X_graph_text_test)[:, 1]
results.append(eval_model("Random Forest Completo", y_true, proba_rf_full, 0.5))
curves.append(("Random Forest Completo", y_true, proba_rf_full))


# XGBOOST

proba_xgb_base = best_xgb_base.predict_proba(X_base_test)[:, 1]
results.append(eval_model("XGBoost Base", y_true, proba_xgb_base, 0.5))
curves.append(("XGBoost Base", y_true, proba_xgb_base))

proba_xgb_text = best_xgb_text.predict_proba(X_text_test)[:, 1]
results.append(eval_model("XGBoost Texto", y_true, proba_xgb_text, 0.5))
curves.append(("XGBoost Texto", y_true, proba_xgb_text))

proba_xgb_graph = best_xgb_graph.predict_proba(X_graph_test)[:, 1]
results.append(eval_model("XGBoost Grafo", y_true, proba_xgb_graph, 0.5))
curves.append(("XGBoost Grafo", y_true, proba_xgb_graph))

proba_xgb_full = best_xgb_full.predict_proba(X_graph_text_test)[:, 1]
results.append(eval_model("XGBoost Completo", y_true, proba_xgb_full, 0.5))
curves.append(("XGBoost Completo", y_true, proba_xgb_full))


# RESULTADOS FINALES

df_results = (
    pd.DataFrame(results)
    .sort_values(by="f1_macro", ascending=False)
    .reset_index(drop=True)
)

df_results

# COMMAND ----------

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))

for name, y_true, y_score in curves:
    
    # ROC 
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

# Línea base (modelo random)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")

plt.xlabel("Tasa de Falsos Positivos")
plt.ylabel("Tasa de Verdaderos Positivos")
plt.title("Curvas ROC de los Modelos Evaluados")

plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, f1_score

# Mejor modelo FULL
# XGBoost TABULAR + TEXT + GRAPH
pipe = best_xgb_full

# Dataset FULL
X_test = X_graph_text_test

# Métrica principal
scorer = make_scorer(f1_score, average="macro")

# Permutation Importance
perm_xgb_full = permutation_importance(
    estimator=pipe,
    X=X_test,
    y=y_test,
    scoring=scorer,
    n_repeats=10,
    random_state=42,
    n_jobs=1
)

# DataFrame resultados
pi_xgb_full_df = (
    pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": perm_xgb_full.importances_mean,
        "importance_std": perm_xgb_full.importances_std
    })
    .sort_values("importance_mean", ascending=False)
)

display(pi_xgb_full_df.head(20))

# Plot Top Variables
top_n = 10

plt.figure(figsize=(8, 6))

plt.barh(
    pi_xgb_full_df.head(top_n).iloc[::-1]["feature"],
    pi_xgb_full_df.head(top_n).iloc[::-1]["importance_mean"]
)

plt.xlabel("Disminución del F1-score macro al permutar la variable")
plt.title("Importancia de Variables – Modelo XGBoost Completo")

plt.tight_layout()
plt.show()
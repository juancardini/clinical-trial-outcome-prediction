# Databricks notebook source
# MAGIC %md
# MAGIC Store CSVs Files in a volume to be used in NEO4J Desktop to create the Graph.

# COMMAND ----------

# Cargar tablas GOLD
df_studies = spark.table("gold.nodes_studies").toPandas()
df_interventions = spark.table("gold.nodes_interventions").toPandas()
df_rels_interv = spark.table("gold.rels_study_intervention").toPandas()
df_sponsors = spark.table("gold.nodes_sponsors").toPandas()
df_rels_spon = spark.table("gold.rels_study_sponsor").toPandas()
df_facilities = spark.table("gold.nodes_facilities").toPandas()
df_rels_fac = spark.table("gold.rels_study_facility").toPandas()
df_countries = spark.table("gold.nodes_countries").toPandas()
df_rels_ctry = spark.table("gold.rels_study_country").toPandas()
df_conditions = spark.table("gold.nodes_conditions").toPandas()
df_rels_cond = spark.table("gold.rels_study_condition").toPandas()
df_designs = spark.table("gold.nodes_designs").toPandas()
df_rels_des = spark.table("gold.rels_study_design").toPandas()




# COMMAND ----------

import pandas as pd

def save_csv_pandas(df, name):
    path = f"/Volumes/workspace/default/tesis_volume/{name}.csv"
    df.to_csv(path, index=False)
    print(f" Archivo guardado: {path}")


# COMMAND ----------

# NODOS
save_csv_pandas(df_studies,       "nodes_studies")
save_csv_pandas(df_interventions, "nodes_interventions")
save_csv_pandas(df_sponsors,      "nodes_sponsors")
save_csv_pandas(df_facilities,    "nodes_facilities")
save_csv_pandas(df_countries,     "nodes_countries")
save_csv_pandas(df_conditions,    "nodes_conditions")
save_csv_pandas(df_designs,    "nodes_designs")


#  RELACIONES 
save_csv_pandas(df_rels_interv,   "rels_study_intervention")
save_csv_pandas(df_rels_spon,     "rels_study_sponsor")
save_csv_pandas(df_rels_fac,      "rels_study_facility")
save_csv_pandas(df_rels_ctry,     "rels_study_country")
save_csv_pandas(df_rels_cond,     "rels_study_condition")
save_csv_pandas(df_rels_des,     "rels_study_design")

# Databricks notebook source
# MAGIC %md ###Objective: Read a CSV with Studies Graph Metrics

# COMMAND ----------

import pandas as pd

path = '/Volumes/workspace/default/tesis_volume/grafometricas.csv'

df_spark = (spark.read

    .format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load(path)
)
df=df_spark.withColumnRenamed("s.community_louvain","community_louvain")
df_grafo = df.createOrReplaceTempView("df_grafo")
print(df.columns)
display(df)


# COMMAND ----------

# MAGIC %md Studies Data

# COMMAND ----------

#  Cargar tablas GOLD en Spark

df_studies_spark        = spark.table("gold.nodes_studies")
df_interventions_spark  = spark.table("gold.nodes_interventions")
df_rels_interv_spark    = spark.table("gold.rels_study_intervention")
df_sponsors_spark       = spark.table("gold.nodes_sponsors")
df_rels_spon_spark      = spark.table("gold.rels_study_sponsor")
df_facilities_spark     = spark.table("gold.nodes_facilities")
df_rels_fac_spark       = spark.table("gold.rels_study_facility")
df_countries_spark      = spark.table("gold.nodes_countries")
df_rels_ctry_spark      = spark.table("gold.rels_study_country")
df_conditions_spark     = spark.table("gold.nodes_conditions")
df_rels_cond_spark      = spark.table("gold.rels_study_condition")


# Crear vistas temporales para usar en SQL

df_studies_spark.createOrReplaceTempView("vw_nodes_studies")
df_interventions_spark.createOrReplaceTempView("vw_nodes_interventions")
df_rels_interv_spark.createOrReplaceTempView("vw_rels_study_intervention")
df_sponsors_spark.createOrReplaceTempView("vw_nodes_sponsors")
df_rels_spon_spark.createOrReplaceTempView("vw_rels_study_sponsor")
df_facilities_spark.createOrReplaceTempView("vw_nodes_facilities")
df_rels_fac_spark.createOrReplaceTempView("vw_rels_study_facility")
df_countries_spark.createOrReplaceTempView("vw_nodes_countries")
df_rels_ctry_spark.createOrReplaceTempView("vw_rels_study_country")
df_conditions_spark.createOrReplaceTempView("vw_nodes_conditions")
df_rels_cond_spark.createOrReplaceTempView("vw_rels_study_condition")


# COMMAND ----------

# MAGIC %sql
# MAGIC create or replace temp view n_conditions as 
# MAGIC select count(distinct condition_name) as n_conditions,study_id 
# MAGIC from gold.rels_study_condition 
# MAGIC group by study_id;
# MAGIC select * from n_conditions

# COMMAND ----------

# MAGIC %sql
# MAGIC create or replace temp view n_countries as 
# MAGIC select study_id,count(distinct country_name) as n_countries 
# MAGIC from gold.rels_study_country
# MAGIC group by study_id;
# MAGIC select * from n_countries

# COMMAND ----------

# MAGIC %sql
# MAGIC create or replace temp view n_interventions as 
# MAGIC select study_id,count (distinct intervention_name) as n_interventions from gold.rels_study_intervention
# MAGIC group by study_id;
# MAGIC select * from n_interventions

# COMMAND ----------

# MAGIC %sql
# MAGIC create or replace temp view n_sponsors as SELECT 
# MAGIC   r.study_id,
# MAGIC   COUNT(DISTINCT CASE 
# MAGIC     WHEN UPPER(s.lead_or_collaborator) = 'LEAD' 
# MAGIC       THEN s.sponsor_name 
# MAGIC   END) AS n_sponsor_lead,
# MAGIC   COUNT(DISTINCT CASE 
# MAGIC     WHEN UPPER(s.lead_or_collaborator) = 'COLLABORATOR' 
# MAGIC       THEN s.sponsor_name 
# MAGIC   END) AS n_sponsor_collaborator
# MAGIC FROM gold.rels_study_sponsor r
# MAGIC JOIN gold.nodes_sponsors s 
# MAGIC   ON UPPER(TRIM(s.sponsor_name)) = UPPER(TRIM(r.sponsor_name))
# MAGIC GROUP BY r.study_id;
# MAGIC select * from n_sponsors
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC create or replace temp view n_facilities as 
# MAGIC   SELECT
# MAGIC     r.study_id,
# MAGIC     count(distinct f.facility_id) as n_facilities
# MAGIC     
# MAGIC   FROM gold.rels_study_facility r
# MAGIC   JOIN gold.nodes_facilities f
# MAGIC     ON r.facility_id = f.facility_id
# MAGIC     group by r.study_id
# MAGIC     ;
# MAGIC     select * from n_facilities
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC
# MAGIC CREATE OR REPLACE TEMP VIEW design_features AS
# MAGIC SELECT
# MAGIC   r.study_id,
# MAGIC
# MAGIC   d.allocation,
# MAGIC   d.primary_purpose,
# MAGIC   d.intervention_model,
# MAGIC   d.masking,
# MAGIC   CASE 
# MAGIC     WHEN d.masking IS NULL OR UPPER(d.masking) = 'NONE' THEN 0 
# MAGIC     ELSE 1 
# MAGIC   END AS is_blinded,
# MAGIC
# MAGIC   -- Cantidad de roles enmascarados
# MAGIC   COALESCE(CAST(d.subject_masked      AS INT), 0) AS subject_masked_int,
# MAGIC   COALESCE(CAST(d.caregiver_masked    AS INT), 0) AS caregiver_masked_int,
# MAGIC   COALESCE(CAST(d.investigator_masked AS INT), 0) AS investigator_masked_int,
# MAGIC
# MAGIC   ( COALESCE(CAST(d.subject_masked      AS INT), 0)
# MAGIC   + COALESCE(CAST(d.caregiver_masked    AS INT), 0)
# MAGIC   + COALESCE(CAST(d.investigator_masked AS INT), 0)
# MAGIC   ) AS n_roles_masked
# MAGIC
# MAGIC FROM gold.rels_study_design r
# MAGIC JOIN gold.nodes_designs d
# MAGIC   ON r.design_id = d.design_id;
# MAGIC select * from design_features

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMP VIEW dataset_ml AS
# MAGIC SELECT
# MAGIC     g.study_id,
# MAGIC s.brief_title,
# MAGIC     -- VARIABLES CLÍNICAS BÁSICAS
# MAGIC     
# MAGIC     s.approved,
# MAGIC
# MAGIC     -- MÉTRICAS DE GRAFO
# MAGIC     g.degree_total,
# MAGIC     g.degree_norm,
# MAGIC     g.pagerank_norm,
# MAGIC     g.betweenness_norm,
# MAGIC     g.closeness_norm,
# MAGIC     g.community_louvain,
# MAGIC
# MAGIC     -- FEATURES DERIVADAS DE NODOS RELACIONADOS
# MAGIC     nc.n_conditions,
# MAGIC     nctr.n_countries,
# MAGIC     ni.n_interventions,
# MAGIC     nf.n_facilities,
# MAGIC     --EMBEDDINGS
# MAGIC     g.emb_0,
# MAGIC     g.emb_1,
# MAGIC     g.emb_2,
# MAGIC     g.emb_3,
# MAGIC     g.emb_4,
# MAGIC     g.emb_5,
# MAGIC     g.emb_6,
# MAGIC     g.emb_7,
# MAGIC     g.emb_8,
# MAGIC     g.emb_9,
# MAGIC     g.emb_10,
# MAGIC     g.emb_11,
# MAGIC     g.emb_12,
# MAGIC     g.emb_13,
# MAGIC     g.emb_14,
# MAGIC     g.emb_15,
# MAGIC     g.emb_16,
# MAGIC     g.emb_17,
# MAGIC     g.emb_18,
# MAGIC     g.emb_19,
# MAGIC     g.emb_20,
# MAGIC     g.emb_21,
# MAGIC     g.emb_22,
# MAGIC     g.emb_23,
# MAGIC     g.emb_24,
# MAGIC     g.emb_25,
# MAGIC     g.emb_26,
# MAGIC     g.emb_27,
# MAGIC     g.emb_28,
# MAGIC     g.emb_29,
# MAGIC     g.emb_30,
# MAGIC     g.emb_31,
# MAGIC     g.emb_32,
# MAGIC     g.emb_33,
# MAGIC     g.emb_34,
# MAGIC     g.emb_35,
# MAGIC     g.emb_36,
# MAGIC     g.emb_37,
# MAGIC     g.emb_38,
# MAGIC     g.emb_39,
# MAGIC     g.emb_40,
# MAGIC     g.emb_41,
# MAGIC     g.emb_42,
# MAGIC     g.emb_43,
# MAGIC     g.emb_44,
# MAGIC     g.emb_45,
# MAGIC     g.emb_46,
# MAGIC     g.emb_47,
# MAGIC     g.emb_48,
# MAGIC     g.emb_49,
# MAGIC     g.emb_50,
# MAGIC     g.emb_51,
# MAGIC     g.emb_52,
# MAGIC     g.emb_53,
# MAGIC     g.emb_54,
# MAGIC     g.emb_55,
# MAGIC     g.emb_56,
# MAGIC     g.emb_57,
# MAGIC     g.emb_58,
# MAGIC     g.emb_59,
# MAGIC     g.emb_60,
# MAGIC     g.emb_61,
# MAGIC     g.emb_62,
# MAGIC     g.emb_63,
# MAGIC
# MAGIC     -- SPONSORS
# MAGIC     ns.n_sponsor_lead,
# MAGIC     ns.n_sponsor_collaborator,
# MAGIC     COALESCE(ns.n_sponsor_lead, 0) 
# MAGIC       + COALESCE(ns.n_sponsor_collaborator, 0) AS n_sponsors_total,
# MAGIC
# MAGIC     -- DISEÑO DEL ESTUDIO
# MAGIC     d.allocation,
# MAGIC     d.primary_purpose,
# MAGIC     d.intervention_model,
# MAGIC     d.masking,
# MAGIC     d.is_blinded
# MAGIC FROM df_grafo g
# MAGIC LEFT JOIN gold.nodes_studies   s   ON s.study_id   = g.study_id
# MAGIC LEFT JOIN n_conditions         nc  ON nc.study_id  = g.study_id
# MAGIC LEFT JOIN n_countries          nctr ON nctr.study_id = g.study_id
# MAGIC LEFT JOIN n_interventions      ni  ON ni.study_id  = g.study_id
# MAGIC LEFT JOIN n_sponsors           ns  ON ns.study_id  = g.study_id
# MAGIC LEFT JOIN n_facilities         nf  ON nf.study_id  = g.study_id
# MAGIC LEFT JOIN design_features      d   ON d.study_id   = g.study_id;
# MAGIC select * from dataset_ml
# MAGIC

# COMMAND ----------

# MAGIC %sql 
# MAGIC CREATE OR REPLACE TABLE gold.dataset_ml AS
# MAGIC SELECT * FROM dataset_ml;
# MAGIC select * from gold.dataset_ml
# MAGIC
# MAGIC
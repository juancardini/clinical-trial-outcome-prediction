# Databricks notebook source
# MAGIC %md #Gold

# COMMAND ----------

# MAGIC %md ###Studies Nodes

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE gold.nodes_studies AS
# MAGIC SELECT DISTINCT
# MAGIC     study_id,
# MAGIC     brief_title,
# MAGIC     start_date,
# MAGIC     completion_date,
# MAGIC     study_type,
# MAGIC     clinical_phase as phase,
# MAGIC     overall_status,
# MAGIC     approved
# MAGIC FROM silver.studies_diabetes;
# MAGIC select   * from gold.nodes_studies 
# MAGIC

# COMMAND ----------

# MAGIC %md ###Conditions

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE gold.nodes_conditions AS
# MAGIC SELECT DISTINCT
# MAGIC     condition_id,
# MAGIC     condition_name_raw as condition_name,
# MAGIC     condition_category
# MAGIC FROM silver.conditions c
# MAGIC JOIN silver.studies_diabetes s
# MAGIC   ON c.study_id = s.study_id;  
# MAGIC   select distinct condition_name from gold.nodes_conditions
# MAGIC

# COMMAND ----------

# MAGIC %md ###Relationships Studies-Conditions

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE gold.rels_study_condition AS
# MAGIC SELECT DISTINCT
# MAGIC     s.study_id,
# MAGIC     c.condition_name_raw as condition_name,
# MAGIC     'HAS_CONDITION' AS rel_type
# MAGIC FROM silver.conditions c
# MAGIC JOIN silver.studies_diabetes s
# MAGIC   ON c.study_id = s.study_id;
# MAGIC select * from gold.rels_study_condition

# COMMAND ----------

# MAGIC %md ###Countries Nodes

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE gold.nodes_countries AS
# MAGIC
# MAGIC     SELECT DISTINCT country_id,
# MAGIC         country_name
# MAGIC     FROM silver.countries c
# MAGIC     JOIN silver.studies_diabetes s
# MAGIC       ON c.study_id = s.study_id;
# MAGIC select * from gold.nodes_countries

# COMMAND ----------

# MAGIC %md ###Relationships Countries-Studies

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE gold.rels_study_country AS
# MAGIC SELECT DISTINCT
# MAGIC     s.study_id,
# MAGIC     c.country_name,
# MAGIC     'DEVELOPED_IN_COUNTRY' AS rel_type
# MAGIC FROM silver.countries c
# MAGIC JOIN silver.studies_diabetes s
# MAGIC     ON c.study_id = s.study_id
# MAGIC ;
# MAGIC select * from gold.rels_study_country
# MAGIC

# COMMAND ----------

# MAGIC %md ###Nodes Interventions

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE gold.nodes_interventions AS
# MAGIC SELECT DISTINCT
# MAGIC     i.intervention_id,
# MAGIC     i.intervention_name,
# MAGIC     i.intervention_type
# MAGIC FROM silver.interventions i
# MAGIC JOIN silver.studies_diabetes s
# MAGIC   ON i.study_id = s.study_id;
# MAGIC   
# MAGIC SELECT * FROM gold.nodes_interventions;
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ###Relationships Studies Interventions

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE gold.rels_study_intervention AS
# MAGIC SELECT DISTINCT
# MAGIC     s.study_id,
# MAGIC     i.intervention_name,
# MAGIC     'HAS_INTERVENTION' AS rel_type
# MAGIC FROM silver.interventions i
# MAGIC JOIN silver.studies_diabetes s
# MAGIC   ON i.study_id = s.study_id;
# MAGIC
# MAGIC SELECT * FROM gold.rels_study_intervention;
# MAGIC

# COMMAND ----------

# MAGIC %md ###Sponsors

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE gold.nodes_sponsors AS
# MAGIC SELECT DISTINCT
# MAGIC     sp.sponsor_id,
# MAGIC     sp.sponsor_name,
# MAGIC     sp.agency_class,
# MAGIC     sp.lead_or_collaborator
# MAGIC FROM silver.sponsors sp
# MAGIC JOIN silver.studies_diabetes s
# MAGIC   ON sp.study_id = s.study_id;
# MAGIC
# MAGIC SELECT * FROM gold.nodes_sponsors ;
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ###Relationships Studies Sponsors

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE gold.rels_study_sponsor AS
# MAGIC SELECT DISTINCT
# MAGIC     s.study_id,
# MAGIC     sp.sponsor_name,
# MAGIC     'HAS_SPONSOR' AS rel_type
# MAGIC FROM silver.sponsors sp
# MAGIC JOIN silver.studies_diabetes s
# MAGIC   ON sp.study_id = s.study_id;
# MAGIC
# MAGIC SELECT distinct sponsor_name FROM gold.rels_study_sponsor;
# MAGIC

# COMMAND ----------

# MAGIC %md ###Designs

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE gold.nodes_designs AS
# MAGIC SELECT DISTINCT
# MAGIC     d.design_id,
# MAGIC     d.allocation,
# MAGIC     d.primary_purpose,
# MAGIC     d.intervention_model,
# MAGIC     d.masking,
# MAGIC     d.subject_masked,
# MAGIC     d.caregiver_masked,
# MAGIC     d.investigator_masked
# MAGIC FROM silver.designs d
# MAGIC JOIN silver.studies_diabetes s
# MAGIC   ON d.study_id = s.study_id;
# MAGIC
# MAGIC SELECT * FROM gold.nodes_designs;
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ###Relationships Studies Designs

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE gold.rels_study_design AS
# MAGIC SELECT DISTINCT
# MAGIC     s.study_id,
# MAGIC     d.design_id,
# MAGIC     'HAS_DESIGN' AS rel_type
# MAGIC FROM silver.designs d
# MAGIC JOIN silver.studies_diabetes s
# MAGIC   ON d.study_id = s.study_id;
# MAGIC
# MAGIC SELECT * FROM gold.rels_study_design;
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ###Facilities

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE gold.nodes_facilities AS
# MAGIC SELECT DISTINCT
# MAGIC     f.facility_id,
# MAGIC     f.facility_name,
# MAGIC     f.city,
# MAGIC     f.state,
# MAGIC     f.country,
# MAGIC     f.latitude,
# MAGIC     f.longitude
# MAGIC FROM silver.facilities f
# MAGIC JOIN silver.studies_diabetes s
# MAGIC   ON f.study_id = s.study_id;
# MAGIC
# MAGIC SELECT distinct facility_name FROM gold.nodes_facilities where lower(country) like '%argentina%';
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ###Relationships Studies Fcilities

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE gold.rels_study_facility AS
# MAGIC SELECT DISTINCT
# MAGIC     s.study_id,
# MAGIC     f.facility_id,
# MAGIC     'HAS_FACILITY' AS rel_type
# MAGIC FROM silver.facilities f
# MAGIC JOIN silver.studies_diabetes s
# MAGIC   ON f.study_id = s.study_id;
# MAGIC
# MAGIC SELECT * FROM gold.rels_study_facility;
# MAGIC
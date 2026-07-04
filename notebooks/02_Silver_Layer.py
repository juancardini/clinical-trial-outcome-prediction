# Databricks notebook source
# MAGIC %md #Silver Layer

# COMMAND ----------

# MAGIC %md ###Studies

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE silver.studies_diabetes AS
# MAGIC WITH base AS (
# MAGIC     SELECT DISTINCT
# MAGIC         s.nct_id AS study_id,
# MAGIC         s.brief_title,
# MAGIC         COALESCE(
# MAGIC             CASE WHEN s.start_date_type = 'ACTUAL' THEN s.start_date else null END,
# MAGIC             s.start_date
# MAGIC         ) AS start_date,
# MAGIC         COALESCE(
# MAGIC             CASE WHEN s.completion_date_type = 'ACTUAL' THEN s.completion_date else null END,
# MAGIC             s.completion_date
# MAGIC         ) AS completion_date,
# MAGIC         s.study_type,
# MAGIC         CASE
# MAGIC     WHEN upper(phase) LIKE '%PHASE%1%/%PHASE%2%' THEN 'PHASE1'
# MAGIC     WHEN upper(phase) LIKE '%PHASE%2%/%PHASE%3%' THEN 'PHASE3'
# MAGIC     WHEN upper(phase) LIKE '%EARLY%' THEN 'PHASE1'
# MAGIC     WHEN upper(phase) LIKE '%PHASE 1%' THEN 'PHASE1'
# MAGIC     WHEN upper(phase) LIKE '%PHASE 2%' THEN 'PHASE2'
# MAGIC     WHEN upper(phase) LIKE '%PHASE 3%' THEN 'PHASE3'
# MAGIC     WHEN upper(phase) LIKE '%PHASE 4%' THEN 'PHASE4'
# MAGIC     WHEN upper(phase) =('NA') THEN 'NOT REPORTED'
# MAGIC     WHEN upper(phase) IS NULL THEN 'NOT REPORTED'
# MAGIC
# MAGIC     ELSE phase
# MAGIC END AS clinical_phase,
# MAGIC         CASE 
# MAGIC             WHEN s.overall_status LIKE '%UNKNOWN%' THEN NULL 
# MAGIC             ELSE s.overall_status 
# MAGIC         END AS overall_status
# MAGIC     FROM bronze.raw_studies s
# MAGIC     JOIN bronze.raw_conditions c 
# MAGIC         ON s.nct_id = c.nct_id   
# MAGIC     WHERE
# MAGIC         LOWER(c.name) LIKE '%diabetes%'
# MAGIC         AND s.start_date > '2000-01-01'
# MAGIC         AND s.study_type = 'INTERVENTIONAL'
# MAGIC         AND s.official_title IS NOT NULL
# MAGIC )
# MAGIC SELECT
# MAGIC     study_id,
# MAGIC     brief_title,
# MAGIC     start_date,
# MAGIC     completion_date,
# MAGIC     study_type,
# MAGIC     clinical_phase,
# MAGIC     overall_status,
# MAGIC     
# MAGIC     CASE
# MAGIC         WHEN trim(upper(overall_status)) = 'COMPLETED'
# MAGIC              AND regexp_extract(trim(upper(clinical_phase)), '([0-9])', 1) IN ('3','4')
# MAGIC         THEN 1
# MAGIC         WHEN trim(upper(overall_status)) IN ('TERMINATED', 'WITHDRAWN', 'SUSPENDED')
# MAGIC         THEN 0
# MAGIC         ELSE NULL
# MAGIC     END AS approved
# MAGIC FROM base;
# MAGIC select * from silver.studies_diabetes 
# MAGIC

# COMMAND ----------

# MAGIC %md ###Conditions

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE silver.conditions AS
# MAGIC SELECT
# MAGIC     c.nct_id AS study_id,
# MAGIC     c.id AS condition_id,
# MAGIC     CASE 
# MAGIC         WHEN LOWER(c.name) IN ('na','n/a','','none','not provided') THEN NULL
# MAGIC         ELSE c.name
# MAGIC     END AS condition_name_raw,
# MAGIC     CASE
# MAGIC         WHEN LOWER(c.name) LIKE '%type 1 diabetes%' 
# MAGIC              OR LOWER(c.name) LIKE '%type i diabetes%' 
# MAGIC              OR LOWER(c.name) LIKE '%juvenile%' 
# MAGIC             THEN 'Type 1 Diabetes'
# MAGIC
# MAGIC         WHEN LOWER(c.name) LIKE '%type 2 diabetes%' 
# MAGIC              OR LOWER(c.name) LIKE '%type ii diabetes%' 
# MAGIC              OR LOWER(c.name) LIKE '%t2dm%' 
# MAGIC             THEN 'Type 2 Diabetes'
# MAGIC
# MAGIC         WHEN LOWER(c.name) LIKE '%gestational%' 
# MAGIC              OR LOWER(c.name) LIKE '%pregnancy%' 
# MAGIC             THEN 'Gestational Diabetes'
# MAGIC
# MAGIC         WHEN LOWER(c.name) LIKE '%prediabetes%' 
# MAGIC              OR LOWER(c.name) LIKE '%impaired glucose%' 
# MAGIC             THEN 'Prediabetes'
# MAGIC
# MAGIC         WHEN LOWER(c.name) LIKE '%mody%' 
# MAGIC              OR LOWER(c.name) LIKE '%monogenic%' 
# MAGIC             THEN 'Monogenic Diabetes'
# MAGIC
# MAGIC         WHEN LOWER(c.name) LIKE '%diabetes%' 
# MAGIC              OR LOWER(c.name) LIKE '%hyperglycemia%' 
# MAGIC             THEN 'Diabetes Mellitus'
# MAGIC
# MAGIC         ELSE 'Non-Diabetes'
# MAGIC     END AS condition_category
# MAGIC FROM bronze.raw_conditions c;
# MAGIC select * from silver.conditions
# MAGIC

# COMMAND ----------

# MAGIC %md ###Countries 

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE silver.countries AS
# MAGIC SELECT
# MAGIC     con.id AS country_id,              
# MAGIC     con.nct_id AS study_id,             
# MAGIC     CASE
# MAGIC         WHEN con.name ILIKE '%czech%'     THEN 'Czech Republic'
# MAGIC         WHEN con.name ILIKE '%guinea%'    THEN 'Guinea'
# MAGIC         WHEN con.name ILIKE '%monaco%'    THEN 'France'
# MAGIC         WHEN con.name ILIKE '%réunion%'   THEN 'France'
# MAGIC         WHEN con.name ILIKE '%serbia%'   THEN 'Serbia and Montenegro'
# MAGIC
# MAGIC         ELSE con.name
# MAGIC     END AS country_name
# MAGIC     FROM bronze.raw_countries con
# MAGIC WHERE
# MAGIC     con.removed = 'false'
# MAGIC     AND con.name IS NOT NULL;
# MAGIC     select *
# MAGIC      from silver.countries
# MAGIC

# COMMAND ----------

# MAGIC %md ###Interventions

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE silver.interventions AS
# MAGIC SELECT
# MAGIC     i.id        AS intervention_id,
# MAGIC     i.nct_id    AS study_id,
# MAGIC     CASE 
# MAGIC         WHEN i.name IN ('NA', 'N/A', '', 'None', 'Not Provided') THEN NULL
# MAGIC         ELSE i.name
# MAGIC     END         AS intervention_name,
# MAGIC     i.intervention_type
# MAGIC FROM bronze.raw_interventions i
# MAGIC WHERE
# MAGIC     i.name IS NOT NULL
# MAGIC   AND NOT (LOWER(i.name) IN ('na','n/a','','none','not provided'));
# MAGIC     select * from silver.interventions
# MAGIC

# COMMAND ----------

# MAGIC %md ###Sponsors

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE silver.sponsors AS
# MAGIC SELECT
# MAGIC     sp.id                 AS sponsor_id,
# MAGIC     sp.nct_id             AS study_id,
# MAGIC     CASE 
# MAGIC         WHEN lower(sp.name) IN ('na', 'n/a', '', 'none', 'not provided') THEN NULL
# MAGIC         ELSE sp.name
# MAGIC     END                   AS sponsor_name,
# MAGIC     sp.agency_class,
# MAGIC     sp.lead_or_collaborator
# MAGIC FROM bronze.raw_sponsors sp
# MAGIC WHERE
# MAGIC     sp.name IS NOT NULL;
# MAGIC     select * from silver.sponsors
# MAGIC

# COMMAND ----------

# MAGIC %md ###Designs

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE silver.designs AS
# MAGIC SELECT DISTINCT
# MAGIC     d.id                  AS design_id,
# MAGIC     d.nct_id              AS study_id,
# MAGIC     case when d.allocation like '%NA%' then null else d.allocation end as allocation,
# MAGIC     d.primary_purpose,
# MAGIC     d.intervention_model,
# MAGIC     d.masking,
# MAGIC     d.subject_masked,
# MAGIC     d.caregiver_masked,
# MAGIC     d.investigator_masked
# MAGIC FROM bronze.raw_designs d;
# MAGIC select * from silver.designs
# MAGIC

# COMMAND ----------

# MAGIC %md ###Facilities

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE silver.facilities AS
# MAGIC SELECT DISTINCT
# MAGIC     f.id        AS facility_id,
# MAGIC     f.nct_id    AS study_id,
# MAGIC     f.name      AS facility_name,
# MAGIC     f.city,
# MAGIC     f.state,
# MAGIC     f.country,
# MAGIC     f.latitude,
# MAGIC     f.longitude
# MAGIC FROM bronze.raw_facilities f
# MAGIC WHERE
# MAGIC     f.name IS NOT NULL;
# MAGIC     select * from silver.facilities
# MAGIC
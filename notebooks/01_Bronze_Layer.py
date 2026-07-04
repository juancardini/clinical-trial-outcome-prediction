# Databricks notebook source
# MAGIC %md #Bronze

# COMMAND ----------

# MAGIC %md
# MAGIC Connection Details to AACT Database

# COMMAND ----------

url = "jdbc:postgresql://aact-db.ctti-clinicaltrials.org:5432/aact"
user= "juanmartincardini"
password= "1161992"
driver="org.postgresql.Driver"


# COMMAND ----------

# MAGIC %md ### Read Studies Data: ctgov.studies

# COMMAND ----------


raw_df_studies=(spark.read
.format("jdbc")
.option("url",url)
.option("driver",driver)
.option("dbtable","ctgov.studies")
.option("user",user)
.option("password",password)
.load()
)
(raw_df_studies.write
 .format("delta")
.mode("overwrite")
.saveAsTable("bronze.raw_studies")
)
display(raw_df_studies)

# COMMAND ----------

# MAGIC %md ###Read Conditions Data: ctgov.conditions

# COMMAND ----------

raw_df_conditions=(spark.read
.format("jdbc")
.option("url",url)
.option("driver",driver)
.option("dbtable","ctgov.conditions")
.option("user",user)
.option("password",password)
.load()
)
(raw_df_conditions.write
 .format("delta")
.mode("overwrite")
.saveAsTable("bronze.raw_conditions")
)
display(raw_df_conditions)

# COMMAND ----------

# MAGIC %md ### Read Countries Data: ctgov.studies

# COMMAND ----------

raw_df_countries=(spark.read
.format("jdbc")
.option("url",url)
.option("driver",driver)
.option("dbtable","ctgov.countries")
.option("user",user)
.option("password",password)
.load()
)
(raw_df_countries.write
 .format("delta")
.mode("overwrite")
.saveAsTable("bronze.raw_countries")
)
display(raw_df_countries)

# COMMAND ----------

# MAGIC %md ### Read Interventions Data: ctgov.interventions

# COMMAND ----------

raw_df_interventions=(spark.read
.format("jdbc")
.option("url",url)
.option("driver",driver)
.option("dbtable","ctgov.interventions")
.option("user",user)
.option("password",password)
.load()
)
(raw_df_interventions.write
 .format("delta")
.mode("overwrite")
.saveAsTable("bronze.raw_interventions")
)
display(raw_df_interventions)

# COMMAND ----------

# MAGIC %md ### Read Sponsors Data: ctgov.sponsors

# COMMAND ----------

raw_df_sponsors=(spark.read
.format("jdbc")
.option("url",url)
.option("driver",driver)
.option("dbtable","ctgov.sponsors")
.option("user",user)
.option("password",password)
.load()
)
(raw_df_sponsors.write
 .format("delta")
.mode("overwrite")
.saveAsTable("bronze.raw_sponsors")
)
display(raw_df_sponsors)

# COMMAND ----------

# MAGIC %md ###Read Designs Data: ctgov.designs

# COMMAND ----------

raw_df_designs=(spark.read
.format("jdbc")
.option("url",url)
.option("driver",driver)
.option("dbtable","ctgov.designs")
.option("user",user)
.option("password",password)
.load()
)
(raw_df_designs.write
 .format("delta")
.mode("overwrite")
.saveAsTable("bronze.raw_designs")
)
display(raw_df_designs)

# COMMAND ----------

# MAGIC %md ### Read Facilities Data: ctgov.facilities

# COMMAND ----------

raw_df_facilities=(spark.read
.format("jdbc")
.option("url",url)
.option("driver",driver)
.option("dbtable","ctgov.facilities")
.option("user",user)
.option("password",password)
.load()
)
(raw_df_facilities.write
 .format("delta")
.mode("overwrite")
.saveAsTable("bronze.raw_facilities")
)
display(raw_df_facilities)
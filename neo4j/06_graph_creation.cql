// Construcción del grafo de ensayos clínicos de diabetes

// Constraints

CREATE CONSTRAINT study_id_unique IF NOT EXISTS
FOR (s:Study)
REQUIRE s.study_id IS UNIQUE;

CREATE CONSTRAINT condition_name_unique IF NOT EXISTS
FOR (c:Condition)
REQUIRE c.condition_name IS UNIQUE;

CREATE CONSTRAINT country_name_unique IF NOT EXISTS
FOR (c:Country)
REQUIRE c.country_name IS UNIQUE;

CREATE CONSTRAINT intervention_name_unique IF NOT EXISTS
FOR (i:Intervention)
REQUIRE i.intervention_name IS UNIQUE;

CREATE CONSTRAINT sponsor_name_unique IF NOT EXISTS
FOR (sp:Sponsor)
REQUIRE sp.sponsor_name IS UNIQUE;

CREATE CONSTRAINT design_id_unique IF NOT EXISTS
FOR (d:Design)
REQUIRE d.design_id IS UNIQUE;

CREATE CONSTRAINT facility_id_unique IF NOT EXISTS
FOR (f:Facility)
REQUIRE f.facility_id IS UNIQUE;


// Study nodes

LOAD CSV WITH HEADERS FROM 'file:///nodes_studies.csv' AS row
MERGE (s:Study {study_id: row.study_id})
SET s.brief_title      = row.brief_title,
    s.start_date       = CASE WHEN row.start_date <> "" THEN date(row.start_date) ELSE null END,
    s.completion_date  = CASE WHEN row.completion_date <> "" THEN date(row.completion_date) ELSE null END,
    s.study_type       = row.study_type,
    s.phase            = row.phase,
    s.overall_status   = row.overall_status,
    s.approved         = CASE 
                           WHEN row.approved = "1" THEN 1
                           WHEN row.approved = "0" THEN 0
                           ELSE null 
                         END;


// Condition nodes

LOAD CSV WITH HEADERS FROM 'file:///nodes_conditions.csv' AS row
WITH 
    trim(row.condition_name) AS condition_name,
    trim(row.condition_id) AS condition_id,
    trim(row.condition_category) AS condition_category
WHERE condition_name IS NOT NULL AND condition_name <> ""
MERGE (c:Condition {condition_name: condition_name})
SET
    c.condition_id = condition_id,
    c.condition_category = condition_category;


// Study - Condition relationships

LOAD CSV WITH HEADERS FROM 'file:///rels_study_condition.csv' AS row
WITH 
  trim(row.study_id) AS study_id,
  trim(row.condition_name) AS condition_name
WHERE study_id <> "" 
  AND condition_name <> ""
MATCH (s:Study {study_id: study_id})
MATCH (c:Condition {condition_name: condition_name})
MERGE (s)-[:HAS_CONDITION]->(c);


// Country nodes

LOAD CSV WITH HEADERS FROM 'file:///nodes_countries.csv' AS row
WITH 
    trim(row.country_name) AS country_name,
    trim(row.country_id) AS country_id
WHERE country_name <> "" AND country_name IS NOT NULL
MERGE (c:Country {country_name: country_name})
SET c.country_id = country_id;


// Study - Country relationships

LOAD CSV WITH HEADERS FROM 'file:///rels_study_country.csv' AS row
WITH 
    trim(row.study_id) AS study_id,
    trim(row.country_name) AS country_name
WHERE study_id <> "" AND country_name <> ""
MATCH (s:Study {study_id: study_id})
MATCH (c:Country {country_name: country_name})
MERGE (s)-[:DEVELOPED_IN_COUNTRY]->(c);


// Intervention nodes

LOAD CSV WITH HEADERS FROM 'file:///nodes_interventions.csv' AS row
WITH 
    trim(row.intervention_name) AS intervention_name,
    trim(row.intervention_id) AS intervention_id,
    trim(row.intervention_type) AS intervention_type
WHERE intervention_name IS NOT NULL AND intervention_name <> ""
MERGE (i:Intervention {intervention_name: intervention_name})
SET 
    i.intervention_id = intervention_id,
    i.intervention_type = intervention_type;


// Study - Intervention relationships

LOAD CSV WITH HEADERS FROM 'file:///rels_study_intervention.csv' AS row
WITH 
    trim(row.study_id) AS study_id,
    trim(row.intervention_name) AS intervention_name
WHERE study_id <> "" AND intervention_name <> ""
MATCH (s:Study {study_id: study_id})
MATCH (i:Intervention {intervention_name: intervention_name})
MERGE (s)-[:HAS_INTERVENTION]->(i);


// Sponsor nodes

LOAD CSV WITH HEADERS FROM 'file:///nodes_sponsors.csv' AS row
WITH 
    trim(row.sponsor_name) AS sponsor_name,
    trim(row.sponsor_id) AS sponsor_id,
    trim(row.agency_class) AS agency_class,
    trim(row.lead_or_collaborator) AS lead_or_collaborator
WHERE sponsor_name IS NOT NULL AND sponsor_name <> ""
MERGE (sp:Sponsor {sponsor_name: sponsor_name})
SET
    sp.sponsor_id = sponsor_id,
    sp.agency_class = agency_class,
    sp.lead_or_collaborator = lead_or_collaborator;


// Study - Sponsor relationships

LOAD CSV WITH HEADERS FROM 'file:///rels_study_sponsor.csv' AS row
WITH 
  trim(row.study_id) AS study_id,
  trim(row.sponsor_name) AS sponsor_name
WHERE study_id <> "" AND sponsor_name <> ""
MATCH (s:Study {study_id: study_id})
MATCH (sp:Sponsor {sponsor_name: sponsor_name})
MERGE (s)-[:HAS_SPONSOR]->(sp);


// Design nodes

LOAD CSV WITH HEADERS FROM 'file:///nodes_designs.csv' AS row
WITH
  trim(row.design_id) AS design_id,
  trim(row.allocation) AS allocation,
  trim(row.primary_purpose) AS primary_purpose,
  trim(row.intervention_model) AS intervention_model,
  trim(row.masking) AS masking,
  trim(row.subject_masked) AS subject_masked,
  trim(row.caregiver_masked) AS caregiver_masked,
  trim(row.investigator_masked) AS investigator_masked
WHERE design_id IS NOT NULL AND design_id <> ""
MERGE (d:Design {design_id: design_id})
SET
  d.allocation          = allocation,
  d.primary_purpose     = primary_purpose,
  d.intervention_model  = intervention_model,
  d.masking             = masking,
  d.subject_masked      = subject_masked,
  d.caregiver_masked    = caregiver_masked,
  d.investigator_masked = investigator_masked;


// Study - Design relationships

LOAD CSV WITH HEADERS FROM 'file:///rels_study_design.csv' AS row
WITH 
  trim(row.study_id)  AS study_id,
  trim(row.design_id) AS design_id
WHERE study_id <> "" AND design_id <> ""
MATCH (s:Study  {study_id: study_id})
MATCH (d:Design {design_id: design_id})
MERGE (s)-[:HAS_DESIGN]->(d);


// Facility nodes

LOAD CSV WITH HEADERS FROM 'file:///nodes_facilities.csv' AS row
WITH
  trim(row.facility_id)   AS facility_id,
  trim(row.facility_name) AS facility_name,
  trim(row.city)          AS city,
  trim(row.state)         AS state,
  trim(row.country)       AS country,
  row.latitude            AS lat_raw,
  row.longitude           AS lon_raw
WHERE facility_id <> "" AND facility_id IS NOT NULL
WITH
  facility_id, facility_name, city, state, country,
  CASE WHEN lat_raw <> "" THEN toFloat(lat_raw) ELSE null END AS latitude,
  CASE WHEN lon_raw <> "" THEN toFloat(lon_raw) ELSE null END AS longitude
MERGE (f:Facility {facility_id: facility_id})
SET
  f.facility_name = facility_name,
  f.city          = city,
  f.state         = state,
  f.country       = country,
  f.latitude      = latitude,
  f.longitude     = longitude;


// Study - Facility relationships

LOAD CSV WITH HEADERS FROM 'file:///rels_study_facility.csv' AS row
WITH 
  trim(row.study_id)    AS study_id,
  trim(row.facility_id) AS facility_id
WHERE study_id <> "" AND facility_id <> ""
MATCH (s:Study {study_id: study_id})
MATCH (f:Facility {facility_id: facility_id})
MERGE (s)-[:HAS_FACILITY]->(f);


// Validation queries

MATCH (s:Study)
RETURN count(s) AS total_studies;

MATCH (c:Condition)
RETURN count(c) AS total_conditions;

MATCH (c:Country)
RETURN count(c) AS total_countries;

MATCH (i:Intervention)
RETURN count(i) AS total_interventions;

MATCH (sp:Sponsor)
RETURN count(sp) AS total_sponsors;

MATCH (d:Design)
RETURN count(d) AS total_designs;

MATCH (f:Facility)
RETURN count(f) AS total_facilities;

MATCH (s:Study)
RETURN
  s.study_id AS study_id,
  size([(s)-[:HAS_CONDITION]->(:Condition) | 1]) AS n_conditions,
  size([(s)-[:DEVELOPED_IN_COUNTRY]->(:Country) | 1]) AS n_countries,
  size([(s)-[:HAS_INTERVENTION]->(:Intervention) | 1]) AS n_interventions,
  size([(s)-[:HAS_SPONSOR]->(:Sponsor) | 1]) AS n_sponsors,
  size([(s)-[:HAS_DESIGN]->(:Design) | 1]) AS n_designs,
  size([(s)-[:HAS_FACILITY]->(:Facility) | 1]) AS n_facilities
ORDER BY study_id
LIMIT 50;

MATCH (s:Study)
WITH s, size([(s)-[:DEVELOPED_IN_COUNTRY]->(:Country) | 1]) AS n_countries
WHERE n_countries = 0
RETURN count(*) AS studies_without_country;

Prediction of Clinical Trial Outcomes Using Graph Metrics Integrated into Machine Learning Models
Este repositorio contiene el código desarrollado para la tesis de Maestría en Ciencia de Datos titulada: Prediction of Clinical Trial Outcomes Using Graph Metrics Integrated into Machine Learning Models.
Tecnologías:
•	Azure Databricks
•	PySpark
•	Spark SQL
•	Neo4j Desktop
•	Cypher
•	Neo4j Graph Data Science (GDS)
•	Python
Descripción
El objetivo de este proyecto es desarrollar un modelo de Machine Learning capaz de predecir el resultado de ensayos clínicos utilizando información proveniente de ClinicalTrials.gov (AACT), enriquecida mediante métricas estructurales obtenidas a partir de un grafo de conocimiento construido en Neo4j.
El proyecto implementa un pipeline completo que comprende:
•	Extracción y transformación de datos mediante Azure Databricks y PySpark
•	Construcción de un grafo de conocimiento en Neo4j.
•	Cálculo de métricas de grafos utilizando Neo4j Graph Data Science.
•	Generación de variables mediante Feature Engineering.
•	Entrenamiento y evaluación de modelos de Machine Learning.
Los notebooks de esta sección implementan el proceso ELT (Extract, Load, Transform) utilizado para preparar los datos de ClinicalTrials.gov (AACT) antes de la construcción del grafo y el entrenamiento de los modelos de Machine Learning.
Notebook 01: 01Bronze_Layer.ipynb

Este notebook implementa la etapa de extracción y carga de los datos de los ensayos clínicos.
•	Conexión a la base de datos pública AACT (ClinicalTrials.gov) mediante JDBC.
•	Lectura de las tablas del esquema ctgov utilizando PySpark.
•	Carga de los datos sin transformaciones en tablas Delta dentro del esquema bronze.
Tablas cargadas:
•	bronze.raw_studies
•	bronze.raw_conditions
•	bronze.raw_countries
•	bronze.raw_interventions
•	bronze.raw_sponsors
•	bronze.raw_designs
•	bronze.raw_facilities
Notebook 02: 02Silver_Layer.ipynb
Este notebook implementa la etapa de transformación de los datos.
A partir de las tablas de la capa Bronze, se realizan las transformaciones necesarias para obtener un conjunto de datos limpio y consistente.
Las principales tareas incluyen:
•	Filtrado de ensayos clínicos relacionados con diabetes.
•	Selección de estudios intervencionales.
•	Limpieza y normalización de datos.
•	Tratamiento de valores nulos.
•	Estandarización de países, condiciones e intervenciones.
•	Construcción de la variable objetivo approved.
Tablas generadas:
•	silver.studies_diabetes
•	silver.conditions
•	silver.countries
•	silver.interventions
•	silver.sponsors
•	silver.designs
•	silver.facilities
Notebook 03: 03Gold_Layer.ipynb
Este notebook prepara los datos para la construcción del grafo de conocimiento en Neo4j.
A partir de las tablas de la capa Silver, se generan las entidades y relaciones que conformarán el grafo.
Las principales tareas incluyen:
•	Construcción de tablas de nodos.
•	Construcción de tablas de relaciones.
•	Exportación de archivos CSV para su importación en Neo4j.
Nodos generados
•	Studies
•	Conditions
•	Countries
•	Interventions
•	Sponsors
•	Facilities
•	Designs
Relaciones generadas
•	HAS_CONDITION
•	HAS_INTERVENTION
•	HAS_SPONSOR
•	HAS_DESIGN
•	HAS_FACILITY
•	DEVELOPED_IN_COUNTRY
Notebook 04: 04_Exploratory_Data_Analysis
Este notebook realiza un análisis exploratorio de los datos obtenidos en la capa Gold antes de su carga en Neo4j.
Objetivo:
Validar la estructura y calidad de los datos generados en la capa Gold antes de la construcción del grafo de conocimiento en Neo4j.
Notebook 05: 05_Export_Data.ipynb
Este notebook exporta las tablas generadas en la capa Gold a archivos CSV que serán utilizados para construir el grafo en Neo4j.
•	Exportación de las tablas de nodos a formato CSV.
•	Exportación de las tablas de relaciones a formato CSV.
•	Almacenamiento de los archivos en un volumen de Azure Databricks.
Archivos generados:
Nodos
•	nodes_studies.csv
•	nodes_conditions.csv
•	nodes_countries.csv
•	nodes_interventions.csv
•	nodes_sponsors.csv
•	nodes_designs.csv
•	nodes_facilities.csv
Relaciones
•	rels_study_condition.csv
•	rels_study_country.csv
•	rels_study_intervention.csv
•	rels_study_sponsor.csv
•	rels_study_design.csv
•	rels_study_facility.csv
Los archivos CSV generados son copiados al directorio import de Neo4j Desktop y utilizados posteriormente por el script 06_graph_creation.cypher mediante la instrucción LOAD CSV para construir el grafo de conocimiento.
Neo4j - Construcción y Análisis del Grafo
Una vez generada la capa Gold, los archivos CSV son importados en Neo4j Desktop mediante consultas Cypher para construir el grafo.
Script: 06_graph_creation.cypher
Este script realiza la construcción completa del grafo a partir de los archivos generados en la capa Gold.
Funcionalidad:
•	Creación de restricciones (constraints) para garantizar la unicidad de los nodos.
•	Importación de nodos mediante LOAD CSV.
•	Creación de relaciones entre las distintas entidades.
•	Validación de la carga del grafo.
Nodos creados
•	Study
•	Condition
•	Country
•	Intervention
•	Sponsor
•	Design
•	Facility
Relaciones creadas
•	HAS_CONDITION
•	HAS_INTERVENTION
•	HAS_SPONSOR
•	HAS_DESIGN
•	HAS_FACILITY
•	DEVELOPED_IN_COUNTRY

Script: 07_graph_metrics.cypher
Este script utiliza la librería Neo4j Graph Data Science (GDS) para proyectar el grafo y calcular las métricas estructurales utilizadas posteriormente como variables de entrada para los modelos de Machine Learning.
•	Proyección del grafo mediante gds.graph.project.
•	Cálculo de métricas de centralidad.
•	Detección de comunidades mediante Louvain.
•	Normalización de las métricas calculadas.
•	Escritura de las propiedades en los nodos Study.
Métricas calculadas
•	Degree Centrality
•	Closeness Centrality
•	Betweenness Centrality
•	PageRank
•	Louvain Community Detection
Los nodos enriquecidos con estas métricas son utilizados posteriormente para generar el dataset final empleado en el entrenamiento y evaluación de los modelos de Machine Learning.
Script: 08_export_graph_metrics.cypher
Una vez calculadas las métricas de grafos, este script exporta la información de los nodos Study a un archivo CSV.
•	Consulta de los nodos Study enriquecidos con las métricas de grafos.
•	Exportación de las métricas estructurales calculadas mediante Cypher.
•	Generación de un archivo CSV que será utilizado posteriormente en Azure Databricks.
Variables exportadas
•	Degree Centrality
•	Closeness Centrality
•	Betweenness Centrality
•	PageRank
•	Louvain Community
Notebook 09: 09_Feature_Engineering.ipynb
Este notebook construye el dataset final utilizado para el entrenamiento de los modelos de Machine Learning.
A partir de las tablas de la capa Gold y de las métricas de grafos exportadas desde Neo4j, se integran todas las variables necesarias para generar un único conjunto de datos listo para el modelado.
•	Lectura del archivo CSV con las métricas de grafos exportadas desde Neo4j.
•	Lectura de las tablas de la capa Gold.
•	Integración de las métricas de grafos con la información clínica de los estudios.
•	Cálculo de variables agregadas:
•	Número de condiciones.
•	Número de países.
•	Número de intervenciones.
•	Número de sponsors principales y colaboradores.
•	Número de centros participantes.
•	Incorporación de variables del diseño del estudio.
•	Generación del dataset final para Machine Learning.
Se genera la tabla: gold.dataset_ml
Este dataset integra variables clínicas, variables derivadas del grafo y variables generadas mediante Feature Engineering, y constituye la entrada utilizada por el notebook 10_Machine_Learning_Models.ipynb para el entrenamiento y evaluación de los modelos predictivos.
Notebook 10: 10_Machine_Learning_Models.ipynb
Este notebook implementa el entrenamiento, optimización y evaluación de los modelos de Machine Learning utilizando el dataset generado durante la etapa de Feature Engineering.
•	Lectura del dataset gold.dataset_ml.
•	Preparación de las variables predictoras y de la variable objetivo.
•	División del conjunto de datos en entrenamiento y prueba.
•	Preprocesamiento de variables numéricas, categóricas y textuales mediante pipelines de Scikit-learn.
•	Entrenamiento y optimización de los modelos utilizando validación cruzada.
•	Comparación del desempeño utilizando diferentes conjuntos de variables
•	Evaluación del desempeño sobre el conjunto de prueba.
•	Comparación de resultados mediante diferentes métricas de clasificación.
•	Análisis de la importancia de las variables y visualización de curvas ROC.
Modelos implementados
•	Logistic Regression
•	Support Vector Machine (SVM)
•	Random Forest
•	XGBoost
Se comparan distintos modelos y configuraciones de variables para evaluar el impacto de las métricas de grafos en la predicción del resultado de los ensayos clínicos.





//Proyeccion grafo
CALL gds.graph.project(
  'fullGraph',
  ['Study','Condition','Country','Intervention','Sponsor','Facility','Design'],
  {
    HAS_CONDITION:        {type: 'HAS_CONDITION',        orientation: 'UNDIRECTED'},
    HAS_INTERVENTION:     {type: 'HAS_INTERVENTION',     orientation: 'UNDIRECTED'},
    DEVELOPED_IN_COUNTRY: {type: 'DEVELOPED_IN_COUNTRY', orientation: 'UNDIRECTED'},
    HAS_FACILITY:         {type: 'HAS_FACILITY',         orientation: 'UNDIRECTED'},
    HAS_SPONSOR:          {type: 'HAS_SPONSOR',          orientation: 'UNDIRECTED'},
    HAS_DESIGN:           {type: 'HAS_DESIGN',           orientation: 'UNDIRECTED'}
  }
);
//Degree
CALL gds.degree.stream('fullGraph')
YIELD nodeId, score
WITH gds.util.asNode(nodeId) AS n, score
WHERE n:Study
RETURN n.study_id AS study_id, score AS degree
ORDER BY degree DESC
LIMIT 20;

CALL gds.degree.write('fullGraph', {
  writeProperty: 'degree_total'
});






//Closeness centrality
CALL gds.closeness.harmonic.stream('fullGraph')
YIELD nodeId, score
WITH gds.util.asNode(nodeId) AS n, score
WHERE n:Study
RETURN n.study_id AS study_id, n.brief_title AS brief_title, score AS closeness
ORDER BY closeness DESC
LIMIT 20;
CALL gds.closeness.harmonic.write('fullGraph', {
  writeProperty: 'closeness'
});
//Betweenness
CALL gds.betweenness.write('fullGraph', {
  writeProperty: 'betweenness',
  samplingSize: 30,
  samplingSeed: 42
})
YIELD nodePropertiesWritten, computeMillis, writeMillis
RETURN nodePropertiesWritten, computeMillis, writeMillis;

MATCH (s:Study)
WITH max(s.betweenness) AS maxVal
MATCH (s:Study)
SET s.betweenness_norm = s.betweenness / maxVal;





//Louvain
CALL gds.louvain.write('fullGraph', {
  writeProperty: 'community_louvain'
})
YIELD communityCount, modularity, modularities, nodePropertiesWritten, computeMillis, writeMillis
RETURN communityCount, modularity, nodePropertiesWritten, computeMillis, writeMillis;

//Page Rank
CALL gds.pageRank.write('fullGraph', {
  writeProperty: 'pagerank',
  maxIterations: 20,        // podés dejar 20
  dampingFactor: 0.85       // standard PageRank
})
YIELD nodePropertiesWritten, didConverge, ranIterations, computeMillis, writeMillis
RETURN nodePropertiesWritten, didConverge, ranIterations, computeMillis, writeMillis;
//Normalizar
MATCH (s:Study)
WITH max(s.pagerank) AS maxVal
MATCH (s:Study)
SET s.pagerank_norm = s.pagerank / maxVal;



//Normalizar degree
MATCH (s:Study)
WITH max(s.degree_total) AS max_degree
MATCH (s:Study)
SET s.degree_norm = s.degree_total / max_degree;
//Normalizar clossness
MATCH (s:Study)
WITH max(s.closeness) AS max_close
MATCH (s:Study)
SET s.closeness_norm = s.closeness / max_close;

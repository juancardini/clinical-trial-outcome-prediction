//Graph Features
MATCH (s:Study)
RETURN
  s.study_id AS study_id,
  s.degree_total AS degree_total,
  s.degree_norm AS degree_norm,
  s.pagerank_norm AS pagerank_norm,
  s.betweenness_norm AS betweenness_norm,
  s.closeness_norm AS closeness_norm,
  s.community_louvain AS community_louvain
 
ORDER BY study_id;
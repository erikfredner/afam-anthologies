  SELECT DISTINCT
 	  a.id AS author_id,    
  	  a."name" AS author,
      a.birth_year,
      e.id AS edition_id,
      COALESCE(v.title, e.title, s."name") AS edition_title,
      e."year" AS ed_pub_year,
      v.id AS volume_id,
      v.volume_number,
      s.id AS series_id,
      s."name" AS series_name
  FROM data_author AS a
  JOIN data_work_authors AS wa
      ON wa.author_id = a.id
  JOIN data_workinanthology AS wia
      ON wia.work_id = wa.work_id
  JOIN data_volume AS v
      ON v.id = wia.volume_id
  JOIN data_edition AS e
      ON e.id = v.edition_id
  LEFT JOIN data_series AS s
      ON s.id = e.series_id
  JOIN data_edition_literary_traditions AS elt
      ON elt.edition_id = e.id
  JOIN data_literarytradition AS lt
      ON lt.id = elt.literarytradition_id
  WHERE lt."name" = 'African-American Literature'
  ORDER BY a."name", e."year", COALESCE(v.title, e.title, s."name"), v.volume_number;

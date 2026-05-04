SELECT a.id,
       a."name"
FROM data_author AS a
JOIN data_work_authors AS wa
  ON wa.author_id = a.id
JOIN data_workinanthology AS wia
  ON wia.work_id = wa.work_id
JOIN data_volume AS v
  ON v.id = wia.volume_id
JOIN data_edition AS e
  ON e.id = v.edition_id
JOIN data_edition_literary_traditions AS elt
  ON elt.edition_id = e.id
JOIN data_literarytradition AS lt
  ON lt.id = elt.literarytradition_id
WHERE lt."name" = 'African-American Literature'
GROUP BY a.id, a."name"
HAVING COUNT(DISTINCT e.id) = (
    SELECT COUNT(DISTINCT e2.id)
    FROM data_edition AS e2
    JOIN data_edition_literary_traditions AS elt2
      ON elt2.edition_id = e2.id
    JOIN data_literarytradition AS lt2
      ON lt2.id = elt2.literarytradition_id
    WHERE lt2."name" = 'African-American Literature'
)
ORDER BY a."name";

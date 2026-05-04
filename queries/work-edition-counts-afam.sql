WITH tagged_editions AS (
    SELECT DISTINCT e.id
    FROM data_edition e
    JOIN data_edition_literary_traditions elt
      ON elt.edition_id = e.id
    JOIN data_literarytradition lt
      ON lt.id = elt.literarytradition_id
    WHERE lt."name" = 'African-American Literature'
)
SELECT
    w.id          AS work_id,
    w.title       AS work_title,
    COUNT(DISTINCT e.id) AS edition_count
FROM data_work w
JOIN data_workinanthology wia
  ON wia.work_id = w.id
JOIN data_volume v
  ON v.id = wia.volume_id
JOIN data_edition e
  ON e.id = v.edition_id
JOIN tagged_editions te
  ON te.id = e.id
GROUP BY w.id, w.title
ORDER BY w.title;

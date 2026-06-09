WITH tagged_editions AS (
    SELECT DISTINCT e.id
    FROM data_edition e
    JOIN data_edition_literary_traditions elt
      ON elt.edition_id = e.id
    JOIN data_literarytradition lt
      ON lt.id = elt.literarytradition_id
    WHERE lt."name" = 'African-American Literature'
)
SELECT DISTINCT
    e.id          AS edition_id,
    e."year"      AS anthology_publication_year,
    e.series_id,
    e.edition_number,
    w.id          AS work_id,
    w.title       AS work_title,
    w.parent_id,
    pw.title      AS parent_work_title,
    a.id          AS author_id,
    a."name"      AS author_name,
    a.birth_year  AS author_birth_year
FROM data_work w
JOIN data_workinanthology wia
  ON wia.work_id = w.id
JOIN data_volume v
  ON v.id = wia.volume_id
JOIN data_edition e
  ON e.id = v.edition_id
JOIN tagged_editions te
  ON te.id = e.id
LEFT JOIN data_work pw
  ON pw.id = w.parent_id
LEFT JOIN data_work_authors wa
  ON wa.work_id = w.id
LEFT JOIN data_author a
  ON a.id = wa.author_id
ORDER BY e."year", e.id, w.id, a.id;

-- Per-edition totals over editions tagged "African-American Literature":
-- distinct authors and distinct works in each edition, with the edition's
-- year / series / number. Reselected-from-previous counts are computed in
-- Python by comparing consecutive editions' author/work sets.
--
-- One row per edition (data_edition.id is already volume-collapsed).
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
    e.id            AS edition_id,
    e."year"        AS anthology_publication_year,
    e.series_id     AS series_id,
    e.edition_number,
    e.title         AS edition_title,
    COUNT(DISTINCT wa.author_id) AS author_count,
    COUNT(DISTINCT w.id)         AS work_count
FROM data_edition e
JOIN tagged_editions te
  ON te.id = e.id
JOIN data_volume v
  ON v.edition_id = e.id
JOIN data_workinanthology wia
  ON wia.volume_id = v.id
JOIN data_work w
  ON w.id = wia.work_id
LEFT JOIN data_work_authors wa
  ON wa.work_id = w.id
GROUP BY e.id, e."year", e.series_id, e.edition_number, e.title
ORDER BY e."year", e.id;

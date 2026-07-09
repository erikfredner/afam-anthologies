-- Long-form (work × author × edition) rows across the two Norton series the
-- NAAL-exclusivity scripts compare:
--   series_id = 3  → NAAAL  (Norton Anthology of African American Literature)
--   series_id = 1  → NAFAM  (Norton Anthology of American Literature)
--
-- NOT restricted to the African-American literary tradition (that is the whole
-- point: we want every NAFAM appearance, to subtract it). One row per
-- (work, author, edition); works without an author keep one row with NULL
-- author columns.
SELECT
    w.id          AS work_id,
    w.title       AS work_title,
    w.parent_id   AS parent_id,
    e.id          AS edition_id,
    e.series_id   AS series_id,
    e.edition_number,
    e."year"      AS anthology_publication_year,
    a.id          AS author_id,
    a."name"      AS author_name
FROM data_work w
JOIN data_workinanthology wia
  ON wia.work_id = w.id
JOIN data_volume v
  ON v.id = wia.volume_id
JOIN data_edition e
  ON e.id = v.edition_id
LEFT JOIN data_work_authors wa
  ON wa.work_id = w.id
LEFT JOIN data_author a
  ON a.id = wa.author_id
WHERE e.series_id IN (1, 3)
ORDER BY e.series_id, e."year", e.id, w.id, a.id;

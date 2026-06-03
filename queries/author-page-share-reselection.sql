-- Long-form (work × volume × author × form) rows for every workinanthology
-- record in editions tagged "African-American Literature".  Carries the page
-- columns needed for span computation (toc_page → toc_next, plus volume
-- first_toc_page / last_toc_page as a fallback), the work's parent_id for
-- root-vs-excerpt filtering, the author's name for labeling, and the form
-- name (LEFT JOIN — works without a form keep one row with NULL form).
--
-- Multi-author works produce one row per author; multi-form works produce one
-- row per form; multi-volume editions one row per volume.  Deduplicate
-- downstream as needed.
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
    w.id              AS work_id,
    w.parent_id       AS parent_id,
    v.id              AS volume_id,
    e.id              AS edition_id,
    e."year"          AS edition_year,
    a.id              AS author_id,
    a."name"          AS author_name,
    f.id              AS form_id,
    f."name"          AS form_name,
    wia.toc_page      AS toc_page,
    wia.toc_next      AS toc_next,
    v.first_toc_page  AS first_toc_page,
    v.last_toc_page   AS last_toc_page
FROM data_work w
JOIN data_workinanthology wia
  ON wia.work_id = w.id
JOIN data_volume v
  ON v.id = wia.volume_id
JOIN data_edition e
  ON e.id = v.edition_id
JOIN tagged_editions te
  ON te.id = e.id
LEFT JOIN data_work_authors wa
  ON wa.work_id = w.id
LEFT JOIN data_author a
  ON a.id = wa.author_id
LEFT JOIN data_work_form wf
  ON wf.work_id = w.id
LEFT JOIN data_form f
  ON f.id = wf.form_id
ORDER BY e."year", e.id, v.id, w.id, a.id, f.id;

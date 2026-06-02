-- Long-form (work × volume × author) rows for every workinanthology record
-- in editions tagged "African-American Literature".  Includes the page columns
-- needed to compute per-work page spans (toc_page → toc_next) and per-volume
-- totals (first_toc_page → last_toc_page), plus author birth_year and the
-- work's parent_id for root-vs-excerpt filtering.
--
-- Multi-author works appear once per author; multi-volume editions appear once
-- per volume.  Deduplicate downstream as needed.
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
    w.id            AS work_id,
    w.parent_id     AS parent_id,
    v.id            AS volume_id,
    e.id            AS edition_id,
    e."year"        AS anthology_publication_year,
    a.id            AS author_id,
    a.birth_year    AS author_birth_year,
    wia.toc_page    AS toc_page,
    wia.toc_next    AS toc_next,
    v.first_toc_page AS first_toc_page,
    v.last_toc_page  AS last_toc_page
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
ORDER BY e."year", e.id, v.id, w.id, a.id;

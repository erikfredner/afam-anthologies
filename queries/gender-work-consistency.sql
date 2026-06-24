-- Long-form (work × volume × author × form) rows for every workinanthology
-- record in editions tagged "African-American Literature", carrying the
-- author's gender and the work's form so the gender × work-selection
-- consistency analysis can run entirely in Python.
--
-- One row per (work, volume, author, form). Multi-author works produce one
-- row per author; multi-form works produce one row per form; multi-volume
-- editions one row per volume. Deduplicate downstream as needed.
--
-- gender / form are LEFT JOINed (NULL when absent). Each author carries one
-- gender label: data_author_genders is many-to-many, so we collapse to the
-- lowest genders_id per author (1=Male, 2=Female, 3=Other) for determinism.
WITH tagged_editions AS (
    SELECT DISTINCT e.id
    FROM data_edition e
    JOIN data_edition_literary_traditions elt
      ON elt.edition_id = e.id
    JOIN data_literarytradition lt
      ON lt.id = elt.literarytradition_id
    WHERE lt."name" = 'African-American Literature'
),
author_gender AS (
    SELECT author_id, MIN(genders_id) AS genders_id
    FROM data_author_genders
    GROUP BY author_id
)
SELECT
    w.id              AS work_id,
    w.parent_id       AS parent_id,
    e.id              AS edition_id,
    e."year"          AS edition_year,
    e.series_id       AS series_id,
    a.id              AS author_id,
    a."name"          AS author_name,
    g."name"          AS gender,
    f.id              AS form_id,
    f."name"          AS form_name
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
LEFT JOIN author_gender ag
  ON ag.author_id = a.id
LEFT JOIN data_genders g
  ON g.id = ag.genders_id
LEFT JOIN data_work_form wf
  ON wf.work_id = w.id
LEFT JOIN data_form f
  ON f.id = wf.form_id
ORDER BY e."year", e.id, w.id, a.id, f.id;

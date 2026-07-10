-- One row per (work, edition) for a given author across AFAM-tagged editions,
-- carrying the work's parent (to drop container/section works) and literary
-- form (to filter to poems). Used by analysis/overlap/sterling_brown_poem_overlap.py.
-- Bind :author_id via the %(author_id)s param.
WITH tagged_editions AS (
    SELECT DISTINCT e.id
    FROM data_edition e
    JOIN data_edition_literary_traditions elt ON elt.edition_id = e.id
    JOIN data_literarytradition lt ON lt.id = elt.literarytradition_id
    WHERE lt."name" = 'African-American Literature'
),
work_min_form AS (
    SELECT work_id, MIN(form_id) AS form_id
    FROM data_work_form
    GROUP BY work_id
)
SELECT DISTINCT
    w.id        AS work_id,
    w.title     AS work_title,
    w.parent_id AS parent_id,
    e.id        AS edition_id,
    e."year"    AS year,
    f."name"    AS form
FROM data_work w
JOIN data_work_authors wa     ON wa.work_id = w.id
JOIN data_workinanthology wia ON wia.work_id = w.id
JOIN data_volume v            ON v.id = wia.volume_id
JOIN data_edition e           ON e.id = v.edition_id
JOIN tagged_editions          ON tagged_editions.id = e.id
LEFT JOIN work_min_form wmf    ON wmf.work_id = w.id
LEFT JOIN data_form f          ON f.id = wmf.form_id
WHERE wa.author_id = %(author_id)s;

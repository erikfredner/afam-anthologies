-- For each work that debuted in any AFAM-tagged anthology, return whether
-- it was reselected by any subsequent AFAM anthology (year strictly greater
-- than the debut year).  Works debuting in the latest AFAM year are excluded
-- because they have had no opportunity to be reselected.
WITH afam_editions AS (
    SELECT DISTINCT e.id AS edition_id, e."year"
    FROM data_edition e
    JOIN data_edition_literary_traditions elt
        ON elt.edition_id = e.id
    JOIN data_literarytradition lt
        ON lt.id = elt.literarytradition_id
    WHERE lt."name" = 'African-American Literature'
),
max_afam_year AS (
    SELECT MAX("year") AS max_year FROM afam_editions
),
work_in_afam AS (
    SELECT DISTINCT
        w.id        AS work_id,
        w.parent_id,
        ae.edition_id,
        ae."year"
    FROM data_work w
    JOIN data_workinanthology wia ON wia.work_id = w.id
    JOIN data_volume v            ON v.id = wia.volume_id
    JOIN afam_editions ae         ON ae.edition_id = v.edition_id
),
debuts AS (
    SELECT work_id, parent_id, MIN("year") AS debut_year
    FROM work_in_afam
    GROUP BY work_id, parent_id
),
eligible_debuts AS (
    SELECT d.work_id, d.parent_id, d.debut_year
    FROM debuts d
    CROSS JOIN max_afam_year m
    WHERE d.debut_year < m.max_year
),
reselection AS (
    SELECT
        ed.work_id,
        ed.parent_id,
        ed.debut_year,
        CASE WHEN COUNT(wia.edition_id) > 0 THEN 1 ELSE 0 END AS is_reselected
    FROM eligible_debuts ed
    LEFT JOIN work_in_afam wia
        ON wia.work_id = ed.work_id
       AND wia."year" > ed.debut_year
    GROUP BY ed.work_id, ed.parent_id, ed.debut_year
)
SELECT work_id, parent_id, debut_year, is_reselected
FROM reselection
ORDER BY debut_year, work_id;

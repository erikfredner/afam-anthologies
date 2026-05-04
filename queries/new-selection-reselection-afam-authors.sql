-- For each author that debuted in any AFAM-tagged anthology, return whether
-- they were reselected by any subsequent AFAM anthology (year strictly greater
-- than the debut year).  Authors debuting in the latest AFAM year are excluded
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
author_in_afam AS (
    SELECT DISTINCT
        wa.author_id,
        ae.edition_id,
        ae."year"
    FROM data_work_authors wa
    JOIN data_workinanthology wia ON wia.work_id = wa.work_id
    JOIN data_volume v            ON v.id = wia.volume_id
    JOIN afam_editions ae         ON ae.edition_id = v.edition_id
    WHERE wa.author_id IS NOT NULL
),
debuts AS (
    SELECT author_id, MIN("year") AS debut_year
    FROM author_in_afam
    GROUP BY author_id
),
eligible_debuts AS (
    SELECT d.author_id, d.debut_year
    FROM debuts d
    CROSS JOIN max_afam_year m
    WHERE d.debut_year < m.max_year
),
reselection AS (
    SELECT
        ed.author_id,
        ed.debut_year,
        CASE WHEN COUNT(aia.edition_id) > 0 THEN 1 ELSE 0 END AS is_reselected
    FROM eligible_debuts ed
    LEFT JOIN author_in_afam aia
        ON aia.author_id = ed.author_id
       AND aia."year" > ed.debut_year
    GROUP BY ed.author_id, ed.debut_year
)
SELECT author_id, debut_year, is_reselected
FROM reselection
ORDER BY debut_year, author_id;

WITH series_editions AS (
    SELECT
        e.id AS edition_id,
        e.edition_number,
        e."year",
        ROW_NUMBER() OVER (
            ORDER BY e."year", e.id
        ) AS edition_seq
    FROM data_edition e
    JOIN data_series s
        ON s.id = e.series_id
    WHERE s."name" = 'The Norton Anthology of African American Literature'
),
edition_works AS (
    SELECT DISTINCT
        se.edition_id,
        se.edition_number,
        se.edition_seq,
        wia.work_id
    FROM series_editions se
    JOIN data_volume v
        ON v.edition_id = se.edition_id
    JOIN data_workinanthology wia
        ON wia.volume_id = v.id
    WHERE wia.work_id IS NOT NULL
),
edition_authors AS (
    SELECT DISTINCT
        ew.edition_id,
        ew.edition_number,
        ew.edition_seq,
        wa.author_id
    FROM edition_works ew
    JOIN data_work_authors wa
        ON wa.work_id = ew.work_id
    WHERE wa.author_id IS NOT NULL
),
edition_counts AS (
    SELECT
        se.edition_id,
        se.edition_number,
        se.edition_seq,
        COUNT(DISTINCT ea.author_id) AS authors,
        COUNT(DISTINCT ew.work_id) AS works
    FROM series_editions se
    LEFT JOIN edition_works ew
        ON ew.edition_id = se.edition_id
    LEFT JOIN edition_authors ea
        ON ea.edition_id = se.edition_id
    GROUP BY
        se.edition_id,
        se.edition_number,
        se.edition_seq
),
reselected_authors AS (
    SELECT
        earlier.edition_id AS earlier_edition_id,
        later.edition_id AS later_edition_id,
        COUNT(DISTINCT earlier.author_id) AS authors_reselected
    FROM edition_authors earlier
    JOIN edition_authors later
        ON later.edition_seq = earlier.edition_seq + 1
       AND later.author_id = earlier.author_id
    GROUP BY
        earlier.edition_id,
        later.edition_id
),
reselected_works AS (
    SELECT
        earlier.edition_id AS earlier_edition_id,
        later.edition_id AS later_edition_id,
        COUNT(DISTINCT earlier.work_id) AS works_reselected
    FROM edition_works earlier
    JOIN edition_works later
        ON later.edition_seq = earlier.edition_seq + 1
       AND later.work_id = earlier.work_id
    GROUP BY
        earlier.edition_id,
        later.edition_id
)
SELECT
    ec.edition_number,
    ec.authors,
    COALESCE(ra.authors_reselected, 0) AS authors_reselected,
    CASE
        WHEN ec.authors = 0 THEN 0
        ELSE ROUND(COALESCE(ra.authors_reselected, 0)::numeric * 100 / ec.authors, 2)
    END AS "authors_reselected_%",
    ec.works,
    COALESCE(rw.works_reselected, 0) AS works_reselected,
    CASE
        WHEN ec.works = 0 THEN 0
        ELSE ROUND(COALESCE(rw.works_reselected, 0)::numeric * 100 / ec.works, 2)
    END AS "works_reselected_%"
FROM edition_counts ec
LEFT JOIN series_editions next_ed
    ON next_ed.edition_seq = ec.edition_seq + 1
LEFT JOIN reselected_authors ra
    ON ra.earlier_edition_id = ec.edition_id
   AND ra.later_edition_id = next_ed.edition_id
LEFT JOIN reselected_works rw
    ON rw.earlier_edition_id = ec.edition_id
   AND rw.later_edition_id = next_ed.edition_id
ORDER BY ec.edition_seq;


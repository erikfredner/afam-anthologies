-- Compare work selections between the 1971 Cavalcade: Negro American Writing
-- from 1760 to the Present (1st ed. of Cavalcade series, edition_id=54) and
-- the 1972 Black Writers of America: A Comprehensive Anthology (edition_id=58).
--
-- source: 'Cavalcade' | 'Black Writers of America' | 'Both'
-- prior_count: distinct African-American Literature-tagged edition appearances
--              before 1971 (strictly < 1971, so before either target anthology)

WITH afam_pre1971 AS (
    SELECT DISTINCT e.id AS edition_id
    FROM data_edition e
    JOIN data_edition_literary_traditions elt ON elt.edition_id = e.id
    JOIN data_literarytradition lt ON lt.id = elt.literarytradition_id
    WHERE lt.name = 'African-American Literature'
      AND e.year < 1971
),
cavalcade_works AS (
    SELECT DISTINCT wia.work_id
    FROM data_workinanthology wia
    JOIN data_volume v ON v.id = wia.volume_id
    WHERE v.edition_id = 54
),
bwa_works AS (
    SELECT DISTINCT wia.work_id
    FROM data_workinanthology wia
    JOIN data_volume v ON v.id = wia.volume_id
    WHERE v.edition_id = 58
),
target_works AS (
    SELECT work_id,
        CASE
            WHEN work_id IN (SELECT work_id FROM bwa_works) THEN 'Both'
            ELSE 'Cavalcade'
        END AS source
    FROM cavalcade_works
    UNION ALL
    SELECT work_id, 'Black Writers of America' AS source
    FROM bwa_works
    WHERE work_id NOT IN (SELECT work_id FROM cavalcade_works)
),
prior_counts AS (
    SELECT wia.work_id, COUNT(DISTINCT v.edition_id) AS prior_count
    FROM data_workinanthology wia
    JOIN data_volume v ON v.id = wia.volume_id
    JOIN afam_pre1971 ap ON ap.edition_id = v.edition_id
    GROUP BY wia.work_id
)
SELECT
    w.id                        AS work_id,
    w.title                     AS work_title,
    w.parent_id                 AS parent_work_id,
    pw.title                    AS parent_work_title,
    a.id                        AS author_id,
    a.name                      AS author_name,
    a.birth_year                AS author_birth_year,
    tw.source,
    COALESCE(pc.prior_count, 0) AS prior_count
FROM target_works tw
JOIN data_work w ON w.id = tw.work_id
LEFT JOIN data_work pw ON pw.id = w.parent_id
LEFT JOIN data_work_authors wa ON wa.work_id = w.id
LEFT JOIN data_author a ON a.id = wa.author_id
LEFT JOIN prior_counts pc ON pc.work_id = tw.work_id
ORDER BY prior_count DESC, a.name, w.title;

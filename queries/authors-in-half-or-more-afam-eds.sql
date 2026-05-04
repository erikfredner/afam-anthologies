WITH tagged_editions AS (
    SELECT DISTINCT e.id
    FROM data_edition e
    JOIN data_edition_literary_traditions elt
      ON elt.edition_id = e.id
    JOIN data_literarytradition lt
      ON lt.id = elt.literarytradition_id
    WHERE lt."name" = 'African-American Literature'
),
total_tagged_editions AS (
    SELECT COUNT(*) AS total_editions
    FROM tagged_editions
),
author_edition_counts AS (
    SELECT
        a.id AS author_id,
        a."name" AS author_name,
        a.birth_year,
        a.death_year,
        COUNT(DISTINCT e.id) AS edition_count
    FROM data_author a
    JOIN data_work_authors wa
      ON wa.author_id = a.id
    JOIN data_workinanthology wia
      ON wia.work_id = wa.work_id
    JOIN data_volume v
      ON v.id = wia.volume_id
    JOIN data_edition e
      ON e.id = v.edition_id
    JOIN tagged_editions te
      ON te.id = e.id
    GROUP BY a.id, a."name", a.birth_year, a.death_year
)
SELECT
    aec.author_id,
    aec.author_name,
    aec.birth_year,
    aec.death_year,
    aec.edition_count
FROM author_edition_counts aec
CROSS JOIN total_tagged_editions tte
WHERE aec.edition_count * 2 >= tte.total_editions
ORDER BY aec.edition_count DESC, aec.author_name;

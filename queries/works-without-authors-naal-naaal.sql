WITH naal_work_counts AS (
    SELECT
        e.id,
        e.edition_number,
        e.year,
        COUNT(DISTINCT wia.work_id) AS work_count
    FROM data_edition AS e
    JOIN data_volume AS v ON v.edition_id = e.id
    JOIN data_workinanthology AS wia ON wia.volume_id = v.id
    WHERE e.series_id = 1
    GROUP BY e.id, e.edition_number, e.year
),
full_naal_ids AS (
    SELECT DISTINCT ON (edition_number) id, edition_number, year
    FROM naal_work_counts
    ORDER BY edition_number, work_count DESC
),
naaal_ids AS (
    SELECT id, edition_number, year
    FROM data_edition
    WHERE series_id = 3
),
all_editions AS (
    SELECT id, edition_number, year, 'NAAL' AS anthology
    FROM full_naal_ids
    UNION ALL
    SELECT id, edition_number, year, 'NAAAL' AS anthology
    FROM naaal_ids
),
edition_stats AS (
    SELECT
        ae.anthology,
        ae.year,
        ae.edition_number,
        COUNT(DISTINCT w.id) AS total_works,
        COUNT(DISTINCT CASE WHEN wa.author_id IS NULL THEN w.id END) AS works_without_author
    FROM all_editions AS ae
    JOIN data_volume AS v ON v.edition_id = ae.id
    JOIN data_workinanthology AS wia ON wia.volume_id = v.id
    JOIN data_work AS w ON w.id = wia.work_id
    LEFT JOIN data_work_authors AS wa ON wa.work_id = w.id
    WHERE w.parent_id IS NULL
    GROUP BY ae.anthology, ae.year, ae.edition_number
)
SELECT
    anthology,
    year,
    edition_number,
    total_works,
    works_without_author,
    ROUND(
        works_without_author::numeric * 100 / NULLIF(total_works, 0),
        2
    ) AS pct_without_author
FROM edition_stats
ORDER BY anthology, year;

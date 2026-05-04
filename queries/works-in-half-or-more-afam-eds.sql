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
work_edition_counts AS (
    SELECT
        w.id AS work_id,
        w.title AS work_title,
        COUNT(DISTINCT e.id) AS edition_count
    FROM data_work w
    JOIN data_workinanthology wia
      ON wia.work_id = w.id
    JOIN data_volume v
      ON v.id = wia.volume_id
    JOIN data_edition e
      ON e.id = v.edition_id
    JOIN tagged_editions te
      ON te.id = e.id
    GROUP BY w.id, w.title
),
work_authors AS (
    SELECT
        wa.work_id,
        a."name" AS author_name
    FROM data_work_authors wa
    JOIN data_author a
      ON a.id = wa.author_id
)
SELECT
    wec.work_id,
    wec.work_title,
    wa.author_name,
    wec.edition_count
FROM work_edition_counts wec
JOIN work_authors wa
  ON wa.work_id = wec.work_id
CROSS JOIN total_tagged_editions tte
WHERE wec.edition_count * 2 >= tte.total_editions
ORDER BY wec.edition_count DESC, wec.work_title, wa.author_name;

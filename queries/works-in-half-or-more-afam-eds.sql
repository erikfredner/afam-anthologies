WITH tagged_editions AS (
    SELECT DISTINCT e.id, e."year"
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
        w.parent_id,
        MIN(te."year") AS debut_year,
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
    GROUP BY w.id, w.title, w.parent_id
),
-- Editions selecting the work's family: the root work plus all of its
-- excerpts, grouped by COALESCE(parent_id, id).
coalesced_edition_counts AS (
    SELECT
        COALESCE(w.parent_id, w.id) AS group_id,
        COUNT(DISTINCT e.id) AS coalesced_edition_count
    FROM data_work w
    JOIN data_workinanthology wia
      ON wia.work_id = w.id
    JOIN data_volume v
      ON v.id = wia.volume_id
    JOIN data_edition e
      ON e.id = v.edition_id
    JOIN tagged_editions te
      ON te.id = e.id
    GROUP BY COALESCE(w.parent_id, w.id)
),
work_reselection_stats AS (
    SELECT
        wec.*,
        COUNT(DISTINCT e.id) FILTER (
            WHERE te."year" > wec.debut_year
        ) AS reselection_count,
        opp.opportunities,
        COUNT(DISTINCT e.id) FILTER (
            WHERE te."year" > wec.debut_year
        )::float / NULLIF(opp.opportunities, 0) AS reselection_rate
    FROM work_edition_counts wec
    JOIN data_workinanthology wia
      ON wia.work_id = wec.work_id
    JOIN data_volume v
      ON v.id = wia.volume_id
    JOIN data_edition e
      ON e.id = v.edition_id
    JOIN tagged_editions te
      ON te.id = e.id
    CROSS JOIN LATERAL (
        SELECT COUNT(*) AS opportunities
        FROM tagged_editions later_te
        WHERE later_te."year" > wec.debut_year
    ) opp
    GROUP BY
        wec.work_id,
        wec.work_title,
        wec.parent_id,
        wec.debut_year,
        wec.edition_count,
        opp.opportunities
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
    wrs.work_id,
    wrs.work_title,
    wa.author_name,
    wrs.edition_count,
    cec.coalesced_edition_count,
    wrs.reselection_count,
    wrs.opportunities,
    wrs.reselection_rate
FROM work_reselection_stats wrs
-- LEFT JOIN keeps authorless works (anonymous spirituals, folk songs)
LEFT JOIN work_authors wa
  ON wa.work_id = wrs.work_id
JOIN coalesced_edition_counts cec
  ON cec.group_id = COALESCE(wrs.parent_id, wrs.work_id)
CROSS JOIN total_tagged_editions tte
WHERE wrs.edition_count * 2 >= tte.total_editions
ORDER BY wrs.edition_count DESC, wrs.work_title, wa.author_name;

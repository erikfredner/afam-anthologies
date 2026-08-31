-- Every work ever selected for an African American anthology, with its full
-- selection record. Identical to works-in-half-or-more-afam-eds.sql except that
-- it drops the "half or more" threshold, so all 3,236 works come back.
--
-- Excerpts count in their own right; parent_work_title names the book an
-- excerpt or a collected poem came from. Authorless works keep one row with a
-- NULL author_name (LEFT JOIN) -- 298 of the 3,236 have no attribution in the
-- database, and that absence is data, not a gap to be filled.
--
-- debut_edition_id names the anthology a work debuted in, ties within a year
-- broken by lowest edition id -- the (year, edition_id) debut ordering used
-- elsewhere in the repo. It groups works into debut cohorts, which share an
-- identical opportunities count. Computed on the individual work_id, like the
-- reselection columns, not on the coalesced work family.
WITH tagged_editions AS (
    SELECT DISTINCT e.id, e."year"
    FROM data_edition e
    JOIN data_edition_literary_traditions elt
      ON elt.edition_id = e.id
    JOIN data_literarytradition lt
      ON lt.id = elt.literarytradition_id
    WHERE lt."name" = 'African-American Literature'
),
work_edition_counts AS (
    SELECT
        w.id AS work_id,
        w.title AS work_title,
        w.parent_id,
        MIN(te."year") AS debut_year,
        (ARRAY_AGG(e.id ORDER BY te."year", e.id))[1] AS debut_edition_id,
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
        wec.debut_edition_id,
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
    wrs.parent_id,
    pw.title AS parent_work_title,
    wa.author_name,
    wrs.debut_edition_id,
    wrs.edition_count,
    cec.coalesced_edition_count,
    wrs.reselection_count,
    wrs.opportunities,
    wrs.reselection_rate
FROM work_reselection_stats wrs
-- LEFT JOIN keeps authorless works (anonymous spirituals, folk songs)
LEFT JOIN work_authors wa
  ON wa.work_id = wrs.work_id
-- The book an excerpt or a collected poem was taken from; NULL for root works.
LEFT JOIN data_work pw
  ON pw.id = wrs.parent_id
JOIN coalesced_edition_counts cec
  ON cec.group_id = COALESCE(wrs.parent_id, wrs.work_id)
ORDER BY wrs.edition_count DESC, wrs.work_title, wa.author_name;

-- Every author ever selected for an African American anthology, with their full
-- selection record. Identical to authors-in-half-or-more-afam-eds.sql except
-- that it drops the "half or more" threshold, so all 575 authors come back.
--
-- edition_count    distinct AFAM-tagged editions that selected any work by them
-- reselection_count / opportunities
--                  editions published after their debut that selected them
--                  again, over the number that could have. opportunities is 0
--                  (and reselection_rate NULL) for an author who debuts in the
--                  most recent anthology.
-- debut_edition_id the anthology they debuted in, ties within a year broken by
--                  lowest edition id -- the (year, edition_id) debut ordering
--                  used elsewhere in the repo. Groups authors into debut
--                  cohorts, which share an identical opportunities count.
WITH tagged_editions AS (
    SELECT DISTINCT e.id, e."year"
    FROM data_edition e
    JOIN data_edition_literary_traditions elt
      ON elt.edition_id = e.id
    JOIN data_literarytradition lt
      ON lt.id = elt.literarytradition_id
    WHERE lt."name" = 'African-American Literature'
),
author_edition_counts AS (
    SELECT
        a.id AS author_id,
        a."name" AS author_name,
        a.birth_year,
        a.death_year,
        MIN(te."year") AS debut_year,
        (ARRAY_AGG(e.id ORDER BY te."year", e.id))[1] AS debut_edition_id,
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
),
author_reselection_stats AS (
    SELECT
        aec.*,
        COUNT(DISTINCT e.id) FILTER (
            WHERE te."year" > aec.debut_year
        ) AS reselection_count,
        opp.opportunities,
        COUNT(DISTINCT e.id) FILTER (
            WHERE te."year" > aec.debut_year
        )::float / NULLIF(opp.opportunities, 0) AS reselection_rate
    FROM author_edition_counts aec
    JOIN data_work_authors wa
      ON wa.author_id = aec.author_id
    JOIN data_workinanthology wia
      ON wia.work_id = wa.work_id
    JOIN data_volume v
      ON v.id = wia.volume_id
    JOIN data_edition e
      ON e.id = v.edition_id
    JOIN tagged_editions te
      ON te.id = e.id
    CROSS JOIN LATERAL (
        SELECT COUNT(*) AS opportunities
        FROM tagged_editions later_te
        WHERE later_te."year" > aec.debut_year
    ) opp
    GROUP BY
        aec.author_id,
        aec.author_name,
        aec.birth_year,
        aec.death_year,
        aec.debut_year,
        aec.debut_edition_id,
        aec.edition_count,
        opp.opportunities
)
SELECT
    ars.author_id,
    ars.author_name,
    ars.birth_year,
    ars.death_year,
    ars.debut_edition_id,
    ars.edition_count,
    ars.reselection_count,
    ars.opportunities,
    ars.reselection_rate
FROM author_reselection_stats ars
ORDER BY ars.edition_count DESC, ars.author_name;

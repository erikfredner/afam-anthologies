-- Count of editions tagged with the African-American Literature tradition.
-- Used by analysis/overlap/sterling_brown_poem_overlap.py for its zero-filled
-- "all AFAM anthologies" universe.
SELECT COUNT(DISTINCT e.id) AS n
FROM data_edition e
JOIN data_edition_literary_traditions elt ON elt.edition_id = e.id
JOIN data_literarytradition lt ON lt.id = elt.literarytradition_id
WHERE lt."name" = 'African-American Literature';

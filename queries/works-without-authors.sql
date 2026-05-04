SELECT
    w.id AS work_id,
    w.title AS work_title,
    COUNT(DISTINCT e.id) AS edition_count
FROM data_work AS w
JOIN data_workinanthology AS wia
    ON wia.work_id = w.id
JOIN data_volume AS v
    ON v.id = wia.volume_id
JOIN data_edition AS e
    ON e.id = v.edition_id
JOIN data_edition_literary_traditions AS elt
    ON elt.edition_id = e.id
JOIN data_literarytradition AS lt
    ON lt.id = elt.literarytradition_id
LEFT JOIN data_work_authors AS wa
    ON wa.work_id = w.id
WHERE lt."name" = 'African-American Literature'
  AND wa.work_id IS NULL
GROUP BY
    w.id,
    w.title
ORDER BY
    edition_count DESC,
    w.title;

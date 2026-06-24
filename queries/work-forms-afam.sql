-- Literary form for every work, collapsed to the lowest form_id when a work is
-- tagged with multiple forms (project convention). Excerpt inheritance from a
-- parent work is handled in Python by the caller.
WITH work_min_form AS (
    SELECT
        wf.work_id,
        MIN(wf.form_id) AS form_id
    FROM data_work_form wf
    GROUP BY wf.work_id
)
SELECT
    wmf.work_id   AS work_id,
    wmf.form_id   AS form_id,
    f."name"      AS form_name
FROM work_min_form wmf
JOIN data_form f
  ON f.id = wmf.form_id;

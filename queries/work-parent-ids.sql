-- Distinct work ids that are the parent of at least one other work: collection
-- volumes (e.g. "Harlem Shadows <volume of poems>", "Lyrics of Lowly Life") and
-- parent novels. These are "container" works. Analyses that want the individual
-- selection units (single poems, excerpts) filter these out and keep only leaf
-- works -- works that are not a parent of anything.
SELECT DISTINCT parent_id AS work_id
FROM data_work
WHERE parent_id IS NOT NULL;

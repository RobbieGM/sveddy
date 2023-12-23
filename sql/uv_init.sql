DROP SCHEMA public CASCADE;
CREATE SCHEMA public;
GRANT ALL ON SCHEMA public TO current_user;
GRANT ALL ON SCHEMA public TO public;
CREATE EXTENSION sveddy;

CREATE TABLE ratings (
	uid integer,
	iid integer,
	rating integer
);

INSERT INTO ratings (uid, iid, rating) VALUES
(1, 1, 4),
(1, 2, 3),
(2, 1, 5),
(3, 3, 4);

CALL initialize_model_uv('ratings', 'uid', 'iid', 'rating', 2);

SELECT * FROM sveddy_models_uv;
SELECT id, cardinality(weights) FROM ratings_sveddy_model_u;
SELECT id, cardinality(weights) FROM ratings_sveddy_model_v;


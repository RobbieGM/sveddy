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
(1, 1, 1),
(1, 2, 1),
(1, 3, 1),
(1, 4, 0),
(1, 5, 0),
(1, 6, 0),
(2, 1, 1),
(2, 2, 1),
(2, 3, 1),
(2, 4, 0),
(2, 5, 0),
(2, 6, 0),
(3, 1, 1),
(3, 2, 1),
(3, 3, 1),
(3, 4, 0),
(3, 5, 0),
(3, 6, 0),
(4, 1, 1),
(4, 2, 1),
(4, 3, 1),
(4, 4, 1),
(4, 5, 1),
(4, 6, 1),
(5, 1, -1),
(5, 2, -1),
(5, 3, -1),
(5, 4, 1),
(5, 5, 1),
(5, 6, 1),
(6, 1, -1),
(6, 2, -1),
(6, 3, -1),
(6, 4, 1),
(6, 5, 0), -- outlier
(6, 6, 1),
(7, 1, -1),
(7, 2, -1),
(7, 3, -1),
(7, 4, 1),
(7, 5, 1),
(7, 6, 1);

CALL initialize_model_uv('ratings', 'uid', 'iid', 'rating', 2);
CALL train_uv('ratings', quiet => true);

SELECT rating, round(predict_uv(ratings_sveddy_model_u.weights, ratings_sveddy_model_v.weights)) + 1 - 1 as predicted -- add and subtract one to avoid flaky negative zeros
FROM ratings
JOIN ratings_sveddy_model_u ON ratings_sveddy_model_u.id = ratings.uid
JOIN ratings_sveddy_model_v ON ratings_sveddy_model_v.id = ratings.iid;

-- Setting regularization_factor too high will cause the model to choose very small values in U, V, leading to near-zero predictions
CALL train_uv('ratings', regularization_factor => 1000, quiet => true);
SELECT rating, round(predict_uv(ratings_sveddy_model_u.weights, ratings_sveddy_model_v.weights)) + 1 - 1 as predicted -- add and subtract one to avoid flaky negative zeros
FROM ratings
JOIN ratings_sveddy_model_u ON ratings_sveddy_model_u.id = ratings.uid
JOIN ratings_sveddy_model_v ON ratings_sveddy_model_v.id = ratings.iid;

DROP SCHEMA public CASCADE;
CREATE SCHEMA public;
GRANT ALL ON SCHEMA public TO current_user;
GRANT ALL ON SCHEMA public TO public;

-- Fill in pre-existing data
CREATE TABLE ratings (
    user_id integer,
    movie_id integer,
    rating integer
);
INSERT INTO ratings (user_id, movie_id, rating) VALUES
-- User 1 likes movies 1 and 2 but not 3 and 4
(1, 1, 5),
(1, 2, 5),
(1, 3, 1),
(1, 4, 2),
-- User 2 likes movies 3 and 4 but not 1 and 2
(2, 1, 1),
(2, 2, 2),
(2, 3, 5),
(2, 4, 5),
-- User 3 likes movie 1. What can we predict about their preferences?
(3, 1, 5);

-- Initialize sveddy
CREATE EXTENSION sveddy;
CALL initialize_model_uv(
    'ratings',
    'user_id',
    'movie_id',
    'rating',
    -- A choice of the hyperparameter k = 1 will work best for this very simple data.
    -- For most cases, k should be higher, typically 5-10.
    2
);
-- A (unrealistically high) number of iterations avoids numerical instability and test flakiness here
CALL train_uv('ratings', quiet => true, max_iterations=>200::smallint);

-- Make a prediction about user 3's rating on movie 2
-- With the UV model, this is internally a dot product between user 3's weights and movie 2's weights
SELECT round(predict_uv(
    (SELECT weights FROM ratings_sveddy_model_u WHERE id = 3),
    (SELECT weights FROM ratings_sveddy_model_v WHERE id = 2)
));


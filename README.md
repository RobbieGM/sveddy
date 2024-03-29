# sveddy

Sveddy is an in-database collaborative filtering system for PostgreSQL. Sveddy supports continuous learning, meaning that when a user expresses their preferences, such as by rating an item, those preferences are immediately taken into account without requiring a full model re-train. Sveddy's UV decomposition algorithm performs very well on the Netflix Prize dataset; see the performance section for more details. 

## Example Usage

```sql
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

-- Initialize sveddy and create a UV decomposition model
CREATE EXTENSION sveddy;
CALL initialize_model_uv(
    'ratings',
    'user_id',
    'movie_id',
    'rating',
    -- A choice of the hyperparameter k = 2 will work best for this very simple data.
    -- For most cases, k should be higher, typically 5-15, depending on the amount of data.
    2
);
CALL train_uv('ratings');

-- Make a prediction about user 3's rating on movie 4
-- With the UV model, this is internally a dot product between user 3's weights and movie 4's weights
SELECT predict_uv(
    (SELECT weights FROM ratings_sveddy_model_u WHERE id = 3),
    (SELECT weights FROM ratings_sveddy_model_v WHERE id = 4)
);
```

## Performance
- Netflix dataset, UV model
    - Hardware: Dell Inspiron 7506, unknown 880 MB/s SSD, Intel i7-1165G7, 16GB RAM
    - Dataset: 100,480,507 ratings, 480,189 users, 17,700 movies
    - Parameters: k = 4, regularization\_constant = 0.05
    - Validation RMSE: 0.87 ([top performing entries](https://en.wikipedia.org/wiki/Netflix_Prize) scored RMSEs around 0.85 on a similar test dataset)<sup>[1](#footnote-1)</sup>
    - Model initialization time (initialize\_model\_uv): 79s
    - Training time: 370s (8 iterations)
    - Continuous learning insertion overhead into ratings table: 450ms

<sup><a name='footnote-1'>[1]</a> It's impossible to make a truly fair comparison here, as participants in the netflix challenge had RMSEs calculated from a separate private test set (the "qualifying set"), whereas Sveddy's validation RMSE was calculated by separating out 1/8 of the public training data as the validation set. The main difference between these is that the qualifying set was made up only of recent ratings, whereas Sveddy's validation set is sampled randomly from the public training set. As much as I would like to know Sveddy's RMSE on the qualifying set, the Netflix Prize closed over 10 years ago and is no longer accepting submissions. Entries using similar techniques scored around RMSEs around 0.89. More information about the Netflix Prize datasets [here](https://web.archive.org/web/20070927051207/http://www.netflixprize.com/assets/NetflixPrizeKDD_to_appear.pdf).</sup>

## Building & Installation

### Linux
1. `git clone https://github.com/RobbieGM/sveddy.git && cd sveddy`
2. `make`
    - Use `make SSE=no` if your CPU does not support [SSE3](https://en.wikipedia.org/wiki/Streaming_SIMD_Extensions).
3. Optionally: `make installcheck` to verify tests pass in your environment
4. `make install` to copy Sveddy into PostgreSQL
    - If using NixOS, instead run `nix-shell` to build an environment with a version of PostgreSQL that has Sveddy installed, and then run the modified postgresql server with `postgres`.
5. Run `CREATE EXTENSION sveddy` in your database and you should be good to go!

## Documentation

### `initialize_model_uv`
`initialize_model_uv(source_table name, user_column name, item_column name, rating_column name, k integer, regularization_factor real DEFAULT 0.05)`
Creates a new Sveddy UV model. A UV model works by calculating a low-rank matrix approximation of the source table via UV decomposition. This is very similar to a truncated SVD (singular value decomposition). Because Sveddy will sometimes try to query entries from `source_table` by user id or item id, it's recommended to create indices on the source table for both the `user_column` and `item_column`. This function creates two new tables: a U table, representing user preferences, and a V table, representing item qualities. The U table will have the name `source_table + "_sveddy_model_u"` and the V table will have the name `source_table + "_sveddy_model_v"`. These tables will contain the weights necessary for the low-rank matrix approximation of ratings in the source table. They are initialized with the necessary ids but with random weights. This procedure does not train the model yet; see `train_uv` for more.

Parameters:
- `source_table` The table containing the data to train from. 
- `user_column` The name of the column in the source table which holds the user id. The column must be of type `integer` (`int32`).
- `item_column` The name of the column in the source table which holds the item id. The column must be of type `integer` (`int32`).
- `rating_column` The name of the column in the source table which holds the value to be predicted (e.g. rating out of 5 stars, watch time, +1 for like or -1 for dislike, etc.). This column must have a numeric type.
- `k` The rank of the low-rank matrix approximation. A higher rank allows the model to learn more about the data at the risk of overfitting. In general, the more data you have, the higher you can safely set k.
- `regularization_factor` The weight magnitude penalty in the U and V tables. Increasing this parameter may help reduce model overfitting. When set too high, the model will make predictions much closer to 0 than desired.

Example:
```sql
CALL initialize_model_uv('ratings', 'user_id', 'movie_id', 'rating', 5, 0.1);
```

### `garbage_collect_uv`
`garbage_collect_uv(source_table name)`
Removes rows in the U and V tables that do not correspond to user or item ids in the source table. For example, if a user deletes their account, their preferences are still saved in the U table. This function removes orphaned rows to save space.
Parameters:
- `source_table` the source table of the model to garbage collect.

Example:
```sql
CALL garbage_collect_uv('ratings');
```

### `train_uv`
`train_uv(source_table name, patience int DEFAULT 4, max_iterations int DEFAULT 8, training_validation_split float DEFAULT 0, quiet boolean DEFAULT false)`
Trains the UV model for the source table whose name is given by the first parameter. This will periodically report RMSE. The current implementation of `train_uv` requires slightly over `4*((k*k+k)*max(# users, # items) + k*(# users) + k*(# items))` bytes of memory to function. For example, if the model's k hyperparameter is set to 5 and there is 10GB of free memory, Sveddy can train on a maximum of about 60 million users and 60 million items. This function should be run on a semi regular basis, as the iterative learning done when an entry is inserted or updated in the source table will lead to a loss of model performance over time if a full re-train is not done.

Parameters:
- `source_table` Indicates the source table of the model which will be trained.
- `patience` If more than this number of training iterations pass without RMSE reaching a new low, training will stop.
- `max_iterations` The maximum number of training iterations.
- `training_validation_split` The (approximate) fraction of entries in the source table which will be used for validation instead of training. If set to a nonzero value, early stopping behavior (set by the `patience` parameter) will respond to validation RMSE instead of training RMSE to avoid overfitting. Also, if `quiet` is false, logs will show both training and validation RMSE.
- `quiet` If set to true, RMSE will not be logged.

Example:
```sql
CALL train_uv('ratings', patience => 6, max_iterations => 16);
```

### `predict_uv`
`predict_uv(user_weights real[], item_weights real[])`
Returns the model's prediction for a user on an item. The `user_weights` argument should come from the row in the U table with the id of the user in question, and similarly the `item_weights` arguments should come from the row in the V table with the id of the item in question.

Example:
```sql
SELECT predict_uv(
    (SELECT weights FROM ratings_sveddy_model_u WHERE id = 3),
    (SELECT weights FROM ratings_sveddy_model_v WHERE id = 5)
);
```

Example (recommend five movies to a user they haven't seen before):
```sql
SELECT items.*, predict_uv(u.weights, v.weights) AS predicted_rating
FROM (
    SELECT id, weights FROM ratings_sveddy_model_v
    WHERE NOT EXISTS (
        SELECT 1 FROM ratings 
        WHERE user_id = 3 AND movie_id = ratings_sveddy_model_v.id
    )
) AS v
CROSS JOIN (
    SELECT weights FROM ratings_sveddy_model_u
    WHERE id = 3
) AS u
LEFT JOIN items ON items.id = v.id
ORDER BY predicted_rating DESC
LIMIT 5;
```

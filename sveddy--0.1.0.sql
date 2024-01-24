-- A table holding all sveddy UV models.
CREATE TABLE IF NOT EXISTS sveddy_models_uv (
	-- The table to predict ratings from. It is expected to contain a row for 
	-- each user's rating on an item.
	source_table name,
	-- The name of the column referring to the user id in the source table.
	user_column name, 
	-- The name of the column referring to the item id in the source table.
	item_column name,
	-- The name of the column referring to the rating in the source table.
	rating_column name,
	-- The name of the table which holds the entries of the U matrix.
	u_table name,
	-- The name of the table which holds the entries of the V matrix.
	v_table name,
	-- The hyperparameter k, which is the rank of the low-rank matrix
	-- approximation R ~= UV^T.
	k integer,
	-- The model's regularization factor. Increasing this may help prevent
	-- overfitting.
	regularization_factor real
);
-- Initializes a UV decomposition model for predicting ratings.
-- The source table is expected to contain columns for the user id, item id,
-- and rating. The user id and item id are both expected to be of type integer.
-- The rating can be any numeric type. It is HIGHLY RECOMMENDED to create
-- indices on the source table for both the user_column and item_column, as
-- the trigger that fires when a rating is added or updated will run queries
-- such as SELECT ... FROM source_table WHERE userid = ?.
--
-- This function creates two new tables: a U table and a V table. The U table
-- will have the name source_table + "_sveddy_model_u" and the V table will have
-- the name source_table + "_sveddy_model_v". These tables contain the weights
-- necessary for the low-rank matrix approximation of ratings in the source
-- table.
-- The schema of these tables is:
--		id integer primary key -- refers to user_column in U table, or
--								  item_column in V table
--		weights real[] -- the weights of this row of the model's U or V matrix
-- After the U and V tables are created, they are populated with rows from the
-- source table. Each unique user id (selected from user_column) gets a row in 
-- U, and each unique item id (selected from item_column) gets a row in V.
-- The weights are initialized to k-length array of random numbers.
--
-- The procedure does not train the model.

CREATE OR REPLACE PROCEDURE initialize_model_uv(
	source_table name,
	user_column name,
	item_column name,
	rating_column name,
	k integer,
	regularization_factor real DEFAULT 0.05
)
LANGUAGE plpgsql
AS $$
DECLARE
	u_table_name name := source_table || '_sveddy_model_u';
	v_table_name name := source_table || '_sveddy_model_v';
BEGIN
	-- Create U and V
	EXECUTE format('CREATE TABLE %I (
			id integer PRIMARY KEY,
			weights real[]
		);', u_table_name);

	EXECUTE format('CREATE TABLE %I (
			id integer PRIMARY KEY,
			weights real[]
		);', v_table_name);

	-- Fill U and V
	EXECUTE format('INSERT INTO %I (id, weights)
				SELECT %I, get_initial_weights_uv(%s)
				FROM %I
				GROUP BY %I;', u_table_name, user_column, k, source_table, user_column);

	EXECUTE format('INSERT INTO %I (id, weights)
				SELECT %I, get_initial_weights_uv(%s)
				FROM %I
				GROUP BY %I;', v_table_name, item_column, k, source_table, item_column);

	-- Triggers to update the model. Deletion trigger not needed because we aren't going to update weights on delete
	EXECUTE format('CREATE OR REPLACE TRIGGER update_model_uv
		AFTER INSERT OR UPDATE ON %I 
		FOR EACH ROW EXECUTE FUNCTION update_model_uv();', source_table, source_table);

	-- Insert the new model into the UV models table
	EXECUTE format('INSERT INTO sveddy_models_uv (
		source_table,
		user_column,
		item_column,
		rating_column,
		u_table,
		v_table,
		k,
		regularization_factor
	) VALUES (%L, %L, %L, %L, %L, %L, %L, %L)
	', source_table, user_column, item_column, rating_column, u_table_name, v_table_name, k, regularization_factor);
END;
$$;

CREATE OR REPLACE PROCEDURE garbage_collect_uv(
	source_table name
)
LANGUAGE plpgsql
AS $$
DECLARE
	u_table_name name := source_table || '_sveddy_model_u';
	v_table_name name := source_table || '_sveddy_model_v';
	user_column name;
	item_column name;
BEGIN
	EXECUTE format('SELECT user_column, item_column
		FROM sveddy_models_uv
		WHERE source_table = %L', source_table) INTO user_column, item_column;
	EXECUTE format('DELETE FROM %1$I u
		WHERE NOT EXISTS (
			SELECT FROM %2I r WHERE r.%3$I = u.id
		)
		', u_table_name, source_table, user_column);
	EXECUTE format('DELETE FROM %1$I v
		WHERE NOT EXISTS (
			SELECT FROM %2I r WHERE r.%3$I = v.id
		)
		', v_table_name, source_table, item_column);
END;
$$;


CREATE OR REPLACE FUNCTION
get_initial_weights_uv(integer) RETURNS real[]
AS 'MODULE_PATHNAME' LANGUAGE C STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION
predict_uv(real[], real[]) RETURNS real
AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Procedure for updating U and V after inserting into source_table. If the
-- inserted record refers to a new user or item, these functions add the
-- necessary rows to U or V. They also update the model.
CREATE OR REPLACE FUNCTION
update_model_uv() RETURNS TRIGGER 
AS 'MODULE_PATHNAME' LANGUAGE C STRICT;

CREATE OR REPLACE PROCEDURE
train_uv(source_table name, patience int DEFAULT 4, max_iterations int DEFAULT 8, training_validation_split float DEFAULT 0, quiet boolean DEFAULT false)
AS 'MODULE_PATHNAME' LANGUAGE C;

DROP SCHEMA public CASCADE;
CREATE SCHEMA public;
GRANT ALL ON SCHEMA public TO current_user;
GRANT ALL ON SCHEMA public TO public;
CREATE EXTENSION sveddy;

SELECT cardinality(arr) from get_initial_weights_uv(8) as arr;

DROP SCHEMA public CASCADE;
CREATE SCHEMA public;
GRANT ALL ON SCHEMA public TO current_user;
GRANT ALL ON SCHEMA public TO public;
CREATE EXTENSION sveddy;


SELECT predict_uv('{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}', '{0.4, 0.5, 0.6, 0.7, 0.8, 0.9}');

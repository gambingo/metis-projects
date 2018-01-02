-- Stored procedure to add a new column from a csv into "cleaned"
-- The COPY command below fails on my machine due to access issues.
-- Runninging /COPY manually will fix this, but sql meta-commands cannot
-- be used functions of stored procedures

CREATE OR REPLACE FUNCTION add_column(col_name TEXT)
RETURNS void AS $$

DECLARE statement TEXT;

BEGIN
  CREATE TABLE temp (
  id DOUBLE PRECISION,
  col_name TEXT,
  PRIMARY KEY (id));

  statement := 'COPY temp FROM '/Users/Joe/Documents/Metis/Projects/metis-03-mcnulty/data/temp_csv/temp.csv' DELIMITER ',' CSV HEADER;';
  EXECUTE statment;

  ALTER TABLE
  clean
  ADD COLUMN col_name TEXT;

  UPDATE clean
  SET col_name = temp.col_name
  FROM temp
  WHERE clean.id = temp.id;

  DROP TABLE temp;
END;
 $$ LANGUAGE plpgsql;

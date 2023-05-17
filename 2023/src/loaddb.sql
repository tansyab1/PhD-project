-- each table in the database and save it to a .csv file
SELECT * FROM `1` INTO OUTFILE './1.csv' FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n';
SELECT * FROM `2` INTO OUTFILE './2.csv' FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n';

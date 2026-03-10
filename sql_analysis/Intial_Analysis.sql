select label, count(*) from  mlops.ml_features_full_dump mffd 
where mffd.partition_file   = 'day_2.csv'  
group by label ;


create table "mlops"."ml_features_full_partition_2" as 
select *  FROM "mlops"."ml_features_full_dump" 
WHERE partition_file = 'day_2.csv'
;


CREATE INDEX idx_ml_features_loaded_at ON mlops.ml_features_full_part_test USING brin (loaded_at);

CREATE INDEX idx_ml_features_part_file ON mlops.ml_features_full_part_test USING btree (partition_file);

CREATE INDEX idx_ml_features_part_label ON mlops.ml_features_full_part_test USING btree (partition_file, label);


CREATE INDEX idx_ml_features_integer ON mlops.ml_features_full_part_test USING btree ("label",
int_feature_1,
int_feature_2,
int_feature_3,
int_feature_4,
int_feature_5,
int_feature_6,
int_feature_7,
int_feature_8,
int_feature_9,
int_feature_10,
int_feature_11,
int_feature_12,
int_feature_13);

int_feature_1

select
'int_feature_1' column_name ,
    COUNT(*) AS total_rows,
    COUNT(int_feature_1) AS non_null,
    COUNT(*) - COUNT( int_feature_1) AS missing,
    AVG(int_feature_1) AS mean,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY int_feature_1) AS median,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY int_feature_1) AS p99,
    MAX(int_feature_1) AS max_value
FROM mlops.ml_features_full_part_test
union 
select
'int_feature_2' column_name ,
    COUNT(*) AS total_rows,
    COUNT(int_feature_2) AS non_null,
    COUNT(*) - COUNT( int_feature_2) AS missing,
    AVG(int_feature_2) AS mean,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY int_feature_2) AS median,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY int_feature_2) AS p99,
    MAX(int_feature_2) AS i2_max_value
FROM mlops.ml_features_full_part_test
union 
select
'int_feature_3' column ,
    COUNT(*) AS total_rows,
    COUNT(int_feature_3) AS i1_non_null,
    COUNT(*) - COUNT( int_feature_3) AS missing,
    AVG(int_feature_3) AS mean,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY int_feature_3) AS median,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY int_feature_3) AS p99,
    MAX(int_feature_3) AS max_value
FROM mlops.ml_features_full_part_test
union 
select
'int_feature_4' column ,
    COUNT(*) AS total_rows,
    COUNT(int_feature_4) AS non_null,
    COUNT(*) - COUNT( int_feature_4) AS missing,
    AVG(int_feature_4) AS mean,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY int_feature_4) AS median,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY int_feature_4) AS p99,
    MAX(int_feature_4) AS max_value
FROM mlops.ml_features_full_part_test
union 
select
'int_feature_5' column ,
    COUNT(*) AS total_rows,
    COUNT(int_feature_5) AS i1_non_null,
    COUNT(*) - COUNT(int_feature_5) AS missing,
    AVG(int_feature_5) AS mean,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY int_feature_5) AS median,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY int_feature_5) AS p99,
    MAX(int_feature_5) AS max_value
FROM mlops.ml_features_full_part_test
union 
select
'int_feature_6' column ,
    COUNT(*) AS total_rows,
    COUNT(int_feature_6) AS i1_non_null,
    COUNT(*) - COUNT( int_feature_6) AS missing,
    AVG(int_feature_6) AS mean,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY int_feature_6) AS median,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY int_feature_6) AS p99,
    MAX(int_feature_6) AS max_value
FROM mlops.ml_features_full_part_test
union 
select
'int_feature_7' column ,
    COUNT(*) AS total_rows,
    COUNT(int_feature_7) AS i1_non_null,
    COUNT(*) - COUNT( int_feature_7) AS missing,
    AVG(int_feature_7) AS mean,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY int_feature_7) AS median,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY int_feature_7) AS p99,
    MAX(int_feature_7) AS max_value
FROM mlops.ml_features_full_part_test
union 
select
'int_feature_8' column ,
    COUNT(*) AS total_rows,
    COUNT(int_feature_8) AS i1_non_null,
    COUNT(*) - COUNT( int_feature_8) AS missing,
    AVG(int_feature_8) AS mean,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY int_feature_8) AS median,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY int_feature_8) AS p99,
    MAX(int_feature_8) AS max_value
FROM mlops.ml_features_full_part_test
union 
select
'int_feature_9' column ,
    COUNT(*) AS total_rows,
    COUNT(int_feature_9) AS i1_non_null,
    COUNT(*) - COUNT( int_feature_9) AS missing,
    AVG(int_feature_9) AS mean,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY int_feature_9) AS median,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY int_feature_9) AS p99,
    MAX(int_feature_9) AS max_value
FROM mlops.ml_features_full_part_test
union 
select
'int_feature_10' column ,
    COUNT(*) AS total_rows,
    COUNT(int_feature_10) AS i1_non_null,
    COUNT(*) - COUNT( int_feature_10) AS missing,
    AVG(int_feature_10) AS mean,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY int_feature_10) AS median,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY int_feature_10) AS p99,
    MAX(int_feature_10) AS max_value
FROM mlops.ml_features_full_part_test
union 
select
'int_feature_11' column ,
    COUNT(*) AS total_rows,
    COUNT(int_feature_11) AS i1_non_null,
    COUNT(*) - COUNT( int_feature_11) AS missing,
    AVG(int_feature_11) AS mean,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY int_feature_11) AS median,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY int_feature_11) AS p99,
    MAX(int_feature_11) AS max_value
FROM mlops.ml_features_full_part_test
union 
select
'int_feature_12' column ,
    COUNT(*) AS total_rows,
    COUNT(int_feature_12) AS i1_non_null,
    COUNT(*) - COUNT( int_feature_12) AS missing,
    AVG(int_feature_12) AS mean,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY int_feature_12) AS median,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY int_feature_12) AS p99,
    MAX(int_feature_12) AS max_value
FROM mlops.ml_features_full_part_test
union 
select
'int_feature_13' column ,
    COUNT(*) AS total_rows,
    COUNT(int_feature_13) AS i1_non_null,
    COUNT(*) - COUNT( int_feature_13) AS missing,
    AVG(int_feature_13) AS mean,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY int_feature_13) AS median,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY int_feature_13) AS p99,
    MAX(int_feature_13) AS max_value
FROM mlops.ml_features_full_part_test
;

,cat_feature_2,cat_feature_3,cat_feature_4,cat_feature_5,cat_feature_6,cat_feature_7,cat_feature_8,cat_feature_9,cat_feature_10,cat_feature_11,cat_feature_12,cat_feature_13


select
'cat_feature_1' column_name ,
    COUNT(*) AS total_rows,
    sum (case when cat_feature_1  in ('__MISSING__','__RARE__') then 1 else 0 end ) AS non_null,
    COUNT(*) - sum (case when cat_feature_1  in ('__MISSING__','__RARE__') then 1 else 0 end ) AS non_missing
FROM mlops.ml_features_full_part_test;


SELECT
    v.column_name,
    COUNT(*) AS total_rows,
    SUM(CASE WHEN v.value = '__RARE__' THEN 1 ELSE 0 END) AS rare_count,
    SUM(CASE WHEN v.value = '__MISSING__' THEN 1 ELSE 0 END) AS missing_count,
    SUM(CASE WHEN v.value IN ('__RARE__','__MISSING__') THEN 1 ELSE 0 END) AS rare_or_missing_count,
    COUNT(*) - SUM(CASE WHEN v.value IN ('__RARE__','__MISSING__') THEN 1 ELSE 0 END) AS non_rare_non_missing_count
FROM mlops.ml_features_full_part_test t
CROSS JOIN LATERAL (
    VALUES
        ('cat_feature_1', t.cat_feature_1),
        ('cat_feature_2', t.cat_feature_2),
        ('cat_feature_3', t.cat_feature_3),
        ('cat_feature_4', t.cat_feature_4),
        ('cat_feature_5', t.cat_feature_5),
        ('cat_feature_6', t.cat_feature_6),
        ('cat_feature_7', t.cat_feature_7),
        ('cat_feature_8', t.cat_feature_8),
        ('cat_feature_9', t.cat_feature_9),
        ('cat_feature_10', t.cat_feature_10),
        ('cat_feature_11', t.cat_feature_11),
        ('cat_feature_12', t.cat_feature_12),
        ('cat_feature_13', t.cat_feature_13),
        ('cat_feature_14', t.cat_feature_14),
        ('cat_feature_15', t.cat_feature_15),
        ('cat_feature_16', t.cat_feature_16),
        ('cat_feature_17', t.cat_feature_17),
        ('cat_feature_18', t.cat_feature_18),
        ('cat_feature_19', t.cat_feature_19),
        ('cat_feature_20', t.cat_feature_20),
        ('cat_feature_21', t.cat_feature_21),
        ('cat_feature_22', t.cat_feature_22),
        ('cat_feature_23', t.cat_feature_23),
        ('cat_feature_24', t.cat_feature_24),
        ('cat_feature_25', t.cat_feature_25),
        ('cat_feature_26', t.cat_feature_26)
) AS v(column_name, value)
GROUP BY v.column_name
ORDER BY v.column_name;

--- bucket analysis on the Integer feature columns with univariate analysis 
/*
  "int_medians": {
    "int_feature_1": 8,
    "int_feature_2": 189,
    "int_feature_3": 4,
    "int_feature_4": 34,
    "int_feature_5": 6,
    "int_feature_6": 0,
    "int_feature_7": 0,
    "int_feature_8": 6,
    "int_feature_9": 5,
    "int_feature_10": 0,
    "int_feature_11": 2,
    "int_feature_12": 2872,
    "int_feature_13": 5
  } */

SELECT
    MAX(int_feature_12),
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY int_feature_12),
    PERCENTILE_CONT(0.999) WITHIN GROUP (ORDER BY int_feature_12)
    from  mlops.ml_features_full_part_test t

SELECT
    v.column_name,
    width_bucket(v.value, 0, 15, 5) AS bucket,
    COUNT(*) AS impressions,
   SUM(label)::float / COUNT(*) AS ctr
FROM mlops.ml_features_full_part_test t
CROSS JOIN LATERAL (
    VALUES
        ('int_feature_1',  t.int_feature_1::float8),
        ('int_feature_2',  t.int_feature_2::float8),
        ('int_feature_3',  t.int_feature_3::float8),
        ('int_feature_4',  t.int_feature_4::float8),
        ('int_feature_5',  t.int_feature_5::float8),
        ('int_feature_6',  t.int_feature_6::float8),
        ('int_feature_7',  t.int_feature_7::float8),
        ('int_feature_8',  t.int_feature_8::float8),
        ('int_feature_9',  t.int_feature_9::float8),
        ('int_feature_10', t.int_feature_10::float8),
        ('int_feature_11', t.int_feature_11::float8),
        ('int_feature_12', t.int_feature_12::float8),
        ('int_feature_13', t.int_feature_13::float8)
) AS v(column_name, value)
WHERE t.partition_file LIKE 'day_2%'   -- or use: t.day = 'day_2'
  AND v.value IS NOT NULL
GROUP BY v.column_name, bucket
ORDER BY v.column_name, bucket;





# https://duckdb.org/docs/stable/extensions/vss.html
import duckdb

# DB名
table_name = "test_vss_table"
index_name = "test_hnsw"

# テスト用クエリ
create_table_query = f"""
INSTALL vss;
LOAD vss;

CREATE TABLE {table_name} (vec FLOAT[3]);
INSERT INTO {table_name}
    SELECT array_value(a, b, c)
    FROM range(1, 10) ra(a), range(1,10) rb(b), range(1, 10) rc(c);
CREATE INDEX {index_name} ON {table_name} USING HNSW (vec);
"""
select_by_similarity_query = f"""
SELECT *
FROM {table_name}
ORDER BY array_distance(vec, [1,2,3]::FLOAT[3])
LIMIT 3;
"""

# テーブルを作成
duckdb.query(create_table_query)

# 類似度検索
result = duckdb.query(select_by_similarity_query)
print(result)
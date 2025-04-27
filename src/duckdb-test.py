# https://qiita.com/ak-sakatoku/items/54ed6ab29708ed4a6bb9
import duckdb
import streamlit as st

# タイトルを出力
st.title("DuckDB test")

# バージョンを出力
st.write(f"DuckDB version: {duckdb.__version__}")

# vector用のDBを作成
table_name = "test_vector_table"
duckdb.query(f"CREATE OR REPLACE TABLE {table_name} (i INTEGER, v FLOAT[3])")

# DBにデータを挿入
index = 0
for x in range(11):
    for y in range(11):
        for z in range(11):
            duckdb.query(f"INSERT INTO {table_name} VALUES ({index}, [{x}, {y}, {z}])")
            index += 1

# vectorを取得して表示
vectors = duckdb.query(f"SELECT * FROM {table_name}").df()
st.dataframe(vectors)

# 類似度検索
x = st.slider("x", 0.0, 10.0, 5.0, step=0.1)
x = st.slider("y", 0.0, 10.0, 5.0, step=0.1)
x = st.slider("z", 0.0, 10.0, 5.0, step=0.1)
similarities = duckdb.query(f"SELECT *, array_distance(v, [{x}, {y}, {z}]::FLOAT[3]) as d FROM {table_name} ORDER BY d LIMIT 30").df()
st.dataframe(similarities)
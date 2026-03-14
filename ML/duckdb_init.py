import duckdb

parquet_glob = r""
con = duckdb.connect()

print(con.execute(f"SELECT COUNT(*) FROM read_parquet('{parquet_glob}')").fetchall())
print(con.execute(f"SELECT MIN(snapshot_ts), MAX(snapshot_ts) FROM read_parquet('{parquet_glob}')").fetchall())
print(con.execute(f"SELECT COUNT(DISTINCT link_id) FROM read_parquet('{parquet_glob}')").fetchall())
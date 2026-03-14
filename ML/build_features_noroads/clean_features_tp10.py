import duckdb
import os

OUT_DIR = r"C"
RAW = os.path.join(OUT_DIR, "features_7d_tp10.parquet")
CLEAN = os.path.join(OUT_DIR, "features_7d_tp10_clean.parquet")

# No road_link features yet, to test a naive model

con = duckdb.connect()
con.execute(f"""
COPY (
    SELECT *
    FROM read_parquet('{RAW}')
    WHERE sb_tm5 IS NOT NULL AND sb_tm10 IS NOT NULL AND sb_tm15 IS NOT NULL AND y_tp10 IS NOT NULL            
)
TO '{CLEAN}' (FORMAT PARQUET);
""")
print("Wrote:", CLEAN)
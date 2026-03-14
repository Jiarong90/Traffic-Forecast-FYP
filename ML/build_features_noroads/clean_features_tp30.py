import duckdb
import os

OUT_DIR = r"C:\Users\Admin\Desktop\Sugarcane\UNI\Y2 Sem 1\FYP\Traffic Forecast Project\Data\ml_datasets"
RAW = os.path.join(OUT_DIR, "features_7d_tp30.parquet")
CLEAN = os.path.join(OUT_DIR, "features_7d_tp30_clean.parquet")


con = duckdb.connect()
con.execute(f"""
COPY (
    SELECT *
    FROM read_parquet('{RAW}')
    WHERE sb_tm5 IS NOT NULL AND sb_tm10 IS NOT NULL AND sb_tm15 IS NOT NULL AND y_tp30 IS NOT NULL            
)
TO '{CLEAN}' (FORMAT PARQUET);
""")
print("Wrote:", CLEAN)
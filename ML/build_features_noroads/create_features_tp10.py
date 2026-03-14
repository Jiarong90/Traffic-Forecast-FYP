import os
import duckdb

SPEED_GLOB = r"C:\Users\Admin\Desktop\Sugarcane\UNI\Y2 Sem 1\FYP\Traffic Forecast Project\Data\parquet_exports\speedbands_by_chunk\speedbands_chunk_*.parquet"
OUT_DIR = r"C:\Users\Admin\Desktop\Sugarcane\UNI\Y2 Sem 1\FYP\Traffic Forecast Project\Data\ml_datasets"
os.makedirs(OUT_DIR, exist_ok=True)

OUT = os.path.join(OUT_DIR, "features_7d_tp10.parquet")

START_TS = 1770040500
END_TS = 1770645300

OUT_SQL = OUT.replace("\\", "/")
SPEED_SQL = SPEED_GLOB.replace("\\", "/")

con = duckdb.connect()
con.execute("PRAGMA threads=4")

con.execute(f"""
COPY (
    SELECT
        link_id,
        snapshot_ts,
        speed_band AS sb,
        LAG(speed_band, 1) OVER w AS sb_tm5,
        LAG(speed_band, 2) OVER w as sb_tm10,
        LAG(speed_band, 3) OVER w as sb_tm15,
        EXTRACT('hour' FROM to_timestamp(snapshot_ts)) AS hour,
        EXTRACT('dow' FROM to_timestamp(snapshot_ts)) as dow,
        LEAD(speed_band, 2) OVER w as y_tp10
    FROM read_parquet('{SPEED_SQL}')
    WHERE snapshot_ts BETWEEN {START_TS} AND {END_TS}
    WINDOW w AS (PARTITION BY link_id ORDER BY snapshot_ts)        
)
TO '{OUT_SQL}' (FORMAT PARQUET);         
""")

print("Wrote:", OUT)
import os
import duckdb

SPEED_GLOB = r"C:\Users\Admin\Desktop\Sugarcane\UNI\Y2 Sem 1\FYP\Traffic Forecast Project\Data\parquet_exports\speedbands_by_chunk\speedbands_chunk_*.parquet"
ROAD_LINKS = r"C:\Users\Admin\Desktop\Sugarcane\UNI\Y2 Sem 1\FYP\Traffic Forecast Project\Data\parquet_exports\road_links.parquet"
INCIDENTS = r"C:\Users\Admin\Desktop\Sugarcane\UNI\Y2 Sem 1\FYP\Traffic Forecast Project\ml\incidents_road_mapped.parquet" 

OUT_DIR = r"C:\Users\Admin\Desktop\Sugarcane\UNI\Y2 Sem 1\FYP\Traffic Forecast Project\Data\ml_datasets"
os.makedirs(OUT_DIR, exist_ok=True)

OUT = os.path.join(OUT_DIR, "features_7d_incidents_tp15.parquet")

START_TS = 1770040500
END_TS = 1770645300

OUT_SQL = OUT.replace("\\", "/")
SPEED_SQL = SPEED_GLOB.replace("\\", "/")
INC_SQL = INCIDENTS.replace("\\", "/")

con = duckdb.connect()


con.execute("PRAGMA preserve_insertion_order=false")
con.execute("PRAGMA threads=4")
con.execute(r"PRAGMA temp_directory='D:\FYP\duckdb_temp'") 

BIN_SCALE = 200

print("Building Feature Table...")

con.execute(f"""
COPY (
    WITH traffic AS (
        SELECT
            link_id,
            snapshot_ts,
            speed_band AS sb,
            LAG(speed_band, 1) OVER w AS sb_tm5,
            LAG(speed_band, 2) OVER w AS sb_tm10,
            LAG(speed_band, 3) OVER w AS sb_tm15,
            LEAD(speed_band, 3) OVER w AS y_tp15
        FROM read_parquet('{SPEED_SQL}')
        WHERE snapshot_ts BETWEEN {START_TS} AND {END_TS}
        WINDOW w AS (PARTITION BY link_id ORDER BY snapshot_ts)
    ),
    incidents AS (
        SELECT 
            nearest_link_id, 
            epoch(CAST(start_time_utc AS TIMESTAMP)) AS start_ts,
            epoch(CAST(end_time_utc AS TIMESTAMP)) AS end_ts
        FROM read_parquet('{INC_SQL}')
        GROUP BY 1, 2, 3
    )
    SELECT
        f.link_id,
        f.snapshot_ts,
        f.sb,
        f.sb_tm5,
        f.sb_tm10,
        f.sb_tm15,
        f.y_tp15,

        r.road_category,

        ((r.start_lat + r.end_lat) / 2.0) AS mid_lat,
        ((r.start_lon + r.end_lon) / 2.0) AS mid_lon,
        floor(((r.start_lat + r.end_lat) / 2.0) * {BIN_SCALE}) AS lat_bin,
        floor(((r.start_lon + r.end_lon) / 2.0) * {BIN_SCALE}) AS lon_bin,
        
        CASE WHEN EXISTS (
            SELECT 1 FROM incidents inc 
            WHERE inc.nearest_link_id = f.link_id 
              AND f.snapshot_ts >= inc.start_ts 
              AND f.snapshot_ts <= inc.end_ts
        ) THEN 1 ELSE 0 END AS has_incident

    FROM traffic f
    JOIN read_parquet('{ROAD_LINKS}') r
      ON f.link_id = r.link_id

    WHERE f.sb BETWEEN 1 AND 8
      AND f.sb_tm5 BETWEEN 1 AND 8
      AND f.sb_tm10 BETWEEN 1 AND 8
      AND f.sb_tm15 BETWEEN 1 AND 8
      AND f.y_tp15 BETWEEN 1 AND 8
)
TO '{OUT_SQL}' (FORMAT PARQUET);
""")

print("Successfully wrote to:", OUT)
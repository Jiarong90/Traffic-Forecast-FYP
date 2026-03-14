import os
import duckdb

SPEED_GLOB = r"C:\Users\Admin\Desktop\Sugarcane\UNI\Y2 Sem 1\FYP\Traffic Forecast Project\Data\parquet_exports\speedbands_by_chunk\speedbands_chunk_*.parquet"
ROAD_LINKS = r"C:\Users\Admin\Desktop\Sugarcane\UNI\Y2 Sem 1\FYP\Traffic Forecast Project\Data\parquet_exports\road_links.parquet"
OUT_DIR = r"C:\Users\Admin\Desktop\Sugarcane\UNI\Y2 Sem 1\FYP\Traffic Forecast Project\Data\ml_datasets"
os.makedirs(OUT_DIR, exist_ok=True)

OUT = os.path.join(OUT_DIR, "features_7d_roads_tp15.parquet")

START_TS = 1770040500
END_TS = 1770645300

OUT_SQL = OUT.replace("\\", "/")
SPEED_SQL = SPEED_GLOB.replace("\\", "/")

con = duckdb.connect()
con.execute("PRAGMA threads=4")

BIN_SCALE = 200

# Join road_link table, start using road link features


con.execute(f"""
COPY (
    SELECT
        f.link_id,
        f.snapshot_ts,
        f.sb,
        f.sb_tm5,
        f.sb_tm10,
        f.sb_tm15,
        f.hour_sg,
        f.dow_sg,
        f.y_tp15,

        r.road_category,

        ((r.start_lat + r.end_lat) / 2.0) AS mid_lat,
        ((r.start_lon + r.end_lon) / 2.0) AS mid_lon,
        floor(((r.start_lat + r.end_lat) / 2.0) * {BIN_SCALE}) AS lat_bin,
        floor(((r.start_lon + r.end_lon) / 2.0) * {BIN_SCALE}) AS lon_bin

    FROM (
        SELECT
            link_id,
            snapshot_ts,
            speed_band AS sb,
            LAG(speed_band, 1) OVER w AS sb_tm5,
            LAG(speed_band, 2) OVER w AS sb_tm10,
            LAG(speed_band, 3) OVER w AS sb_tm15,

            EXTRACT('hour' FROM (to_timestamp(snapshot_ts) + INTERVAL '8 hours')) AS hour_sg,
            EXTRACT('dow'  FROM (to_timestamp(snapshot_ts) + INTERVAL '8 hours')) AS dow_sg,

            LEAD(speed_band, 3) OVER w AS y_tp15

        FROM read_parquet('{SPEED_SQL}')
        WHERE snapshot_ts BETWEEN {START_TS} AND {END_TS}
        WINDOW w AS (PARTITION BY link_id ORDER BY snapshot_ts)
    ) f
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

print("Wrote:", OUT)
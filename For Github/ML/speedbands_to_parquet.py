import os, sqlite3, pandas as pd, time

DB_PATH = r""
OUT_DIR = r""
CHUNK_SIZE = 10_000_000
COMPRESSION = "snappy"

os.makedirs(OUT_DIR, exist_ok=True)

conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)

print("Starting export stream...")

query = "SELECT id, link_id, speed_band, min_speed, max_speed, snapshot_time FROM speedbands"

chunk_idx = 0
# Read query with chunksize iterator
for df in pd.read_sql_query(query, conn, chunksize=CHUNK_SIZE):
    t0 = time.perf_counter()
    
    for c in ["id", "link_id", "speed_band", "min_speed", "max_speed"]:
        df[c] = pd.to_numeric(df[c], downcast="integer")
        
    # Handle timestamps
    ts = pd.to_datetime(df["snapshot_time"], utc=True, errors="coerce")
    if ts.isna().any():
        print(f"Error in {chunk_idx}")
        
    # Convert to unix timestamp, SQLite3 store it as string
    df["snapshot_ts"] = (ts.astype("int64") // 1_000_000_000)
    df = df.drop(columns=["snapshot_time"])
    
    t1 = time.perf_counter()
    
    # Save to parquet
    out_path = os.path.join(OUT_DIR, f"speedbands_chunk_{chunk_idx}.parquet")
    df.to_parquet(out_path, index=False, compression=COMPRESSION)
    
    t2 = time.perf_counter()
    
    print(f"Chunk written {chunk_idx}")
    chunk_idx += 1

conn.close()
print("Done.")
import sqlite3, pandas as pd, time

DB_PATH = r""
conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)

# Check if query works 

t0=time.time()
print(pd.read_sql_query("SELECT id, link_id, speed_band, min_speed, max_speed, snapshot_time FROM speedbands LIMIT 5", conn))
print("LIMIT 5 secs:", time.time()-t0)

conn.close()
import pandas as pd
import numpy as np
import sqlite3
from sklearn.neighbors import BallTree

DB_PATH = r"trafficdata.db"
INCIDENT_PARQUET = "incidents_cleaned.parquet"

incidents = pd.read_parquet(INCIDENT_PARQUET)

conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)

road_query = """
    SELECT link_id, start_lat, start_lon, end_lat, end_lon
    FROM road_links
"""

road_links_data = pd.read_sql_query(road_query, conn)

# Calculate midpoints from start and end points to determine middle of road, for coordinate matching
road_links_data["mid_lat"] = (road_links_data["start_lat"] + road_links_data["end_lat"]) / 2
road_links_data["mid_lon"] = (road_links_data["start_lon"] + road_links_data["end_lon"]) / 2

road_coords = np.radians(road_links_data[["mid_lat", "mid_lon"]].values)
incidents_coords = np.radians(incidents[["lat", "lon"]].values)

# Use Balltree to create a search structure
tree = BallTree(road_coords, metric="haversine")
# For every incident coordinate, check the nearest neighbor
dist, idx = tree.query(incidents_coords, k=1)
dist_m = dist[:,0] * 6371000

# Take nearest result for each incideint, get the link_id
nearest_indices = idx[:, 0]
nearest_rows = road_links_data.iloc[nearest_indices]
nearest_link_ids = nearest_rows["link_id"].values


incidents["nearest_link_id"] = nearest_link_ids
incidents["dist_to_road_m"] = dist_m



print(incidents.head())
out = "incidents_road_mapped.parquet"
incidents.to_parquet(out, index=False)

print("Rows:", len(incidents))
print(incidents["dist_to_road_m"].describe(percentiles=[0.5, 0.9, 0.99]))
print("Over 200m:", (incidents["dist_to_road_m"] > 200).sum())
print("Over 500m:", (incidents["dist_to_road_m"] > 500).sum())

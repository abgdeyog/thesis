from pymongo import MongoClient
conn = MongoClient()
db_name = "fall_detection"
optical_flow_frames_collection = conn[db_name]["optical_flow_frames"]
last_frames_collection = conn[db_name]["last_frames"]
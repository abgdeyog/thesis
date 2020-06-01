from collections import deque
from src.db_config import optical_flow_frames_collection, last_frames_collection
import pickle
from bson.binary import Binary

stack_size = 20

# function to upload the new optical flow image to stack of fixed size in db
def push_image_to_optical_flow_stack(frame_x, frame_y, source_id):
    try:
        optical_flow_frames = deque(pickle.loads(optical_flow_frames_collection.find_one({"source_id": source_id})['frames']))
        length = len(optical_flow_frames)
    except:
        length = 0
        optical_flow_frames = []
        optical_flow_frames_collection.insert_one({'source_id': source_id, 'frames': pickle.dumps([])})
    if length < stack_size:
        optical_flow_frames.append(frame_x)
        optical_flow_frames.append(frame_y)
    else:
        optical_flow_frames.popleft()
        optical_flow_frames.popleft()
        optical_flow_frames.append(frame_x)
        optical_flow_frames.append(frame_y)
    optical_flow_frames_collection.update({"source_id": source_id}, {"$set": {"frames": Binary(pickle.dumps(optical_flow_frames))}})
    return length

# request the optical flow stack from db
def get_optical_flow_stack(source_id):
    data = optical_flow_frames_collection.find_one({'source_id': source_id})
    if data is not None:
        return pickle.loads(data['frames'])
    else:
        return []


# upload the new image from input to db
def load_new_frame(frame, source_id):
    frame = Binary(pickle.dumps(frame))
    last_frames_collection.update({'source_id': source_id}, {'$set': {'source_id': source_id, 'frame': frame}},
                                  upsert=True)


# method that returns the last frame, that was stored in db
def get_last_frame(source_id):
    data = last_frames_collection.find_one({'source_id': source_id})
    if data is not None:
        return pickle.loads(data['frame'])
    else:
        return None

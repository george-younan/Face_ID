"""
Face Recognition Module - Handles face detection, recognition, registration, and data management.
"""

import os
import json
import threading
import time
import cv2
import face_recognition
import numpy as np
from datetime import datetime

FACES_DIR = "faces"
DATA_FILE = os.path.join(FACES_DIR, "data.json")
os.makedirs(FACES_DIR, exist_ok=True)

# Detection runs on a downscaled frame; the HOG pyramid is not upsampled.
DETECT_SCALE = 0.5
DETECT_UPSAMPLE = 0
MATCH_THRESHOLD = 0.5

# Building the 128-d encoding costs ~70 ms per face, so labels are reused
# between refreshes and whenever the faces in view have not changed.
MIN_ENCODE_INTERVAL = 0.5


def _box_iou(a, b):
    """Intersection over union of two (top, right, bottom, left) boxes."""
    top = max(a[0], b[0])
    bottom = min(a[2], b[2])
    left = max(a[3], b[3])
    right = min(a[1], b[1])
    if bottom <= top or right <= left:
        return 0.0

    intersection = (bottom - top) * (right - left)
    area_a = (a[2] - a[0]) * (a[1] - a[3])
    area_b = (b[2] - b[0]) * (b[1] - b[3])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


# Load or initialize face data
if os.path.exists(DATA_FILE):
    try:
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        data = {"faces": [], "last_updated": str(datetime.now())}
else:
    data = {"faces": [], "last_updated": str(datetime.now())}

def save_data():
    """Save face data to JSON file"""
    data["last_updated"] = str(datetime.now())
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)

class FaceRecognition:
    def __init__(self):
        self.known_encodings = []
        self.known_names = []
        self.known_faces_data = []
        # Per-frame track cache: [{"box": (t, r, b, l), "encoding": ..., "name": ...}]
        self._tracks = []
        self._last_encode = 0.0
        self._encode_interval = MIN_ENCODE_INTERVAL
        # detect_and_recognize is re-entrant across browser tabs; serialise it.
        self._lock = threading.Lock()
        self.load_known_faces()

    def load_known_faces(self):
        """Load known faces from data file"""
        with self._lock:
            self.known_encodings = []
            self.known_names = []
            self.known_faces_data = []

            for face in data["faces"]:
                try:
                    encoding = np.array(face["encoding"])
                    self.known_encodings.append(encoding)
                    self.known_names.append(face["name"])
                    self.known_faces_data.append(face)
                except Exception as e:
                    print(f"Error loading face {face.get('name', 'Unknown')}: {e}")

            # Labels may have changed (registration/deletion), so re-encode next frame.
            self._tracks = []
            self._encode_interval = MIN_ENCODE_INTERVAL

        print(f"Loaded {len(self.known_names)} known faces")
        return self.known_faces_data

    def detect_and_recognize(self, frame):
        """Detect and recognize faces in a frame.

        Detection runs on every frame. The expensive encoding step only runs when
        the faces in view change or the refresh interval expires; in between, the
        previous labels are carried over by matching boxes between frames.
        """
        with self._lock:
            return self._detect_and_recognize(frame)

    def _detect_and_recognize(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=DETECT_SCALE, fy=DETECT_SCALE)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_locations = face_recognition.face_locations(
            rgb_frame, model="hog", number_of_times_to_upsample=DETECT_UPSAMPLE
        )

        mapping = self._match_tracks(face_locations)
        if mapping is not None and time.monotonic() - self._last_encode < self._encode_interval:
            # Cheap path: same faces, reuse their encodings and labels.
            encodings = [self._tracks[i]["encoding"] for i in mapping]
            names = [self._tracks[i]["name"] for i in mapping]
        else:
            started = time.monotonic()
            encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            names = self._recognize(encodings)
            self._last_encode = time.monotonic()
            # Keep the expensive step to a minority of the frame budget.
            self._encode_interval = max(
                MIN_ENCODE_INTERVAL, (self._last_encode - started) * 3
            )

        self._tracks = [
            {"box": box, "encoding": encoding, "name": name}
            for box, encoding, name in zip(face_locations, encodings, names)
        ]

        # Scale coordinates back to original size
        inverse_scale = 1 / DETECT_SCALE
        scaled_locations = [
            (
                int(top * inverse_scale),
                int(right * inverse_scale),
                int(bottom * inverse_scale),
                int(left * inverse_scale),
            )
            for (top, right, bottom, left) in face_locations
        ]

        return scaled_locations, names, encodings

    def _recognize(self, encodings):
        """Match encodings against the registered faces."""
        names = ["Unknown"] * len(encodings)

        if len(self.known_encodings) == 0:
            return names

        for i, encoding in enumerate(encodings):
            distances = face_recognition.face_distance(self.known_encodings, encoding)
            if len(distances) > 0:
                min_distance = np.min(distances)
                if min_distance < MATCH_THRESHOLD:
                    names[i] = self.known_names[int(np.argmin(distances))]

        return names

    def _match_tracks(self, face_locations):
        """Map each detected box to the matching cached track, or None if any is new."""
        if len(face_locations) != len(self._tracks):
            return None

        mapping = []
        used = set()
        for box in face_locations:
            best_index, best_iou = None, 0.3
            for index, track in enumerate(self._tracks):
                if index in used:
                    continue
                iou = _box_iou(box, track["box"])
                if iou > best_iou:
                    best_index, best_iou = index, iou
            if best_index is None:
                return None
            used.add(best_index)
            mapping.append(best_index)

        return mapping

    def register_face(self, face_encoding, name, face_image, extra_data=None):
        """Register a new face"""
        try:
            # Check if name already exists
            for face in data["faces"]:
                if face["name"].lower() == name.lower():
                    # Update existing face
                    face["encoding"] = face_encoding.tolist()
                    face.update(extra_data or {})
                    face["updated_at"] = str(datetime.now())
                    save_data()
                    self.load_known_faces()
                    return True
            
            # Create new face entry
            idx = len(data["faces"]) + 1
            filename = f"{name}_{idx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            path = os.path.join(FACES_DIR, filename)
            
            # Save face image
            cv2.imwrite(path, face_image)
            
            # Create entry
            entry = {
                "name": name,
                "filename": filename,
                "encoding": face_encoding.tolist(),
                "registered_at": str(datetime.now()),
                "updated_at": str(datetime.now())
            }
            
            if extra_data:
                entry.update(extra_data)
            
            data["faces"].append(entry)
            save_data()
            self.load_known_faces()
            
            return True
            
        except Exception as e:
            print(f"Error registering face: {e}")
            return False

    def get_face_count(self):
        """Get total number of registered faces"""
        return len(data["faces"])

    def delete_face(self, name):
        """Delete a registered face"""
        for i, face in enumerate(data["faces"]):
            if face["name"].lower() == name.lower():
                # Remove image file
                img_path = os.path.join(FACES_DIR, face["filename"])
                if os.path.exists(img_path):
                    os.remove(img_path)
                
                # Remove from data
                data["faces"].pop(i)
                save_data()
                self.load_known_faces()
                return True
        return False
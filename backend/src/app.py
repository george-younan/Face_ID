"""
The main Flask application for real-time face recognition.
Handles video streaming, face detection, registration, and data endpoints.
"""

import os
import cv2
import json
import numpy as np
import base64
import threading
import time
from collections import deque
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from faces import FaceRecognition, FACES_DIR

# Serve the built Vue app (run `pnpm build` in frontend/) from frontend/dist.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIST_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "frontend", "dist"))

app = Flask(__name__, static_folder=DIST_DIR, static_url_path="")
CORS(app, origins=["http://localhost:5173"])

fr = FaceRecognition()

# The browser owns camera capture. Flask only receives frames for processing.
detecting = False
current_frame_faces = []
current_frame_data = []
frame_lock = threading.Lock()

# Real statistics (all guarded by stats_lock)
SERVER_START = time.time()
stats_lock = threading.Lock()
frame_times = deque(maxlen=30)  # timestamps of recently processed frames
known_detections = 0            # face observations matched to a registered person
unknown_detections = 0          # face observations that matched nobody

def process_camera_frame(frame):
    """Run recognition on one frame received from the browser."""
    global known_detections, unknown_detections

    face_locations, names, encodings = fr.detect_and_recognize(frame)
    response_faces = []
    unknown_faces = []
    known_faces = []

    for (top, right, bottom, left), name, encoding in zip(face_locations, names, encodings):
        # Clamp coordinates before cropping to avoid invalid NumPy slices.
        height, width = frame.shape[:2]
        top, bottom = max(0, top), min(height, bottom)
        left, right = max(0, left), min(width, right)
        if bottom <= top or right <= left:
            continue

        # Crop the face for registration and the detected/known face lists.
        face_crop = frame[top:bottom, left:right].copy()
        ok, buffer = cv2.imencode('.jpg', face_crop)
        if not ok:
            continue
        face_image = buffer.tobytes()

        location = [top, right, bottom, left]
        response_faces.append({"name": name, "location": location})

        if name == "Unknown":
            unknown_faces.append({
                "encoding": encoding,
                "image": face_image,
                "location": tuple(location),
            })
        else:
            known_faces.append({
                "name": name,
                "image": face_image,
                "extra_data": next(
                    (item for item in fr.known_faces_data if item["name"] == name), None
                ),
            })

    # Swap the shared state under a short lock; encoding and recognition happen outside it.
    with frame_lock:
        current_frame_faces.clear()
        current_frame_faces.extend(unknown_faces)
        current_frame_data.clear()
        current_frame_data.extend(known_faces)

    unknown_count = sum(1 for face in response_faces if face["name"] == "Unknown")
    with stats_lock:
        unknown_detections += unknown_count
        known_detections += len(response_faces) - unknown_count
        frame_times.append(time.time())

    return response_faces


@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Accept one JPEG frame from the browser and return face coordinates."""
    if not detecting:
        return jsonify({"success": False, "message": "Detection is not running"}), 409

    if not request.data:
        return jsonify({"success": False, "message": "Empty frame"}), 400

    try:
        encoded = np.frombuffer(request.data, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"success": False, "message": "Invalid image frame"}), 400

        faces = process_camera_frame(frame)
        return jsonify({"success": True, "faces": faces, "width": frame.shape[1], "height": frame.shape[0]})
    except Exception as exc:
        app.logger.exception("Failed to process browser camera frame")
        return jsonify({"success": False, "message": str(exc)}), 500


@app.route('/')
def index():
    if not os.path.exists(os.path.join(DIST_DIR, "index.html")):
        return ("Frontend not built. Run `pnpm build` in frontend/, "
                "or use `pnpm dev` and open http://localhost:5173.", 404)
    return send_from_directory(DIST_DIR, "index.html")

@app.route('/faces_count')
def faces_count():
    with frame_lock:
        total_faces = len(current_frame_faces) + len(current_frame_data)
    return jsonify({"faces": total_faces, "fps": current_fps()})

def current_fps():
    """Real processing rate, measured from the timestamps of the last frames."""
    with stats_lock:
        if not detecting or len(frame_times) < 2:
            return 0.0
        if time.time() - frame_times[-1] > 2:  # stalled
            return 0.0
        span = frame_times[-1] - frame_times[0]
        return round((len(frame_times) - 1) / span, 1) if span > 0 else 0.0

def db_size_bytes():
    """Total size of everything in the faces directory (images + data.json)."""
    total = 0
    if os.path.isdir(FACES_DIR):
        for entry in os.scandir(FACES_DIR):
            if entry.is_file():
                total += entry.stat().st_size
    return total

@app.route('/stats')
def stats():
    with stats_lock:
        known, unknown = known_detections, unknown_detections
    total = known + unknown
    return jsonify({
        "total_registered": fr.get_face_count(),
        "recognition_rate": round(known / total * 100) if total else None,
        "known_detections": known,
        "unknown_detections": unknown,
        "db_bytes": db_size_bytes(),
        "uptime_seconds": int(time.time() - SERVER_START),
        "detecting": detecting,
    })

@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    """Start/stop recognition; the browser is responsible for camera capture."""
    global detecting
    payload = request.get_json(silent=True) or {}
    action = payload.get("action")

    if action == "start":
        if not detecting:
            detecting = True
            with stats_lock:
                frame_times.clear()
            with frame_lock:
                current_frame_faces.clear()
                current_frame_data.clear()
        return jsonify({"success": True, "status": "started", "message": "Face detection started"})

    if action == "stop":
        detecting = False
        with frame_lock:
            current_frame_faces.clear()
            current_frame_data.clear()
        return jsonify({"success": True, "status": "stopped", "message": "Detection stopped"})

    return jsonify({"success": False, "message": "Invalid action"}), 400

@app.route('/register_face', methods=['POST'])
def register_face():
    try:
        idx = int(request.form.get("index"))
        name = request.form.get("name", "").strip()
        dob = request.form.get("dob", "").strip()
        nid = request.form.get("nid", "").strip()
        
        if not name:
            return jsonify({
                "success": False,
                "message": "Name is required"
            })
        
        extra_data = {}
        if dob:
            extra_data["dob"] = dob
        if nid:
            extra_data["national_id"] = nid
        
        with frame_lock:
            # Take a snapshot of current unknown faces for reference
            unknown_snapshot = current_frame_faces.copy()
            
            if idx >= len(unknown_snapshot):
                return jsonify({
                    "success": False,
                    "message": "Face no longer available. Please refresh the list."
                })
            
            face = unknown_snapshot[idx]
            face_image = cv2.imdecode(
                np.frombuffer(face["image"], np.uint8), 
                cv2.IMREAD_COLOR
            )
            
            # Register the face
            success = fr.register_face(
                face["encoding"], 
                name, 
                face_image, 
                extra_data
            )
            
            if success:
                # Remove this face from current frame faces
                if idx < len(current_frame_faces):
                    current_frame_faces.pop(idx)
                
                return jsonify({
                    "success": True,
                    "message": f"{name} registered successfully!"
                })
            else:
                return jsonify({
                    "success": False,
                    "message": "Failed to register face"
                })
                
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error: {str(e)}"
        })

@app.route('/delete_face', methods=['POST'])
def delete_face():
    """Remove a registered face (database entry and image file)"""
    try:
        payload = request.get_json(silent=True) or {}
        name = (payload.get("name") or "").strip()
        if not name:
            return jsonify({"success": False, "message": "Name is required"}), 400
        
        if fr.delete_face(name):
            return jsonify({"success": True, "message": f"{name} removed"})
        return jsonify({"success": False, "message": f"{name} not found"}), 404
    except Exception as e:
        return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500

@app.route('/known_faces_list')
def known_faces_list():
    """Return list of all registered faces (static data)"""
    try:
        faces_list = []
        known_faces = fr.load_known_faces()
        
        for face in known_faces:
            try:
                img_path = os.path.join(FACES_DIR, face["filename"])
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        _, buffer = cv2.imencode('.jpg', img)
                        img_b64 = base64.b64encode(buffer).decode('utf-8')
                        
                        face_data = {
                            "name": face["name"],
                            "image": img_b64,
                            "extra_data": {
                                k: v for k, v in face.items() 
                                if k not in ["name", "filename", "encoding"]
                            }
                        }
                        faces_list.append(face_data)
            except Exception as e:
                print(f"Error loading face {face.get('name', 'Unknown')}: {e}")
        
        return jsonify(faces_list)
        
    except Exception as e:
        print(f"Error in known_faces_list: {e}")
        return jsonify([])

@app.route('/static_unknown_faces')
def static_unknown_faces():
    """Return a static snapshot of current unknown faces"""
    try:
        with frame_lock:
            # Create a snapshot that won't change
            snapshot = []
            for idx, face in enumerate(current_frame_faces):
                img_b64 = base64.b64encode(face["image"]).decode('utf-8')
                snapshot.append({
                    "index": idx,
                    "image": img_b64,
                    "timestamp": time.time()  # For cache busting
                })
        
        return jsonify(snapshot)
        
    except Exception as e:
        print(f"Error in static_unknown_faces: {e}")
        return jsonify([])

@app.route('/update_unknown_snapshot', methods=['POST'])
def update_unknown_snapshot():
    """Update the snapshot of unknown faces"""
    try:
        with frame_lock:
            snapshot = []
            for idx, face in enumerate(current_frame_faces):
                img_b64 = base64.b64encode(face["image"]).decode('utf-8')
                snapshot.append({
                    "index": idx,
                    "image": img_b64,
                    "timestamp": time.time()
                })
        
        return jsonify({
            "success": True,
            "snapshot": snapshot
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        })

@app.route('/detected_faces')
def detected_faces():
    """Return faces detected in current frame"""
    try:
        with frame_lock:
            faces_list = []
            for f in current_frame_data:
                img_b64 = base64.b64encode(f["image"]).decode('utf-8')
                faces_list.append({
                    "name": f["name"],
                    "extra_data": f.get("extra_data"),
                    "image": img_b64
                })
        
        return jsonify(faces_list)
        
    except Exception as e:
        print(f"Error in detected_faces: {e}")
        return jsonify([])

@app.route('/refresh_faces')
def refresh_faces():
    """Force refresh of face data"""
    try:
        fr.load_known_faces()
        return jsonify({
            "success": True,
            "message": "Faces database refreshed"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error refreshing faces: {str(e)}"
        })

@app.route('/camera_status')
def camera_status():
    """Camera devices belong to the browser, not the Flask host."""
    return jsonify({
        "available": False,
        "detecting": detecting,
        "message": "Select and enable a camera in the browser"
    })


@app.route('/test_camera')
def test_camera():
    return jsonify({
        "success": False,
        "message": "Camera testing is handled by the browser. Use the camera controls in the UI."
    }), 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
from common import *
from mjpeg_streamer import MjpegServer, Stream
from insightface.app import FaceAnalysis
from PIL import Image
from datetime import datetime, timedelta
import streamsync as ss
import pandas as pd
import numpy as np
import imutils
import cv2
import requests
import copy
import os

np.int = int
RES_X = 1280
RES_Y = 720
CAP_X = 1280
CAP_Y = 720
INTERVAL = 120
NUM_DIMENSIONS = 512
DEFAULT_THRESHOLD = 1.0
DET_THRESHOLD = 0.4
DET_SIZE = (640, 640)
FPS = 20
DB_PATH = './../db/'
LOG_FILE = './../logs.csv'

placeholder_preview = open("./static/placeholder_preview.png", "rb").read()
placeholder = cv2.imdecode(np.frombuffer(open("./static/placeholder.jpg", "rb").read(), np.uint8), cv2.IMREAD_COLOR)

def process_image(img, faces, matches, threshold, log, send, url, location, live_result, interval):
    img = draw_boxes(img, faces)
    dets = []
    for face in faces:
        neighbors, distances = index.query(face.normed_embedding, k=matches)
        if distances[0] < threshold:
            name = names[neighbors[0]][0]
        else:
            name = "Unknown"
        cv2.rectangle(img, (int(face.bbox[0]), int(face.bbox[1]) - 30), (int(face.bbox[0]) + len(name)*10 + 20, int(face.bbox[1])), (255, 255, 255), cv2.FILLED)
        cv2.putText(img,
                name,
                (int(face.bbox[0] + 10), int(face.bbox[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1)
        if (send or log) and (name != "Unknown"):
            if not (database[name]["latest"] and (datetime.now() - database[name]["latest"] < timedelta(seconds=interval))):
                if send:
                    # send_data(f"{url}?name={requests.utils.quote(name)}&location={requests.utils.quote(location)}")
                    send_data(url, name, location)
                if log:
                    with open(LOG_FILE, "a") as of:
                        of.write(f"{datetime.now()},{name}\n")
            database[name]["latest"] = datetime.now()
        if live_result:
            dets.extend([(str(datetime.now()), names[neighbors[neighbor]][0], names[neighbors[neighbor]][1], neighbor + 1, distances[neighbor]) for neighbor in range(len(neighbors))])
    return img, dets

def play(state):
    try:
        params = copy.deepcopy(state["params"])
        if params["src"].isdigit():
            src = int(params["src"])
        else:
            src = params["src"]
        matches = min(len(index), int(params["matches"]))
        url = params["url"]
        location = params["location"]
        log = params["log"]["0"] == "yes"
        send = params["send"]["0"] == "yes"
        live_result = params["live_result"]["0"] == "yes"
        save = params["save"]["0"] == "yes"
        res_x = int(params["res_x"])
        res_y = int(params["res_y"])
        cap_x = int(params["cap_x"])
        cap_y = int(params["cap_y"])
        interval = int(params["send_interval"])
        if not live_result:
            matches = 1
    except Exception as e:
        state.add_notification("error", "Error processing parameters", str(e))
        return
    try:
        threshold = float(params["threshold"])
    except:
        threshold = DEFAULT_THRESHOLD
    vc = cv2.VideoCapture(src)
    vc.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    try:
        vc.set(cv2.CAP_PROP_FRAME_WIDTH, cap_x)
        vc.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_y)
        vc.set(cv2.CAP_PROP_FPS, FPS)
        # width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # fps = vc.get(cv2.CAP_PROP_FPS)
        state["running"] = "yes"
        while state["running"] == "yes":
            _, img = vc.read()
            if img is None:
                state.add_notification("error", "Cannot open source", "Failed to open video source.")
                stream.set_frame(placeholder)
                state["running"] = "no"
                break
            if (res_x != cap_x) or (res_y != cap_y):
                img = img[:res_x,:res_y]
            faces = app.get(img)
            if state["saving"] == "yes":
                person_name = state["filename"]
                if person_name not in database:
                    database[person_name] = {"latest": None}
                cur = 0
                while f"{cur}.png" in sorted(database[person_name]):
                    cur += 1
                if save:
                    save_path = os.path.join(DB_PATH, person_name)
                    os.makedirs(save_path, exist_ok=True)
                    cv2.imwrite(os.path.join(save_path, f"{cur}.png"), img)
                if faces: update_index(faces, person_name, f"{cur}.png")
                state.add_notification("success", "Saved new face", f"Saved new face of {person_name}.")
                state["saving"] = "no"
            img, dets = process_image(img, faces, matches, threshold, log, send, url, location, live_result, interval)
            if live_result and dets:
                state["results"] = pd.concat([state["results"], pd.DataFrame(columns=("Time", "Identity", "Image", "Ranking", "Distance",), data=dets)], axis=0)
            stream.set_frame(img)
    except Exception as e:
        raise(e)
    finally:
        stream.set_frame(placeholder)
        vc.release()
        state["running"] = "no"

def stop(state):
    stream.set_frame(placeholder)
    state["running"] = "no"
    state.add_notification("info", "Stopping", "Stopping face recognition system...")

def save_face(state):
    state["saving"] = "yes"

def update_index(faces, person_name, img_name):
    database[person_name][img_name] = len(names)
    names.append((person_name, img_name))
    index.add_item(faces[get_largest_face(faces)].normed_embedding)

def rebuild_index(state):
    state["rebuilding"] = "yes"
    global names, database, index
    names, database, index = load_faces(app, DB_PATH, CAP_X, CAP_Y, NUM_DIMENSIONS)
    refresh_lib(state)
    state.add_notification("success", "Refreshed faces", "Faces successfully refreshed.")
    state["rebuilding"] = "no"

def ui_send(state):
    state["params"]["send"] = {
        "0": "yes",
        "1": "no",
    }

def ui_send_not(state):
    state["params"]["send"] = {
        "0": "no",
        "1": "yes",
    }

def ui_live_results(state):
    state["params"]["live_result"] = {
        "0": "yes",
        "1": "no",
    }

def ui_live_results_not(state):
    state["params"]["live_result"] = {
        "0": "no",
        "1": "yes",
    }

def ui_save(state):
    state["params"]["save"] = {
        "0": "yes",
        "1": "no",
    }

def ui_save_not(state):
    state["params"]["save"] = {
        "0": "no",
        "1": "yes",
    }

def ui_log(state):
    state["params"]["log"] = {
        "0": "yes",
        "1": "no",
    }

def ui_log_not(state):
    state["params"]["log"] = {
        "0": "no",
        "1": "yes",
    }

def refresh_lib(state):
    refresh_people(state)
    refresh_images(state)

def refresh_people(state):
    dirs = sorted(database.keys())
    state["lib"]["selected_person"] = None
    state["lib"]["select_person"] = dict(zip(dirs, dirs))

def refresh_images(state):
    state["lib"]["viewer"] = placeholder_preview
    state["lib"]["selected_image"] = None
    if state["lib"]["selected_person"] is None:
        state["lib"]["select_image"] = {}
    else:
        images = sorted(database[state["lib"]["selected_person"]].keys())
        state["lib"]["select_image"] = dict(zip(images, images))

def select_person(state):
    refresh_images(state)

def select_image(state):
    if (state["lib"]["selected_person"] != None) and (state["lib"]["selected_image"] != None):
        try:
            img = np.array(Image.open(os.path.join(DB_PATH, state["lib"]["selected_person"].replace("/", "."), state["lib"]["selected_image"])))[:, :, ::-1]
            # img = imutils.resize(img, width=CAP_X)[:CAP_Y,:]
            state["lib"]["viewer"] = cv2.imencode(".png", img)[1].tobytes()
            faces = app.get(img)
            img = draw_boxes(img, faces)
            for face in faces:
                neighbors, distances = index.query(face.normed_embedding, k=1)
                if distances[0] < float(state["params"]["threshold"]):
                    name = names[neighbors[0]][0]
                else:
                    name = "Unknown"
                cv2.rectangle(img, (int(face.bbox[0]), int(face.bbox[1]) - 30), (int(face.bbox[0]) + len(name)*10 + 20, int(face.bbox[1])), (255, 255, 255), cv2.FILLED)
                cv2.putText(img,
                        name,
                        (int(face.bbox[0] + 10), int(face.bbox[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1)
            state["lib"]["viewer"] = cv2.imencode(".png", img)[1].tobytes()
        except:
            state["lib"]["viewer"] = placeholder_preview
    else:
        state["lib"]["viewer"] = placeholder_preview

def lib_delete(state):
    if (state["lib"]["selected_person"] != None) and (state["lib"]["selected_image"] != None):
        person_name = state["lib"]["selected_person"]
        img_name = state["lib"]["selected_image"]
        save = state["params"]["save"]["0"] == "yes"

        index.mark_deleted(database[person_name][img_name])
        if save: os.remove(os.path.join(DB_PATH, person_name, img_name))
        database[person_name].pop(img_name)
        refresh_images(state)
        if database[person_name] == {}:
            if save: os.rmdir(os.path.join(DB_PATH, person_name))
            database.pop(person_name)
            refresh_people(state)
    else:
        state.add_notification("error", "No image selected", "No image selected to delete.")

def clear_results(state):
    state["results"] = pd.DataFrame(columns=("Time", "Identity", "Image", "Ranking", "Distance",), data=[(str(datetime.min), "SAMPLE", "0.png", 1, 1.0)])

server = MjpegServer("0.0.0.0", 8501)
stream = Stream("stream", quality=100, fps=FPS)
stream.set_frame(placeholder)
server.add_stream(stream)
server.start()

app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_thresh=DET_THRESHOLD, det_size=DET_SIZE)

names, database, index = load_faces(app, DB_PATH, CAP_X, CAP_Y, NUM_DIMENSIONS)

initial_state = ss.init_state({
    "image": "http://0.0.0.0:8501/0",
    "running": "no",
    "results": pd.DataFrame(columns=("Time", "Identity", "Image", "Ranking", "Distance",), data=[(str(datetime.min), "SAMPLE", "0.png", 1, 1.0)]),
    "filename": "",
    "saving": "no",
    "rebuilding": "no",
    "lib": {
        "viewer": placeholder_preview,
        "select_person": {},
        "select_image": {},
        "selected_person": None,
        "selected_image": None,
    },
    "params": {
        "src": "0",
        "matches": "1",
        "location": "Entrance",
        "threshold": str(DEFAULT_THRESHOLD),
        "url": "http://0.0.0.0:8080/",
        "res_x": str(CAP_X),
        "res_y": str(CAP_Y),
        "cap_x": str(CAP_X),
        "cap_y": str(CAP_Y),
        "send_interval": str(INTERVAL),
        "send": {
            "0": "yes",
            "1": "no",
        },
        "live_result": {
            "0": "yes",
            "1": "no",
        },
        "save": {
            "0": "yes",
            "1": "no",
        },
        "log": {
            "0": "yes",
            "1": "no",
        },
    }
})

refresh_lib(initial_state)

initial_state.add_notification("success", "Ready", "Face recognition app ready.")

from PIL import Image
from voyager import Index, Space
import numpy as np
import cv2
import imutils
import threading
import requests
import os
from tqdm import tqdm
# from multiprocessing.dummy import Pool
# pool = Pool(12)

def load_faces(app, db_path, res_x, res_y, num_dimensions):
    names = []
    embeddings = []
    database = {}

    for dirpath, dirnames, filenames in tqdm(os.walk(db_path)):
        for filename in filenames:
            if filename.endswith(".png") or filename.endswith(".jpg"):
                img_name = os.path.join(dirpath, filename)
                image = np.array(Image.open(img_name))[:, :, ::-1]
                # image = imutils.resize(image, width=res_x)
                faces = app.get(image) # app.get(image[:res_y,:])
                if len(faces) == 0: raise Exception(str([len(image), img_name]))
                if faces:
                    largest = get_largest_face(faces)
                    embedding = faces[largest].normed_embedding
                    embeddings.append(embedding)
                    dirname = os.path.basename(os.path.normpath(dirpath)).replace(".", "/")
                    if dirname not in database:
                        database[dirname] = {"latest": None}
                    database[dirname][filename] = len(names)
                    names.append((dirname, filename))
    index = build_index(embeddings, num_dimensions)
    return names, database, index

def build_index(embeddings, num_dimensions):
    index = Index(Space.Euclidean, num_dimensions=num_dimensions)
    if embeddings:
        index.add_items(np.array(embeddings))
    return index

def request_task(url, name, location):
    try:
        requests.post(url, json={"name": name, "location": location})
    except:
        pass

# def request_task_get(url):
#     try:
#         requests.get(url)
#     except:
#         pass

def send_data(url, name, location):
# def send_data(url):
    # pool.apply_async(requests.get, (url,))
    threading.Thread(target=request_task, args=(url, name, location)).start()
    # threading.Thread(target=request_task_get, args=(url,)).start()

def get_largest_face(faces):
    largest = 0
    max_area = (faces[0].bbox[2] - faces[0].bbox[0]) * (faces[0].bbox[3] - faces[0].bbox[1])
    for count in range(1, len(faces)):
        face = faces[count]
        area = (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
        if area > max_area:
            max_area = area
            largest = count
    return largest

def draw_boxes(img, faces):
    if faces:
        largest = get_largest_face(faces)
        for count in range(len(faces)):
            face = faces[count]
            if count != largest:
                cv2.rectangle(img, (int(face.bbox[0]), int(face.bbox[1])), (int(face.bbox[2]), int(face.bbox[3])), (255, 0, ), 2)
            else:
                cv2.rectangle(img, (int(face.bbox[0]), int(face.bbox[1])), (int(face.bbox[2]), int(face.bbox[3])), (0, 255, 0), 2)
    return img

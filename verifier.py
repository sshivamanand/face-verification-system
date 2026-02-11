import numpy as np
import os
from deepface import DeepFace


MODEL_NAME = "ArcFace"
DATABASE_DIR = "database"


def get_embedding(img_path):
    result = DeepFace.represent(
        img_path=img_path,
        model_name=MODEL_NAME,
        detector_backend="retinaface",
        align=True,
        normalization="ArcFace"
    )

    return np.array(result[0]["embedding"])


def cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (
        np.linalg.norm(emb1) * np.linalg.norm(emb2)
    )


def check_if_exists(test_embedding):

    if not os.path.exists(DATABASE_DIR):
        return 0, "DATABASE EMPTY"

    files = [f for f in os.listdir(DATABASE_DIR) if f.endswith(".npy")]

    if len(files) == 0:
        return 0, "DATABASE EMPTY"

    best_similarity = -1

    for file in files:

        emb_path = os.path.join(DATABASE_DIR, file)

        stored_embedding = np.load(emb_path)

        similarity = cosine_similarity(test_embedding, stored_embedding)

        if similarity > best_similarity:
            best_similarity = similarity

    similarity_percent = best_similarity * 100

    if similarity_percent >= 80:
        decision = "FACE ALREADY REGISTERED"

    elif similarity_percent >= 65:
        decision = "LIKELY MATCH"

    else:
        decision = "NO MATCH FOUND"

    return similarity_percent, decision

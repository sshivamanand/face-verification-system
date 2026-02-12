import numpy as np
from deepface import DeepFace
from db import get_connection

MODEL_NAME = "ArcFace"


def get_embedding(img_path):
    result = DeepFace.represent(
        img_path=img_path,
        model_name=MODEL_NAME,
        detector_backend="retinaface",
        align=True,
        normalization="ArcFace"
    )

    return np.array(result[0]["embedding"])


def search_face(query_embedding):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT name, embedding <=> %s AS distance
        FROM faces
        ORDER BY embedding <=> %s
        LIMIT 1;
        """,
        (query_embedding, query_embedding)
    )

    result = cur.fetchone()
    cur.close()
    conn.close()

    if result is None:
        return 0, "DATABASE EMPTY"

    name, distance = result
    similarity = (1 - distance) * 100

    if similarity >= 80:
        decision = f"MATCH FOUND: {name}"
    elif similarity >= 65:
        decision = f"POSSIBLE MATCH: {name}"
    else:
        decision = "NO MATCH FOUND"

    return similarity, decision

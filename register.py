import cv2
import os
import numpy as np
from verifier import get_embedding, cosine_similarity


DATABASE_DIR = "database"
THRESHOLD = 0.75   


def face_already_exists(new_embedding):
    if not os.path.exists(DATABASE_DIR):
        return False, 0

    best_similarity = -1

    for file in os.listdir(DATABASE_DIR):
        if file.endswith(".npy"):
            emb_path = os.path.join(DATABASE_DIR, file)
            stored_embedding = np.load(emb_path)
            similarity = cosine_similarity(new_embedding, stored_embedding)
            if similarity > best_similarity:
                best_similarity = similarity

    similarity_percent = best_similarity * 100

    if similarity_percent >= THRESHOLD * 100:
        return True, similarity_percent
    return False, similarity_percent


def register_face():
    os.makedirs(DATABASE_DIR, exist_ok=True)
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Press SPACE to register face")
    print("Press ESC to exit")

    while True:

        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        cv2.imshow("Register Face", frame)
        key = cv2.waitKey(1)

        if key == 32:
            temp_path = "temp.jpg"
            cv2.imwrite(temp_path, frame)
            print("Extracting embedding...")
            embedding = get_embedding(temp_path)
            os.remove(temp_path)
            exists, similarity = face_already_exists(embedding)

            if exists:
                print(f"\nFace already registered ({similarity:.2f}% similarity)")
                print("Registration cancelled")
            else:
                count = len([f for f in os.listdir(DATABASE_DIR) if f.endswith(".npy")])
                emb_path = os.path.join(DATABASE_DIR, f"emb_{count}.npy")
                np.save(emb_path, embedding)
                print(f"\nFace registered successfully as emb_{count}.npy")
            break

        elif key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    register_face()

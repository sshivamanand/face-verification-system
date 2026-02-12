import cv2
import os
from verifier import get_embedding, search_face
from db import get_connection


TEMP_IMAGE = "temp.jpg"


def save_face(name, embedding):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO faces (name, embedding) VALUES (%s, %s)",
        (name, embedding.tolist())
    )

    conn.commit()
    cur.close()
    conn.close()


def register_face():
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

        if key == 32:  # SPACE
            cv2.imwrite(TEMP_IMAGE, frame)
            print("\nExtracting embedding...")
            embedding = get_embedding(TEMP_IMAGE)
            print("Checking if face already exists...")
            similarity, decision = search_face(embedding)

            if similarity >= 80:
                print(f"\nFace already registered ({similarity:.2f}%)")

            else:
                name = input("\nEnter person name: ")
                save_face(name, embedding)
                print(f"\nFace registered successfully for: {name}")

            os.remove(TEMP_IMAGE)
            break

        elif key == 27:
            print("Cancelled")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    register_face()

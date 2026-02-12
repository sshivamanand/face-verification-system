import cv2
import os
from verifier import get_embedding, search_face

CAPTURED_IMAGE = "captured.jpg"


def capture_image(output_path):
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Webcam opened")
    print("Press SPACE to capture image")
    print("Press ESC to exit")

    captured = False

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        cv2.imshow("Face Verification", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # SPACE
            cv2.imwrite(output_path, frame)
            print("Image captured")
            captured = True
            break

        elif key == 27:  # ESC
            print("Cancelled")
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured


def main():
    success = capture_image(CAPTURED_IMAGE)
    if not success:
        return

    print("\nExtracting embedding..")
    embedding = get_embedding(CAPTURED_IMAGE)
    print("Searching in vector database..")
    similarity, decision = search_face(embedding)

    os.remove(CAPTURED_IMAGE)

    print("\nResult:")
    print("-" * 10)
    print(f"Similarity: {similarity:.2f}%")
    print(f"{decision}")


if __name__ == "__main__":
    main()

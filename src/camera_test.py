import cv2

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    print("Frame received:", ret)

    if not ret:
        break

    print("Showing window...")

    cv2.imshow("Camera Test", frame)

    key = cv2.waitKey(1)

    print("Key:", key)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
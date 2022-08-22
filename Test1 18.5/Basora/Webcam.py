import cv2
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, image = cap.read()

    cv2.imshow("original", image)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        cap.release()
        break
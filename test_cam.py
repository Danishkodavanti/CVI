import cv2

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
print("Opened:", cap.isOpened())
ret, frame = cap.read()
print("Read:", ret)
if ret:
    cv2.imshow("test", frame)
    cv2.waitKey(3000)
cap.release()
cv2.destroyAllWindows()
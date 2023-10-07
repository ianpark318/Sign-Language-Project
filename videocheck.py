import cv2 as cv


cam = cv.VideoCapture(0)

if cam.isOpened():
    while True:
        ret, frame = cam.read()
        if ret:
            cv.imshow('test_video', frame)

            # if cv.waitKey(1) != -1:
            #     break

            if cv.waitKey(1) == ord('s'):
                break
        else:
            break

cam.release()
cv.destroyAllWindows()
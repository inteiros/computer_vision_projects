import cv2

def initialize_camera(camera_index=0):
    video_cam = cv2.VideoCapture(camera_index)
    if not video_cam.isOpened():
        print("couldn't access the camera.")
        exit()
    return video_cam

def main():
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    video_cam = initialize_camera()
    print("press 'q' to exit.")

    button_pressed = False
    while not button_pressed:
        ret, frame = video_cam.read()

        if ret:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = face_classifier.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=2)

            for (x, y, w, h) in face_cascade:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            text = "number of detected faces = " + str(len(face_cascade))

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, text, (0, 30), font, 1, (255, 0, 0), 1)

            cv2.imshow("result", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                button_pressed = True
                break

    video_cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

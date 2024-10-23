import cv2

cap = cv2.VideoCapture(0)

skip = 0
face_data = []
dataset_path = None

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")

if face_cascade.empty():
    print("Error in loading the face cascade")
if eye_cascade.empty():
    print("Error in loading the eye cascade")
if smile_cascade.empty():
    print("Error in loading the smile cascade")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

    for face in faces:
        x, y, w, h = face

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        offset = 10
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        
        face_section_resized = cv2.resize(face_section, (200, 200))

        cv2.imshow("Cropped Face", face_section_resized)

        roi_gray = gray_frame[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 255), 2)

    cv2.imshow("Video frame", frame)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('a'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2 
import threading
from deepface import DeepFace
import pathlib

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "C:/Users/me095/OneDrive/Desktop/one file/WorkOnly/All Code/Python/AI/AI-FaceDetection/haarcascade_frontalface_default.xml"
clf = cv2.CascadeClassifier(str(cascade_path))

counter = 0
face_match = False

reference_img = cv2.imread("C:/Users/me095/OneDrive/Desktop/one file/WorkOnly/All Code/Python/AI/AI-FaceDetection/Dechthana.jpg")
reference_img2 = reference_img.copy()

def check_face(frame):
    global face_match
    try:
        if DeepFace.verify(frame, reference_img2)['verified']:
            face_match = True
        else:
            face_match = False
    except ValueError:
        face_match

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors=9,
        minSize = (100, 100),
        flags = cv2.CASCADE_SCALE_IMAGE
    ) 
    if counter % 30 == 0:
            try:
                threading.Thread(target=check_face,args=(frame.copy(),)).start()
            except ValueError:
                pass
    counter += 1
    if face_match:
        cv2.putText(frame,"MATCH!",(20,450),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    else:
        cv2.putText(frame,"NoT MATCH!",(20,450),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)
    for (x,y,width,hight) in faces:
        cv2.rectangle(frame,(x,y),(x+width, y+hight),(1000,1000,5),2)
    cv2.imshow("Faces",frame)
    if cv2.waitKey(1) == ord("q"):
        break
cv2.destroyAllWindows()
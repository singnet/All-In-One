from demo import load_model,get_layer,selective_search_demo
import cv2
import dlib
import numpy as np
import os
from keras.models import Model


def images_demo(model,images_dir,detector):
    for imgfile in os.listdir(images_dir):
        image = cv2.imread(os.path.join(images_dir,imgfile))
        print image.shape
        faces = detector(image)
        for i in range(len(faces)):
            face = faces[i]
            face_image = image[face.top():face.bottom(),face.left():face.right()]
            face_image = cv2.resize(face_image,(227,227))
            face_image = face_image.astype(np.float32)/255
            predictions = model.predict(face_image.reshape(-1,227,227,3))
            image = cv2.rectangle(image, (face.left(),face.top()), (face.right(),face.bottom()), (255,0,0),thickness=3)
            age_estimation = predictions[0][0]
            smile_detection = predictions[1][0]
            gender_probablity = predictions[2][0]

            age = str(int(age_estimation))
            smile = np.argmax(smile_detection)
            gender = np.argmax(gender_probablity)

            if(smile==0):
                smile = "False"
            else:
                smile = "True"
            if(gender == 0):
                gender= "Female"
            else:
                gender = "Male"

            cv2.putText(image, "Age: "+age, (face.left() + 10, face.top() + 10), cv2.FONT_HERSHEY_DUPLEX, 0.4,
                    (255,0,0))
            cv2.putText(image, "Smile: "+smile, (face.left() + 10, face.top() + 20), cv2.FONT_HERSHEY_DUPLEX, 0.4,(255,0,0))
            cv2.putText(image, "Gender: "+gender, (face.left() + 10, face.top() + 30), cv2.FONT_HERSHEY_DUPLEX, 0.4,
                    (255,0,0))
        cv2.imshow("image",image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def process_video(model, path,detector):
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        _,frame = cap.read()
        faces = detector(frame)
        for i in range(len(faces)):
            face = faces[i]
            face_image = frame[
                max(face.top(),0):min(face.bottom(),frame.shape[0]),
                max(face.left(),0):min(face.right(),frame.shape[1])]
            face_image = cv2.resize(face_image,(227,227))
            face_image = face_image.astype(np.float32)/255
            predictions = model.predict(face_image.reshape(-1,227,227,3))
            frame = cv2.rectangle(frame, (face.left(),face.top()), (face.right(),face.bottom()), (255,0,0),thickness=3)
            age_estimation = predictions[0][0]
            smile_detection = predictions[1][0]
            gender_probablity = predictions[2][0]

            age = str(int(age_estimation))
            smile = np.argmax(smile_detection)
            gender = np.argmax(gender_probablity)
            print gender_probablity

            if(smile==0):
                smile = "False"
            else:
                smile = "True"
            if gender == 0:
                gender= "Female"
            else:
                gender = "Male"


            cv2.putText(frame, "Age: "+age, (face.left() + 10, face.top() + 10), cv2.FONT_HERSHEY_DUPLEX, 0.4,
                    (255,0,0))
            cv2.putText(frame, "Smile: "+smile, (face.left() + 10, face.top() + 20), cv2.FONT_HERSHEY_DUPLEX, 0.4,(255,0,0))
            cv2.putText(frame, "Gender: "+gender, (face.left() + 10, face.top() + 30), cv2.FONT_HERSHEY_DUPLEX, 0.4,
                    (255,0,0))
        cv2.imshow("image",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()

def webcam_demo(model,detector):
    process_video(model,-1,detector)

def video_demo(model,video_path,detector):
    process_video(model,video_path,detector)

def main():
    selective_search_demo()

if __name__ == "__main__":
    main()

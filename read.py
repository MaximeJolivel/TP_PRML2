import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from Annoted_Image import Annoted_Image, BGR_SPACE, YCrCb_SPACE, FDDBAnnotationFileReader


def main():
    iter_reader = iter(FDDBAnnotationFileReader("FDDB-folds/FDDB-fold-07-ellipseList.txt"))
    AnnImage = next(iter_reader)

    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
    eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

    print(face_cascade)

    CvImage = cv.imread(AnnImage.path)
    gray = cv.cvtColor(CvImage, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv.rectangle(CvImage,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = CvImage[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    print(AnnImage)
    plt.imshow(CvImage[...,::-1])
    plt.show()


if __name__ == '__main__':
    main()

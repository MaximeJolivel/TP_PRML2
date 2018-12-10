import numpy as np
import cv2 as cv

BGR_SPACE = 0
YCrCb_SPACE = 1

def FDDBAnnotationFileReader(fddb_file):
    """ lit le fichier images genere par Ellipses_in_dir """
    f_read = open(fddb_file)
    lines = f_read.readlines()
    iter_lines = iter(lines)
    end_doc = False
    while end_doc == False:
        try:
            image_path = next(iter_lines)
            nb_faces = int(next(iter_lines))
            coord_faces = []
            for i in range(nb_faces):
                coord_faces.append(next(iter_lines))
            yield Annoted_Image(image_path, nb_faces, coord_faces)
        except StopIteration:
            end_doc = True

def cv_read_image(path, color_space = BGR_SPACE):
    image = cv.imread(path)
    if color_space == YCrCb_SPACE:
        image = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    return image

class Annoted_Image:
    """docstring for Annoted_Image
        description of an image with its number of faces and the coordinates """
    def __init__(self, path, nb_faces, coord_faces):
        self.path = path.strip()+".jpg"
        self.nb_faces = nb_faces
        cv_image = cv_read_image(self.path)
        self.height = cv_image.shape[0]
        self.width = cv_image.shape[1]
        self.coord_faces = []
        for i in range(self.nb_faces):
            coords_list = coord_faces[i].strip().split(" ")
            self.coord_faces.append([float(coords_list[j]) for j in range(5)])

    def __str__(self):
        ch_str = self.path + " has "+ str(self.nb_faces) + " faces ( width : "+str(self.width)+", height : "+str(self.height)+"):\n"
        for i in range(self.nb_faces):
            ch_str += str(self.coord_faces[i])
            ch_str += "\n"
        return ch_str

    def Gen_Mask(self):
        mask = np.zeros((self.height, self.width), np.uint8)
        for ellipse in self.coord_faces:
            center = (int(ellipse[3]), int(ellipse[4]))
            axes = (int(ellipse[0]), int(ellipse[1]))
            angle = 360*ellipse[2]/(2*3.14159)
            cv.ellipse(mask, center, axes, angle, 0.0, 360.0, color = 1, thickness = -1)
        return mask

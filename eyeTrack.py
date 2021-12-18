import cv2
import mediapipe as mp
import time
import numpy as np
class EyeTracking():
    def __init__(self, getCoordinates = True):
        self.getCoordinates = getCoordinates
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh()
        self.drawSpec = self.mpDraw.DrawingSpec([200, 256, 0], thickness=1, circle_radius=3)
    def findEyePoints(self, img):
        face_points = []
        getCoordinates = True
        t = (time.time() % 12) / 12.0
        min_x = 1000
        min_y = 1000
        max_x = 0
        max_y = 0
        y_bar = 0
        ret_x = []
        leye = [124, 189, 223, 230]
        reye = [417, 346 ,443, 253]
        ih, iw, ic = img.shape
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        if self.results.multi_face_landmarks:
            for face in self.results.multi_face_landmarks:
                #self.mpDraw.draw_landmarks(img, face, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec, self.drawSpec)
                if self.getCoordinates:
                    i = 0
                    for lm in face.landmark:
                        x, y, z = int(lm.x*iw), int(lm.y*ih), lm.z
                        '''
                        if i in leye:
                            cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (255, 255, 255), 1,
                                        cv2.LINE_AA)
                        min_x = min(min_x, x)
                        min_y = min(min_y, y)
                        max_x = max(max_x, x)
                        max_y = max(max_y, y)
                        '''
                        face_points.append([x, y, z])
                        #i += 1
                    '''
                    ret_x = []
                    for fce in face_points:
                        ret_x.append((fce[0] - face_points[0][0]) / (max_x - min_x))
                    '''

        #cv2.imshow("Image", img[min_y:max_y][min_x][max_x])

        min_yl = face_points[leye[2]][1]
        max_yl = face_points[leye[3]][1]
        min_xl = face_points[leye[0]][0]
        max_xl = face_points[leye[1]][0]

        min_yr = face_points[reye[2]][1]
        max_yr = face_points[reye[3]][1]
        min_xr = face_points[reye[0]][0]
        max_xr = face_points[reye[1]][0]

        min_y = min(min_yl, min_yr)
        max_y = max(max_yl, max_yr)
        min_x = min(min_xl, min_xr)
        max_x = max(max_xl, max_xr)


        both_eyes = img[min_y:max_y + 1, min_x:max_x + 1]
        left_eye = img[min_yl: max_yl + 1, min_xl:max_xl + 1]
        right_eye = img[min_yr:max_yr + 1, min_xr: max_xr + 1]
        '''
        fce = img.copy()
        for row in range(min_y, max_y):
            cur_row = []
            cur_rowl = []
            cur_rowr = []
            for i in range(min_x, max_x + 1):
                cur_cur_row = []
                cur_cur_rowl = []
                cur_cur_rowr = []
                for l in fce[row][i]:
                    cur_cur_row.append(l)
                    if i < max_xl:
                        cur_cur_rowl.append(l)
                    elif i > min_xr:
                        cur_cur_rowr.append(l)
                cur_row.append(cur_cur_row)
                if i < max_xl:
                    cur_rowl.append(cur_cur_rowl)
                elif i > min_xr:
                    cur_rowr.append(cur_cur_rowr)
            both_eyes.append(cur_row)
            if row > min_yl and row < max_yl:
                left_eye.append(cur_rowl)
            if row > min_yr and row < max_yr:
                right_eye.append(cur_rowr)
        both_eyes = np.array(both_eyes)
        left_eye = np.array(left_eye)
        right_eye = np.array(right_eye)
        '''
        #return img, ret_x
        return both_eyes, left_eye, right_eye



    #1 Draw general outline for user to put face in
    def findBoxSum(self, box):
        count = 0
        for x in box:
            for y in x:
                count+=y
        return count
    def getInstruction(self, t, center = 0):
        cap = cv2.VideoCapture(0)
        detector = EyeTracking()
        success, img = cap.read()
        img = cv2.flip(img, 1)
        # for row in range(len(left_eye) - ):
        #    for col in range(len(left_eye[row])):
        try:
            eye_points, left_eye, right_eye = detector.findEyePoints(img)
            left_eye_resized = cv2.resize(left_eye, (316, 180))
            right_eye_resized = cv2.resize(right_eye, (316, 180))
            gray_eyel = cv2.cvtColor(left_eye_resized, cv2.COLOR_BGR2GRAY)
            darkest_box = []
            dark_val = 255 * 2500
            # print(gray_eyel)

            for row in range(20, len(gray_eyel) - 80, 5):
                for column in range(40, len(gray_eyel[0]) - 75, 5):
                    cur_box = gray_eyel[row:row + 20, column:column + 40]
                    cur_val = self.findBoxSum(cur_box)
                    if cur_val < dark_val:
                        dark_val = cur_val
                        darkest_box = [row + 30, column + 50]
            # time.sleep(100)
            # exit()
            cv2.circle(gray_eyel, (darkest_box[1], darkest_box[0]), 5, 5)
            if t == 15:
                center = darkest_box[1]
                print('ESTABLISHED CENTER')
                return 0, center
            if t > 15:
                if darkest_box[1] > center + 5:
                    print('GOING RIGHT')
                    return 1, center
                elif darkest_box[1] < center - 5:
                    print('GOING LEFT')
                    return -1, center
                else:
                    print('STAYING CENTER')
                    return 0, center

            # both_eye_resized = cv2.resize(eye_points, (1280, 720))
            #cv2.imshow("Image", gray_eyel)
            cv2.waitKey(1)
        except:
            print('No eyes!')
            return -2, center
        return -3, center


def main():
    cap = cv2.VideoCapture(0)
    detector = EyeTracking()
    getCoordinates = True
    face_points = []
    i = 0
    img1 = []
    img2 = []
    center = 0
    t = 0
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        #for row in range(len(left_eye) - ):
        #    for col in range(len(left_eye[row])):
        #try:
        eye_points, left_eye, right_eye = detector.findEyePoints(img)
        left_eye_resized = cv2.resize(left_eye, (316, 180))
        right_eye_resized = cv2.resize(right_eye, (316, 180))
        gray_eyel = cv2.cvtColor(left_eye_resized, cv2.COLOR_BGR2GRAY)
        darkest_box = []
        dark_val = 255 * 2500
        #print(gray_eyel)

        for row in range(20, len(gray_eyel) - 80, 5):
            for column in range(40, len(gray_eyel[0]) - 75, 5):
                cur_box = gray_eyel[row:row+20, column:column+40]
                cur_val = detector.findBoxSum(cur_box)
                if cur_val < dark_val:
                    dark_val = cur_val
                    darkest_box = [row + 30, column + 50]
        #time.sleep(100)
        #exit()
        cv2.circle(gray_eyel, (darkest_box[1], darkest_box[0]), 5, 5)
        t += 1
        if t == 15:
            center = darkest_box[1]
        if t > 15:
            if darkest_box[1] > center + 5:
                print('RIGHT')
            elif darkest_box[1] < center - 5:
                print('LEFT')
            else:
                print('CENTER')

        #both_eye_resized = cv2.resize(eye_points, (1280, 720))
        cv2.imshow("Image", gray_eyel)
        cv2.waitKey(1)
        #except:
        #    print('No eyes!')



if __name__ == "__main__":
    main()
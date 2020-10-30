# CAMERA-BASED ROAD SIGN RECOGNITION 
 
# import libraries
from PyQt5 import QtCore, QtGui, QtWidgets
from keras.models import load_model
from PIL import Image
import numpy as np
import cv2

# initialize variables
classes = 43
classes1 = { 1: "Speed limit (20km/h)",
    2: "Speed limit (30km/h)",
    3: "Speed limit (50km/h)",
    4: "Speed limit (60km/h)",
    5: "Speed limit (70km/h)",
    6: "Speed limit (80km/h)",
    7: "End of speed limit (80km/h)",
    8: "Speed limit (100km/h)",
    9: "Speed limit (120km/h)",
    10: "No passing",
    11: "No passing veh over 3.5 tons",
    12: "Right-of-way at intersection",
    13: "Priority road",
    14: "Yield",
    15: "Stop",
    16: "No vehicles",
    17: "Veh > 3.5 tons prohibited",
    18: "No entry",
    19: "General caution",
    20: "Dangerous curve left",
    21: "Dangerous curve right",
    22: "Double curve",
    23: "Bumpy road",
    24: "Slippery road",
    25: "Road narrows on the right",
    26: "Road work",
    27: "Traffic signals",
    28: "Pedestrians",
    29: "Children crossing",
    30: "Bicycles crossing",
    31: "Beware of ice/snow",
    32: "Wild animals crossing",
    33: "End speed + passing limits",
    34: "Turn right ahead",
    35: "Turn left ahead",
    36: "Ahead only",
    37: "Go straight or right",
    38: "Go straight or left",
    39: "Keep right",
    40: "Keep left",
    41: "Roundabout mandatory",
    42: "End of no passing",
    43: "End no passing veh > 3.5 tons"}

cam = cv2.VideoCapture(0)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(697, 494)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(170, 0, 381, 91))
        font = QtGui.QFont()
        font.setFamily("Bebas Neue")
        font.setPointSize(38)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.openCamBtn = QtWidgets.QPushButton(self.centralwidget)
        self.openCamBtn.setGeometry(QtCore.QRect(190, 410, 101, 41))
        self.openCamBtn.setObjectName("openCamBtn")
        self.closeCamBtn = QtWidgets.QPushButton(self.centralwidget)
        self.closeCamBtn.setGeometry(QtCore.QRect(310, 410, 101, 41))
        self.closeCamBtn.setObjectName("closeCamBtn")
        self.classifyBtn = QtWidgets.QPushButton(self.centralwidget)
        self.classifyBtn.setGeometry(QtCore.QRect(430, 410, 101, 41))
        self.classifyBtn.setStyleSheet("")
        self.classifyBtn.setObjectName("classifyBtn")
        self.imgLabel = QtWidgets.QLabel(self.centralwidget)
        self.imgLabel.setGeometry(QtCore.QRect(180, 80, 361, 251))
        self.imgLabel.setFrameShape(QtWidgets.QFrame.Box)
        self.imgLabel.setText("")
        self.imgLabel.setObjectName("imgLabel")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(250, 350, 241, 31))
        self.textEdit.setObjectName("textEdit")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 697, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.classifyBtn.clicked.connect(self.classifyFunction)

        self.openCamBtn.clicked.connect(self.openCamera)
        self.closeCamBtn.clicked.connect(self.stopCamera)
        self.classifyBtn.clicked.connect(self.classifyFunction)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Road Sign Recognition"))
        self.label.setText(_translate("MainWindow", "ROAD SIGN RECOGNITION"))
        self.openCamBtn.setText(_translate("MainWindow", "Open Camera"))
        self.closeCamBtn.setText(_translate("MainWindow", "Stop Camera"))
        self.classifyBtn.setText(_translate("MainWindow", "Classify"))

    # image classification function
    def classifyFunction(self):
        model = load_model('deep-learning/road-sign-recognition/model/model.h5')
        print("Loaded model from disk");
        path = 'deep-learning/road-sign-recognition/img/rsr.jpg'
        # print(path)
        test_image = Image.open(path)
        test_image = test_image.resize((30, 30)) # resize image
        test_image = np.expand_dims(test_image, axis=0) # expand dimensions
        test_image = np.array(test_image) # convert to array

        result = model.predict_classes(test_image)[0]
        sign = classes1[result + 1]
        print(sign)
        self.textEdit.setText(sign)

        if path: # load saved image from path
            print(path)
            self.file = path
            pixmap = QtGui.QPixmap(path) # setup pixmap with the provided image
            pixmap = pixmap.scaled(self.imgLabel.width(), self.imgLabel.height(), QtCore.Qt.KeepAspectRatio) # scale pixmap
            self.imgLabel.setPixmap(pixmap) # set the pixmap onto the label
            self.imgLabel.setAlignment(QtCore.Qt.AlignCenter) # align the label to center
    
    # open camera function
    def openCamera(self):
        while True: 
            _, frame = cam.read() # read camera
            cv2.imshow('road sign recognition', frame) # display window

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'): # save image when 's' is pressed
                cv2.imwrite('deep-learning/road-sign-recognition/img/rsr.jpg', frame)
            elif key == 27: # close window when 'esc' is pressed
                break
    
    # stop camera function
    def stopCamera(self):
        # release and destroy camera
        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

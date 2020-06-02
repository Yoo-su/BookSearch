from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import QtCore
import sys
import cv2

def openImgWin(img,bookInfo):
    app = QtWidgets.QApplication([])

    win=QtWidgets.QWidget()
    win.setStyleSheet('background:white')
    win.move(1050,180)
    vbox=QtWidgets.QVBoxLayout()  
    
    label = QtWidgets.QLabel()
    
    labelA=QtWidgets.QLabel(win)
    labelB=QtWidgets.QLabel(win)
    labelC=QtWidgets.QLabel(win)
    labelD=QtWidgets.QTextBrowser(win)
    labela=QtWidgets.QLabel(win)
    labelb=QtWidgets.QLabel(win)
    
    labelA.setText('제목: '+bookInfo['title'])
    labelA.setFont(QtGui.QFont("나눔고딕",12))
    
    labelB.setText('저자명: '+bookInfo['author'])
    labelB.setFont(QtGui.QFont("나눔고딕",12))
    
    labelC.setText('정가: '+bookInfo['price']+' / 할인가: '+bookInfo['discount'])
    labelC.setFont(QtGui.QFont("나눔고딕",12))
    
    labela.setText('출판사: '+bookInfo['publisher'])
    labela.setFont(QtGui.QFont("나눔고딕",12))
    
    labelb.setText('출간일: '+bookInfo['pubdate'])
    labelb.setFont(QtGui.QFont("나눔고딕",12))

    labelD.append(bookInfo['description'])
    labelD.setFont(QtGui.QFont("나눔고딕",12))
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    h,w,c = img.shape
    qImg = QtGui.QImage(img.data, w, h, w*c, QtGui.QImage.Format_RGB888)

    pixmap = QtGui.QPixmap.fromImage(qImg)

    label.setPixmap(pixmap)

    vbox.addWidget(label)
    vbox.addWidget(labelA)
    vbox.addWidget(labelB)
    vbox.addWidget(labelC)
    vbox.addWidget(labela)
    vbox.addWidget(labelb)
    vbox.addWidget(labelD)

    win.setLayout(vbox)
    

    win.show()

    sys.exit(app.exec_())
    


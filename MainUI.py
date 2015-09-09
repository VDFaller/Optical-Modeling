# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainUI.ui'
#
# Created: Tue Jul 21 15:25:01 2015
#      by: PyQt5 UI code generator 5.4
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1065, 664)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.NonGraph = QtWidgets.QGridLayout()
        self.NonGraph.setObjectName("NonGraph")
        self.plot = QtWidgets.QPushButton(self.centralwidget)
        self.plot.setObjectName("plot")
        self.NonGraph.addWidget(self.plot, 2, 0, 1, 2)
        self.CdTeThickness = QtWidgets.QLineEdit(self.centralwidget)
        self.CdTeThickness.setObjectName("CdTeThickness")
        self.NonGraph.addWidget(self.CdTeThickness, 1, 0, 1, 1)
        self.CdSeThickness = QtWidgets.QLineEdit(self.centralwidget)
        self.CdSeThickness.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.CdSeThickness.setObjectName("CdSeThickness")
        self.NonGraph.addWidget(self.CdSeThickness, 1, 1, 1, 1)
        self.label = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.NonGraph.addWidget(self.label, 0, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.NonGraph.addWidget(self.label_2, 0, 1, 1, 1)
        self.gridLayout.addLayout(self.NonGraph, 1, 0, 1, 1)
        self.GraphFrame = QtWidgets.QFrame(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.GraphFrame.sizePolicy().hasHeightForWidth())
        self.GraphFrame.setSizePolicy(sizePolicy)
        self.GraphFrame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.GraphFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.GraphFrame.setObjectName("GraphFrame")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.GraphFrame)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_3 = QtWidgets.QLabel(self.GraphFrame)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.verticalLayout.addWidget(self.label_3)
        self.gridLayout.addWidget(self.GraphFrame, 1, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1065, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.plot.setText(_translate("MainWindow", "Plot"))
        self.label.setText(_translate("MainWindow", "CdTe Thickness"))
        self.label_2.setText(_translate("MainWindow", "CdSe Thickness"))
        self.label_3.setText(_translate("MainWindow", "Graph"))


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThreadPool
from PyQt5.QtWidgets import QFileDialog

from QThread import Worker
from DistbutionCanvas import DistrbutionCanvas
import ProcessData
from MainCanvas import Canvas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar



class NavigationToolbar_edited(NavigationToolbar):
    # only display the buttons we need
    NavigationToolbar.toolitems = (
        ('Home', 'Reset original view', 'home', 'home'),
        ('Back', 'Back to previous view', 'back', 'back'),
        ('Forward', 'Forward to next view', 'forward', 'forward'),
        (None, None, None, None),
        (None, None, None, None),
        ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
        ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
        # ('Subplots', 'Configure subplots', 'subplots', 'configure_subplots'),
        ("Customize", "Edit axis, curve and image parameters", "qt4_editor_options", "edit_parameters"),
        (None, None, None, None),
        (None, None, None, None),
        ('Save', 'Save the figure', 'filesave', 'save_figure'),
    )

class Ui_MainWindow(object):
    def initialize(self,mainWindow):

        self.setGlobalVariables()
        self.setConstants()
        self.setupUi(mainWindow)
        self.setThreadPool()


    def setupUi(self, MainWindow):
        """initialize the user interface"""
        MainWindow.setObjectName("MainWindow")
        MainWindow.setFixedSize(1335, 879)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.Whole_Frame = QtWidgets.QFrame(self.centralwidget)
        self.Whole_Frame.setGeometry(QtCore.QRect(10, 10, 1311, 841))
        self.Whole_Frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.Whole_Frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.Whole_Frame.setObjectName("Whole_Frame")
        self.setUIElements()
        self.setupVisualizationCanvas()
        self.setupControlFrame()
        self.setUIControls()
        self.retranslateUi(MainWindow)
        self.tabWidget_content_properties.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def setUIElements(self):
        """Control panel label"""
        self.label_Control = QtWidgets.QLabel(self.Whole_Frame)
        self.label_Control.setGeometry(QtCore.QRect(1098, 0, 161, 21))

        font = QtGui.QFont()
        font.setFamily("Georgia")
        font.setPointSize(16)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)

        self.label_Control.setFont(font)
        self.label_Control.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_Control.setAlignment(QtCore.Qt.AlignCenter)
        self.label_Control.setObjectName("label_Control")

    def setupVisualizationCanvas(self):
        self.canvas = Canvas(self.Whole_Frame, 10, 9, 100)
        self.canvas.setGeometry(QtCore.QRect(0, 0, 1050, 820))
        self.nav = NavigationToolbar_edited(self.canvas, self.Whole_Frame, coordinates=True)
        self.nav.setGeometry(0, 0, 1060, 30)

    def setupControlFrame(self):
        self.control_frame = QtWidgets.QFrame(self.Whole_Frame)
        self.control_frame.setGeometry(QtCore.QRect(1076, 20, 221, 801))
        self.control_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.control_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.control_frame.setObjectName("control_frame")

        self.setupMapProperties()
        self.setDataProperties()
        self.setupContentProperties()
        self.setupMenuBar()
        MainWindow.setCentralWidget(self.centralwidget)

    def setupMenuBar(self):
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1335, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuHome = QtWidgets.QMenu(self.menubar)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.menuHome.setFont(font)
        self.menuHome.setObjectName("menuHome")
        self.menuGo_to = QtWidgets.QMenu(self.menubar)
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionForward = QtWidgets.QAction(MainWindow)
        self.actionForward.setObjectName("actionForward")
        self.actionBackward = QtWidgets.QAction(MainWindow)
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionOpen.triggered.connect(self.fileDialog)
        self.actionQuit = QtWidgets.QAction(MainWindow)
        self.actionQuit.setObjectName("actionQuit")
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.actionQuit)
        self.menubar.addAction(self.menuFile.menuAction())

    def setThreadPool(self):
        self.threadpool = QThreadPool()

    def fileDialog(self):
        dialog = QFileDialog()
        fileNames = QFileDialog.getOpenFileNames(dialog, "Open File")
        if (len(fileNames[0]) > 0):
            self.processFilesNames(fileNames[0])

    def setupContentProperties(self):
        comboboxfont = QtGui.QFont()
        comboboxfont.setPointSize(8)
        comboboxfont.setBold(False)
        comboboxfont.setWeight(50)
        self.groupBox_content_properties = QtWidgets.QGroupBox(self.control_frame)
        self.groupBox_content_properties.setGeometry(QtCore.QRect(10, 551, 210, 251))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox_content_properties.setFont(font)
        self.groupBox_content_properties.setAlignment(QtCore.Qt.AlignCenter)
        self.groupBox_content_properties.setObjectName("groupBox_content_properties")
        self.tabWidget_content_properties = QtWidgets.QTabWidget(self.groupBox_content_properties)
        self.tabWidget_content_properties.setGeometry(QtCore.QRect(0, 20, 210, 231))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.tabWidget_content_properties.setFont(font)
        self.tabWidget_content_properties.setAutoFillBackground(False)
        self.tabWidget_content_properties.setTabPosition(QtWidgets.QTabWidget.North)
        self.tabWidget_content_properties.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.tabWidget_content_properties.setObjectName("tabWidget_content_properties")
        self.Contour = QtWidgets.QWidget()
        self.Contour.setObjectName("Contour")
        self.groupBox_contours = QtWidgets.QGroupBox(self.Contour)
        self.groupBox_contours.setGeometry(QtCore.QRect(0, 6, 204, 91))
        self.groupBox_contours.setAutoFillBackground(False)
        self.groupBox_contours.setFlat(False)
        self.groupBox_contours.setCheckable(True)
        self.groupBox_contours.setObjectName("groupBox_contours")
        self.checkBox_fill_contours = QtWidgets.QCheckBox(self.groupBox_contours)
        self.checkBox_fill_contours.setGeometry(QtCore.QRect(160, 19, 37, 17))
        self.checkBox_fill_contours.setObjectName("checkBox_fill_contours")
        self.label_colormap_contours = QtWidgets.QLabel(self.groupBox_contours)
        self.label_colormap_contours.setGeometry(QtCore.QRect(5, 44, 47, 13))
        self.label_colormap_contours.setObjectName("label_colormap_contours")
        self.frame_opacity_contours = QtWidgets.QFrame(self.groupBox_contours)
        self.frame_opacity_contours.setGeometry(QtCore.QRect(63, 66, 136, 21))
        self.frame_opacity_contours.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_opacity_contours.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_opacity_contours.setObjectName("frame_opacity_contours")
        self.horizontalSlider_opacity_contours = QtWidgets.QSlider(self.frame_opacity_contours)
        self.horizontalSlider_opacity_contours.setGeometry(QtCore.QRect(3, 0, 103, 21))
        self.horizontalSlider_opacity_contours.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_opacity_contours.setObjectName("horizontalSlider_opacity_contours")

        self.label_opacity_contours_value = QtWidgets.QLabel(self.frame_opacity_contours)
        self.label_opacity_contours_value.setGeometry(QtCore.QRect(108, 0, 27, 20))
        self.label_opacity_contours_value.setText("")
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_opacity_contours_value.setFont(font)
        self.label_opacity_contours_value.setObjectName("label_opacity_contours_value")
        self.label_opacity_contours_value.setFrameShape(QtWidgets.QFrame.Box)

        self.label_opacity_contours = QtWidgets.QLabel(self.groupBox_contours)
        self.label_opacity_contours.setGeometry(QtCore.QRect(5, 69, 47, 13))
        self.label_opacity_contours.setObjectName("label_opacity_contours")
        self.label_fill_contours = QtWidgets.QLabel(self.groupBox_contours)
        self.label_fill_contours.setGeometry(QtCore.QRect(5, 19, 71, 16))
        self.label_fill_contours.setObjectName("label_fill_contours")
        self.comboBox_colormap_contours = QtWidgets.QComboBox(self.groupBox_contours)
        self.comboBox_colormap_contours.setGeometry(QtCore.QRect(63, 40, 136, 20))
        self.comboBox_colormap_contours.setIconSize(QtCore.QSize(75, 40))
        self.comboBox_colormap_contours.setObjectName("comboBox_colormap_contours")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/newPrefix/obspy-imaging-cm-1.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.comboBox_colormap_contours.addItem(icon1, "")
        self.groupBox_lines = QtWidgets.QGroupBox(self.Contour)
        self.groupBox_lines.setGeometry(QtCore.QRect(0, 101, 204, 101))
        self.groupBox_lines.setCheckable(True)
        self.groupBox_lines.setObjectName("groupBox_lines")
        self.frame_opacity_lines = QtWidgets.QFrame(self.groupBox_lines)
        self.frame_opacity_lines.setGeometry(QtCore.QRect(63, 43, 134, 21))
        self.frame_opacity_lines.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_opacity_lines.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_opacity_lines.setObjectName("frame_opacity_lines")
        self.horizontalSlider_opacity_lines = QtWidgets.QSlider(self.frame_opacity_lines)
        self.horizontalSlider_opacity_lines.setGeometry(QtCore.QRect(3, 0, 102, 21))
        self.horizontalSlider_opacity_lines.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_opacity_lines.setObjectName("horizontalSlider_opacity_lines")

        self.label_opacity_lines_value = QtWidgets.QLabel(self.frame_opacity_lines)
        self.label_opacity_lines_value.setGeometry(QtCore.QRect(106, 0, 27, 20))
        self.label_opacity_lines_value.setText("")
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_opacity_lines_value.setFont(font)
        self.label_opacity_lines_value.setObjectName("label_opacity_lines_value")
        self.label_opacity_lines_value.setFrameShape(QtWidgets.QFrame.Box)

        self.label_color_lines = QtWidgets.QLabel(self.groupBox_lines)
        self.label_color_lines.setGeometry(QtCore.QRect(5, 21, 47, 13))
        self.label_color_lines.setObjectName("label_color_lines")
        self.label_opacity = QtWidgets.QLabel(self.groupBox_lines)
        self.label_opacity.setGeometry(QtCore.QRect(5, 47, 47, 13))
        self.label_opacity.setObjectName("label_opacity")
        self.comboBox_color_lines = QtWidgets.QComboBox(self.groupBox_lines)
        self.comboBox_color_lines.setGeometry(QtCore.QRect(63, 18, 134, 18))
        self.comboBox_color_lines.setIconSize(QtCore.QSize(75, 40))
        self.comboBox_color_lines.setObjectName("comboBox_color_lines")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/newPrefix/index.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.comboBox_color_lines.addItem(icon2, "")
        self.frame_line_width = QtWidgets.QFrame(self.groupBox_lines)
        self.frame_line_width.setGeometry(QtCore.QRect(63, 70, 134, 21))
        self.frame_line_width.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_line_width.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_line_width.setObjectName("frame_line_width")
        self.horizontalSlider_line_width = QtWidgets.QSlider(self.frame_line_width)
        self.horizontalSlider_line_width.setGeometry(QtCore.QRect(3, 0, 102, 21))
        self.horizontalSlider_line_width.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_line_width.setObjectName("horizontalSlider_line_width")

        self.label_line_width_value = QtWidgets.QLabel(self.frame_line_width)
        self.label_line_width_value.setGeometry(QtCore.QRect(106, 0, 27, 20))
        self.label_line_width_value.setText("")
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_line_width_value.setFont(font)
        self.label_line_width_value.setObjectName("label_line_width_value")
        self.label_line_width_value.setFrameShape(QtWidgets.QFrame.Box)

        self.label_line_width = QtWidgets.QLabel(self.groupBox_lines)
        self.label_line_width.setGeometry(QtCore.QRect(5, 70, 51, 16))
        self.label_line_width.setObjectName("label_line_width")
        self.tabWidget_content_properties.addTab(self.Contour, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.groupBox_arrow = QtWidgets.QGroupBox(self.tab_2)
        self.groupBox_arrow.setGeometry(QtCore.QRect(0, 10, 204, 95))
        self.groupBox_arrow.setCheckable(True)
        self.groupBox_arrow.setObjectName("groupBox_arrow")
        self.label_arrow_scale = QtWidgets.QLabel(self.groupBox_arrow)
        self.label_arrow_scale.setGeometry(QtCore.QRect(5, 20, 68, 16))
        self.label_arrow_scale.setObjectName("label_arrow_scale")

        self.comboBox_arrow_scale = QtWidgets.QComboBox(self.groupBox_arrow)
        self.comboBox_arrow_scale.setGeometry(QtCore.QRect(105, 19, 91, 21))
        self.comboBox_arrow_scale.setObjectName("comboBox_arrow_scale")

        self.label_arrow_color = QtWidgets.QLabel(self.groupBox_arrow)
        self.label_arrow_color.setGeometry(QtCore.QRect(5, 71, 100, 16))
        self.label_arrow_color.setObjectName("label_arrow_color")
        self.comboBox_arrow_color = QtWidgets.QComboBox(self.groupBox_arrow)
        self.comboBox_arrow_color.setGeometry(QtCore.QRect(105, 70, 91, 21))
        self.comboBox_arrow_color.setIconSize(QtCore.QSize(75, 40))
        self.comboBox_arrow_color.setObjectName("comboBox_arrow_color")

        self.label_arrow_color_2 = QtWidgets.QLabel(self.groupBox_arrow)
        self.label_arrow_color_2.setGeometry(QtCore.QRect(5, 45, 100, 16))
        self.label_arrow_color_2.setObjectName("label_arrow_color_2")
        self.comboBox_arrow_color_2 = QtWidgets.QComboBox(self.groupBox_arrow)
        self.comboBox_arrow_color_2.setGeometry(QtCore.QRect(105, 44, 91, 21))
        self.comboBox_arrow_color_2.setIconSize(QtCore.QSize(75, 40))
        self.comboBox_arrow_color_2.setObjectName("comboBox_arrow_color_2")

        self.tabWidget_content_properties.addTab(self.tab_2, "")

        #COMPASS####################
        self.label_compass = QtWidgets.QLabel(self.Whole_Frame)
        self.label_compass.setGeometry(QtCore.QRect(600, 0, 90, 90))
        self.label_compass.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_compass.setFrameShadow(QtWidgets.QFrame.Raised)
        self.label_compass.setText("")
        self.label_compass.setPixmap(QtGui.QPixmap("C:/Users/Phoenix/Desktop/contourdiff-master/compass-design_1174-3.jpg"))
        self.label_compass.setScaledContents(True)
        self.label_compass.setWordWrap(False)
        self.label_compass.setObjectName("label_compass")
        self.widget = QtWidgets.QWidget(self.Whole_Frame)
        self.widget.setGeometry(QtCore.QRect(2, 802, 1061, 22))
        self.widget.setObjectName("widget")

    def setupMapProperties(self):
        comboboxfont = QtGui.QFont()
        comboboxfont.setPointSize(8)
        comboboxfont.setBold(False)
        comboboxfont.setWeight(50)
        line_edit_font = QtGui.QFont()
        line_edit_font.setPointSize(7)
        line_edit_font.setBold(False)
        self.groupBox_map_properties = QtWidgets.QGroupBox(self.control_frame)
        self.groupBox_map_properties.setGeometry(QtCore.QRect(10, 373, 210, 171))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)

        self.groupBox_map_properties.setFont(font)
        self.groupBox_map_properties.setAlignment(QtCore.Qt.AlignCenter)
        self.groupBox_map_properties.setObjectName("groupBox_map_properties")
        self.label_Quadtree = QtWidgets.QLabel(self.groupBox_map_properties)
        self.label_Quadtree.setGeometry(QtCore.QRect(6, 22, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)

        self.label_Quadtree.setFont(font)
        self.label_Quadtree.setObjectName("label_Quadtree")
        self.label_Direction = QtWidgets.QLabel(self.groupBox_map_properties)
        self.label_Direction.setGeometry(QtCore.QRect(4, 47, 51, 16))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)

        self.label_Direction.setFont(font)
        self.label_Direction.setObjectName("label_Direction")
        self.comboBox_Quadtree = QtWidgets.QComboBox(self.groupBox_map_properties)
        self.comboBox_Quadtree.setFont(comboboxfont)
        self.comboBox_Quadtree.setGeometry(QtCore.QRect(97, 21, 101, 18))
        self.comboBox_Quadtree.setFrame(True)
        self.comboBox_Quadtree.setObjectName("comboBox_Quadtree")
        self.label_content = QtWidgets.QLabel(self.groupBox_map_properties)
        self.label_content.setGeometry(QtCore.QRect(6, 74, 70, 13))

        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)

        self.label_content.setFont(font)
        self.label_content.setObjectName("label_content")
        self.checkBox_content_basemap = QtWidgets.QCheckBox(self.groupBox_map_properties)
        self.checkBox_content_basemap.setGeometry(QtCore.QRect(66, 73, 70, 17))

        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)

        self.checkBox_content_basemap.setFont(font)
        self.checkBox_content_basemap.setObjectName("checkBox_content_basemap")
        self.checkBox_content_vectors = QtWidgets.QCheckBox(self.groupBox_map_properties)
        self.checkBox_content_vectors.setGeometry(QtCore.QRect(137, 73, 61, 17))

        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)

        self.checkBox_content_vectors.setFont(font)
        self.checkBox_content_vectors.setObjectName("checkBox_content_vectors")
        self.checkBox_direction_positive = QtWidgets.QCheckBox(self.groupBox_map_properties)
        self.checkBox_direction_positive.setGeometry(QtCore.QRect(66, 48, 70, 17))

        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)

        self.checkBox_direction_positive.setFont(font)
        self.checkBox_direction_positive.setObjectName("checkBox_direction_positive")
        self.checkBox_direction_negative = QtWidgets.QCheckBox(self.groupBox_map_properties)
        self.checkBox_direction_negative.setGeometry(QtCore.QRect(137, 48, 71, 17))

        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)

        self.checkBox_direction_negative.setFont(font)
        self.checkBox_direction_negative.setObjectName("checkBox_direction_negative")
        self.label_magnitude = QtWidgets.QLabel(self.groupBox_map_properties)
        self.label_magnitude.setGeometry(QtCore.QRect(6, 118, 51, 16))

        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)

        self.label_magnitude.setFont(font)
        self.label_magnitude.setObjectName("label_magnitude")
        self.frame_magnitude = QtWidgets.QFrame(self.groupBox_map_properties)
        self.frame_magnitude.setGeometry(QtCore.QRect(66, 120, 130, 18))
        self.frame_magnitude.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_magnitude.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_magnitude.setObjectName("frame_magnitude")
        self.horizontalSlider_magnitude = QtWidgets.QSlider(self.frame_magnitude)
        self.horizontalSlider_magnitude.setGeometry(QtCore.QRect(2, 0, 95, 18))
        self.horizontalSlider_magnitude.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_magnitude.setObjectName("horizontalSlider_magnitude")

        self.label__magnitude_value = QtWidgets.QLabel(self.frame_magnitude)
        self.label__magnitude_value.setGeometry(QtCore.QRect(99, 0, 30, 17))
        self.label__magnitude_value.setText("")
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label__magnitude_value.setFont(font)
        self.label__magnitude_value.setObjectName("label__magnitude_value")
        self.label__magnitude_value.setFrameShape(QtWidgets.QFrame.Box)

        self.pushButton_mapproperties_apply = QtWidgets.QPushButton(self.groupBox_map_properties)
        self.pushButton_mapproperties_apply.setGeometry(QtCore.QRect(105, 145, 93, 21))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.pushButton_mapproperties_apply.setFont(font)
        self.pushButton_mapproperties_apply.setObjectName("pushButton_mapproperties_apply")
        self.pushButton_mapproperties_apply.clicked.connect(self.pushButton_mapproperties_apply_listener)
        self.checkBox_content_simplified_contourmap = QtWidgets.QCheckBox(self.groupBox_map_properties)
        self.checkBox_content_simplified_contourmap.setGeometry(QtCore.QRect(66, 94, 131, 17))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.checkBox_content_simplified_contourmap.setFont(font)
        self.checkBox_content_simplified_contourmap.setObjectName("checkBox_content_simplified_contourmap")
        self.groupBox_map_properties.setEnabled(False)
        self.groupBox_data_properties = QtWidgets.QGroupBox(self.control_frame)
        self.groupBox_data_properties.setGeometry(QtCore.QRect(10, 80, 210, 285))

    def setDataProperties(self):
        comboboxfont = QtGui.QFont()
        comboboxfont.setPointSize(8)
        comboboxfont.setBold(False)
        comboboxfont.setWeight(50)
        line_edit_font = QtGui.QFont()
        line_edit_font.setPointSize(7)
        line_edit_font.setBold(False)
        line_edit_font.setWeight(50)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox_data_properties.setFont(font)
        self.groupBox_data_properties.setAlignment(QtCore.Qt.AlignCenter)
        self.groupBox_data_properties.setObjectName("groupBox_data_properties")

        self.label_isoline1 = QtWidgets.QLabel(self.groupBox_data_properties)
        self.label_isoline1.setGeometry(QtCore.QRect(6, 182, 51, 16))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_isoline1.setFont(font)
        self.label_isoline1.setObjectName("label_isoline1")

        self.label_isoline2 = QtWidgets.QLabel(self.groupBox_data_properties)
        self.label_isoline2.setGeometry(QtCore.QRect(6, 206, 51, 16))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.label_isoline2.setFont(font)
        self.label_isoline2.setObjectName("label_isoline2")

        self.label_isoline3 = QtWidgets.QLabel(self.groupBox_data_properties)
        self.label_isoline3.setGeometry(QtCore.QRect(6, 230, 51, 16))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.label_isoline3.setFont(font)
        self.label_isoline3.setObjectName("label_isoline3")

        self.pushButton_distribution = QtWidgets.QPushButton(self.groupBox_data_properties)
        self.pushButton_distribution.setGeometry(QtCore.QRect(7, 48, 191, 22))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.pushButton_distribution.setFont(font)
        self.pushButton_distribution.setObjectName("pushButton_distribution")
        self.pushButton_distribution.clicked.connect(self.setPushButton_distrbution)

        self.frame_isoline1 = QtWidgets.QFrame(self.groupBox_data_properties)
        self.frame_isoline1.setGeometry(QtCore.QRect(57, 182, 140, 21))
        self.frame_isoline1.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_isoline1.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_isoline1.setObjectName("frame_isoline1")
        self.horizontalSlider_isoline1 = QtWidgets.QSlider(self.frame_isoline1)
        self.horizontalSlider_isoline1.setGeometry(QtCore.QRect(3, 0, 103, 21))
        self.horizontalSlider_isoline1.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_isoline1.setObjectName("horizontalSlider_isoline1")
        self.label_isoline1_value = QtWidgets.QLabel(self.frame_isoline1)
        self.label_isoline1_value.setGeometry(QtCore.QRect(109, 0, 30, 20))
        self.label_isoline1_value.setText("")
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_isoline1_value.setFont(font)
        self.label_isoline1_value.setObjectName("label_isoline1_value")
        self.label_isoline1_value.setFrameShape(QtWidgets.QFrame.Box)

        self.frame_isoline2 = QtWidgets.QFrame(self.groupBox_data_properties)
        self.frame_isoline2.setGeometry(QtCore.QRect(57, 206, 140, 21))
        self.frame_isoline2.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_isoline2.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_isoline2.setObjectName("frame_isoline2")

        self.horizontalSlider_isoline2 = QtWidgets.QSlider(self.frame_isoline2)
        self.horizontalSlider_isoline2.setGeometry(QtCore.QRect(3, 0, 103, 21))
        self.horizontalSlider_isoline2.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_isoline2.setObjectName("horizontalSlider_isoline2")

        self.label_isoline2_value = QtWidgets.QLabel(self.frame_isoline2)
        self.label_isoline2_value.setGeometry(QtCore.QRect(109, 0, 30, 20))
        self.label_isoline2_value.setText("")
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_isoline2_value.setFont(font)
        self.label_isoline2_value.setObjectName("label_isoline2_value")
        self.label_isoline2_value.setFrameShape(QtWidgets.QFrame.Box)

        self.frame_isoline3 = QtWidgets.QFrame(self.groupBox_data_properties)
        self.frame_isoline3.setGeometry(QtCore.QRect(57, 230, 140, 21))
        self.frame_isoline3.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_isoline3.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_isoline3.setObjectName("frame_isoline3")
        self.horizontalSlider_isoline3 = QtWidgets.QSlider(self.frame_isoline3)
        self.horizontalSlider_isoline3.setGeometry(QtCore.QRect(3, 0, 103, 21))
        self.horizontalSlider_isoline3.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_isoline3.setObjectName("horizontalSlider_isoline3")

        self.label_isoline3_value = QtWidgets.QLabel(self.frame_isoline3)
        self.label_isoline3_value.setGeometry(QtCore.QRect(109, 0, 30, 20))
        self.label_isoline3_value.setText("")
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_isoline3_value.setFont(font)
        self.label_isoline3_value.setObjectName("label_isoline3_value")
        self.label_isoline3_value.setFrameShape(QtWidgets.QFrame.Box)

        self.pushButton_dataproperties_reset = QtWidgets.QPushButton(self.groupBox_data_properties)
        self.pushButton_dataproperties_reset.setGeometry(QtCore.QRect(6, 259, 93, 21))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.pushButton_dataproperties_reset.setFont(font)
        self.pushButton_dataproperties_reset.setObjectName("pushButton_dataproperties_reset")
        self.pushButton_dataproperties_reset.clicked.connect(self.pushButton_dataproperties_reset_listener)
        self.pushButton_dataproperties_apply = QtWidgets.QPushButton(self.groupBox_data_properties)
        self.pushButton_dataproperties_apply.setGeometry(QtCore.QRect(105, 259, 93, 21))
        self.pushButton_dataproperties_apply.clicked.connect(self.pushButton_dataproperties_apply_listener)

        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.pushButton_dataproperties_apply.setFont(font)
        self.pushButton_dataproperties_apply.setObjectName("pushButton_dataproperties_apply")
        self.comboBox_column_name = QtWidgets.QComboBox(self.groupBox_data_properties)
        self.comboBox_column_name.setFont(comboboxfont)
        self.comboBox_column_name.setGeometry(QtCore.QRect(77, 20, 119, 20))
        self.comboBox_column_name.setObjectName("comboBox_column_name")
        self.label_column_name = QtWidgets.QLabel(self.groupBox_data_properties)
        self.label_column_name.setGeometry(QtCore.QRect(7, 20, 71, 16))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.label_column_name.setFont(font)
        self.label_column_name.setObjectName("label_column_name")
        self.groupBox_datadirectory = QtWidgets.QGroupBox(self.control_frame)
        self.groupBox_datadirectory.setGeometry(QtCore.QRect(10, 3, 210, 71))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox_datadirectory.setFont(font)
        self.groupBox_datadirectory.setAlignment(QtCore.Qt.AlignCenter)
        self.groupBox_datadirectory.setObjectName("groupBox_datadirectory")

        self.pushButton_obs_dir = QtWidgets.QPushButton(self.groupBox_datadirectory)
        self.pushButton_obs_dir.setGeometry(QtCore.QRect(158, 41, 42, 20))
        self.pushButton_obs_dir.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("C:/Users/Phoenix/Desktop/contourdiff-master/file.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)

        self.pushButton_obs_dir.setIcon(icon)
        self.pushButton_obs_dir.setObjectName("pushButton_obs_dir")
        self.pushButton_obs_dir.clicked.connect(self.fileDialog)

        self.label_obs_dir = QtWidgets.QLabel(self.groupBox_datadirectory)
        self.label_obs_dir.setGeometry(QtCore.QRect(6, 42, 153, 18))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.label_obs_dir.setFont(font)
        self.label_obs_dir.setFrameShape(QtWidgets.QFrame.Box)
        self.label_obs_dir.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_obs_dir.setLineWidth(1)
        self.label_obs_dir.setMidLineWidth(0)
        self.label_obs_dir.setObjectName("label_obs_dir")
        self.label_currentfile = QtWidgets.QLabel(self.groupBox_datadirectory)
        self.label_currentfile.setGeometry(QtCore.QRect(6, 20, 193, 18))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.label_currentfile.setFont(font)
        self.label_currentfile.setFrameShape(QtWidgets.QFrame.Box)
        self.label_currentfile.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_currentfile.setLineWidth(1)
        self.label_currentfile.setMidLineWidth(0)
        self.label_currentfile.setObjectName("label_currentfile")
        self.dist_canvas = DistrbutionCanvas(self.groupBox_data_properties)
        self.dist_canvas.setGeometry(QtCore.QRect(7, 75, 191, 101))

    def setConstants(self):
        self.quadtree_depth = np.arange(0, 1000, 10).astype('str').tolist()[1:]
        # self.combox_color_map_values = ['Colormap 2','Colormap 3','Colormap 4','Colormap 5','Colormap 6']
        self.combox_color_map_values = ['Colormap 1', 'Colormap 2', 'Colormap 3', 'Colormap 4', 'Colormap 5', 'Colormap 6']
        self.comboBox_color_lines_values = ['copper','binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat' ]
        # self.comboBox_arrow_scale_values = ['Linear', 'Exponential', 'Logarithmic', 'Normalized']
        self.comboBox_arrow_scale_values = ['Normalized', 'Exponential', 'Logarithmic', 'Linear']
        self.comboBox_color_vector_high2low_values = ['Black', 'Blue', 'Green', 'Red']
        self.comboBox_color_vector_low2high_values = ['Blue', 'Black', 'Green', 'Red']

    def setGlobalVariables(self):
        self.filenameToPathDict = {}
        self.column_list = list()
        self.data = pd.DataFrame()
        self.weighted_graph = pd.DataFrame()
        self.filtered_graph = pd.DataFrame()

    def setUIControls(self):
        self.setIsoLineSlider1()
        self.setIsoLineSlider2()
        self.setIsoLineSlider3()
        self.setCombox_Quadtree()
        self.checkedBoxes_init_()
        self.magnitudeValueChanged()
        self.setColorMapCombobox()
        self.setHorizontalSliderOpacityContours()
        self.setLineColor()
        self.setContourLineOppacity()
        self.setLineWidthSlider()

    def processUserAction(self):pass


################### STARTING POINT
    def processSelectedData(self):
        print('processSD')
        start_time_contourmap = time.time()

        file_list = list(self.filenameToPathDict.values())
        self.data = ProcessData.importData(file_list[-1])
        # self.data = ProcessData.importData(file_list[0])
        column_index = self.comboBox_column_name.currentIndex()
        column = self.column_list[column_index]

        # data normalizing
        self.data['levels'] = (self.data[column] - self.data[column].min()) / (self.data[column].max() - self.data[column].min())

        levels = self.getcutOffValues()
        # quantile_values = self.quantile_calculation()
        quantile_values = np.quantile(self.data['levels'][self.data['levels'] > 0], levels)
        if (quantile_values[0] == quantile_values[1]):
            quantile_values[1] = quantile_values[1] + .00000001
        elif (quantile_values[1] == quantile_values[2]):
            quantile_values[2] = quantile_values[2] + .00000001

        # creating contourmap with normalized data and three levels of user selected cutoff points
        # cntr_set = plt.contourf(np.array(self.data['levels']).reshape(699, 639), quantile_values, colors=['g', 'r', 'y'])
        cntr_set = plt.contour(np.array(self.data['levels']).reshape(699, 639), quantile_values, colors=['g', 'r', 'y'])

        # modeling the graph as dataframe from contourset
        cntr_data = ProcessData.modelTheGraph(cntr_set)
        self.weighted_graph = ProcessData.createWeightedGraph(cntr_data, file_list, column)
        quad_tree_depth_index = self.comboBox_Quadtree.currentIndex()
        quad_tree_depth = self.quadtree_depth[quad_tree_depth_index]
        print("Time1 %s seconds" % (time.time() - start_time_contourmap))


    def filter_weighted_graph(self):
        quad_tree_depth_index = self.comboBox_Quadtree.currentIndex()
        quad_tree_depth = self.quadtree_depth[quad_tree_depth_index]
        self.filtered_graph = ProcessData.filterBasedOnGrid(int(quad_tree_depth), self.weighted_graph)

    def select_column_weighted_graph(self,column):
        print('select')
        return self.weighted_graph[column]
    def drawOnMainCanvas(self):pass

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "ContourDiff"))
        self.label_Control.setText(_translate("MainWindow", "Control Panel"))
        self.groupBox_map_properties.setTitle(_translate("MainWindow", "Map Properties"))
        self.label_Quadtree.setText(_translate("MainWindow", "Quadtree Depth"))
        self.label_Direction.setText(_translate("MainWindow", " Direction"))
        self.label_content.setText(_translate("MainWindow", "Content"))
        self.checkBox_content_basemap.setText(_translate("MainWindow", "BaseMap"))
        self.checkBox_content_vectors.setText(_translate("MainWindow", "Vectors"))
        self.checkBox_direction_positive.setText(_translate("MainWindow", "High->Low"))
        self.checkBox_direction_negative.setText(_translate("MainWindow", "Low->High"))
        self.label_magnitude.setText(_translate("MainWindow", "Magnitude"))
        self.pushButton_mapproperties_apply.setText(_translate("MainWindow", "Apply"))
        self.checkBox_content_simplified_contourmap.setText(_translate("MainWindow", "Simplified ContourMap"))
        self.groupBox_data_properties.setTitle(_translate("MainWindow", "Data Properties"))
        self.label_isoline1.setText(_translate("MainWindow", "Isoline 1"))
        self.label_isoline2.setText(_translate("MainWindow", "Isoline 2"))
        self.label_isoline3.setText(_translate("MainWindow", "Isoline 3"))
        self.pushButton_distribution.setText(_translate("MainWindow", "Distribution"))
        self.pushButton_dataproperties_reset.setText(_translate("MainWindow", "Reset"))
        self.pushButton_dataproperties_apply.setText(_translate("MainWindow", "Apply"))
        self.label_column_name.setText(_translate("MainWindow", "Column Name"))
        self.groupBox_datadirectory.setTitle(_translate("MainWindow", "Data Directory"))
        self.label_obs_dir.setText(_translate("MainWindow", "Observation directory ..."))
        self.label_currentfile.setText(_translate("MainWindow", "Current file ..."))
        self.groupBox_content_properties.setTitle(_translate("MainWindow", "Content Properties"))
        self.groupBox_contours.setTitle(_translate("MainWindow", "Contours"))
        self.checkBox_fill_contours.setText(_translate("MainWindow", "Yes"))
        self.label_colormap_contours.setText(_translate("MainWindow", "Colormap"))
        self.label_opacity_contours.setText(_translate("MainWindow", "Opacity"))
        self.label_fill_contours.setText(_translate("MainWindow", "Fill Contours"))
        self.comboBox_colormap_contours.setCurrentText(_translate("MainWindow", "Colormap 1"))
        self.comboBox_colormap_contours.setItemText(0, _translate("MainWindow", "Colormap 1"))
        self.groupBox_lines.setTitle(_translate("MainWindow", "Lines"))
        self.label_color_lines.setText(_translate("MainWindow", "Colormap"))
        self.label_opacity.setText(_translate("MainWindow", "Opacity"))

        # self.comboBox_color_lines.setCurrentText(_translate("MainWindow", "copper"))
        # self.comboBox_color_lines.setItemText(0, _translate("MainWindow", "copper"))
        self.comboBox_color_lines.setCurrentText(_translate("MainWindow", "Colormap 1"))
        self.comboBox_color_lines.setItemText(0, _translate("MainWindow", "Colormap 1"))

        self.label_line_width.setText(_translate("MainWindow", "Line Width"))
        self.tabWidget_content_properties.setTabText(self.tabWidget_content_properties.indexOf(self.Contour),
                                                     _translate("MainWindow", "Contours"))
        self.groupBox_arrow.setTitle(_translate("MainWindow", "Arrow"))
        self.label_arrow_scale.setText(_translate("MainWindow", "Arrow Scale"))
        self.comboBox_arrow_scale.setCurrentText(_translate("MainWindow", ""))

        # self.comboBox_arrow_scale.addItems(["Linear", "Exponential", "Logarithmic", "Normalized"])
        self.comboBox_arrow_scale.addItems(["Normalized", "Exponential", "Logarithmic", "Linear"])

        self.comboBox_arrow_scale.currentIndexChanged.connect(self.arrowscale_change_listener)

        self.label_arrow_color.setText(_translate("MainWindow", "Color (Low to High)"))
        self.comboBox_arrow_color.setCurrentText(_translate("MainWindow", ""))

        self.comboBox_arrow_color.addItems(['Blue', 'Black', 'Green', 'Red'])
        self.comboBox_arrow_color.currentIndexChanged.connect(self.color_vector_low2high_change_listener)

        self.label_arrow_color_2.setText(_translate("MainWindow", "Color (High to Low)"))
        self.comboBox_arrow_color_2.setCurrentText(_translate("MainWindow", ""))
        self.comboBox_arrow_color_2.addItems(['Black', 'Blue', 'Green', 'Red'])
        self.comboBox_arrow_color_2.currentIndexChanged.connect(self.color_vector_high2low_change_listener)

        self.tabWidget_content_properties.setTabText(self.tabWidget_content_properties.indexOf(self.tab_2),
                                                     _translate("MainWindow", "Vector"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))
        self.actionOpen.setShortcut(_translate("MainWindow", "Ctrl+o"))
        self.actionQuit.setText(_translate("MainWindow", "Quit"))
        self.actionQuit.setShortcut(_translate("MainWindow", "Ctrl+q"))

        #--------------------Event Handler-------------------------------------------#

    def processFilesNames(self, fileNames):
        """Sort List based time line"""
        self.filenameToPathDict = {}
        fileList = list()
        for i in range(len(fileNames)):
            fileName = fileNames[i].split("/")
            fileList.append(fileName[len(fileName) - 1])
            self.filenameToPathDict[fileName[len(fileName) - 1]] = fileNames[i]
            self.label_currentfile.setText(list(self.filenameToPathDict.keys())[-1])
            observation_directory = list(self.filenameToPathDict.values())[0].split("/")[-3:-1]
            connector = "/"
            observation_directory = connector.join(observation_directory)
            self.label_obs_dir.setText(observation_directory)
        self.groupBox_map_properties.setEnabled(False)
        worker = Worker(self.setupValuesColumnNameComboBox)
        worker.signals.finished.connect(self.thread_complete)
        self.threadpool.start(worker)

    def setupValuesColumnNameComboBox(self):
        self.column_list = list()
        first_file = list(self.filenameToPathDict.values())[0]
        temp_df = ProcessData.importData(first_file)
        column_list = temp_df.columns.tolist()
        if 'time' in column_list:column_list.remove('time')
        if 'latitude' in column_list: column_list.remove('latitude')
        if 'longitude' in column_list:column_list.remove('longitude')
        if 'x' in column_list: column_list.remove('x')
        if 'y' in column_list: column_list.remove('y')
        self.column_list = column_list
        self.comboBox_column_name.clear()
        self.comboBox_column_name.addItems(self.column_list)

### ISOLINES#############

    def setIsoLineSlider1(self):
        self.horizontalSlider_isoline1.setMinimum(0)
        self.horizontalSlider_isoline1.setMaximum(100)
        self.horizontalSlider_isoline1.setTickInterval(1)
        self.horizontalSlider_isoline1.setValue(25)
        self.label_isoline1_value.setText(str(self.horizontalSlider_isoline1.value() / 100))
        self.horizontalSlider_isoline1.valueChanged.connect(self.setIsoLineSlider1Listener)

    def setIsoLineSlider1Listener(self):
        value = self.horizontalSlider_isoline1.value() / 100
        self.label_isoline1_value.setText(str(value))

    def setIsoLineSlider2(self):
        self.horizontalSlider_isoline2.setMinimum(0)
        self.horizontalSlider_isoline2.setMaximum(100)
        self.horizontalSlider_isoline2.setTickInterval(1)
        self.horizontalSlider_isoline2.setValue(50)
        self.label_isoline2_value.setText(str(self.horizontalSlider_isoline2.value() / 100))
        self.horizontalSlider_isoline2.valueChanged.connect(self.setIsoLineSlider2Listener)

    def setIsoLineSlider2Listener(self):
        value = self.horizontalSlider_isoline2.value() / 100
        self.label_isoline2_value.setText(str(value))

    def setIsoLineSlider3(self):
        self.horizontalSlider_isoline3.setMinimum(0)
        self.horizontalSlider_isoline3.setMaximum(100)
        self.horizontalSlider_isoline3.setTickInterval(1)
        self.horizontalSlider_isoline3.setValue(75)
        self.label_isoline3_value.setText(str(self.horizontalSlider_isoline3.value() / 100))
        self.horizontalSlider_isoline3.valueChanged.connect(self.setIsoLineSlider3Listener)

    def setIsoLineSlider3Listener(self):
        value = self.horizontalSlider_isoline3.value() / 100
        self.label_isoline3_value.setText(str(value))




    def setupDistCanvas(self):pass

    def getcutOffValues(self):
        cut_off_point_1 = self.horizontalSlider_isoline1.value() / 100
        cut_off_point_2 = self.horizontalSlider_isoline2.value() / 100
        cut_off_point_3 = self.horizontalSlider_isoline3.value() / 100
        return [cut_off_point_1, cut_off_point_2, cut_off_point_3]

    def quantile_calculation(self):
        file_list = list(self.filenameToPathDict.values())
        # self.data = ProcessData.importData(file_list[0])
        self.data = ProcessData.importData(file_list[-1])
        column_index = self.comboBox_column_name.currentIndex()
        column = self.column_list[column_index]

        self.data['levels'] = (self.data[column] - self.data[column].min()) / (self.data[column].max() - self.data[column].min())
        levels = self.getcutOffValues()

        quantile_values = np.quantile(self.data['levels'], levels)
        # if (quantile_values[0] == 0 and quantile_values[1] == 0 and quantile_values[2] == 0):
        #     quantile_values[1] = quantile_values[0] + .00000001
        #     quantile_values[2] = quantile_values[1] + .00000001
        # elif (quantile_values[0] == 1 and quantile_values[1] == 1 and quantile_values[2] == 1):
        #     quantile_values[1] = quantile_values[2] - 0.000001
        #     quantile_values[0] = quantile_values[1] - 0.000001
        # elif (quantile_values[0] == 0 and quantile_values[1] == 1 and quantile_values[2] == 1):
        #     quantile_values[1] = quantile_values[2] - 0.000001
        # elif (quantile_values[0] == 0 and quantile_values[1] == 0 and quantile_values[2] == 1):
        #     quantile_values[1] = quantile_values[0] + .00000001
        # elif (quantile_values[0] == quantile_values[1]):
        #     quantile_values[1] = quantile_values[0] + .00000001
        # elif (quantile_values[1] == quantile_values[2]):
        #     quantile_values[2] = quantile_values[1] + .00000001
        if (quantile_values[0] == quantile_values[1]):
            quantile_values[1] = quantile_values[1] + .00000001
        elif (quantile_values[1] == quantile_values[2]):
            quantile_values[2] = quantile_values[2] + .00000001

        return quantile_values

    def setPushButton_distrbution(self):
        if (self.filenameToPathDict):
            self.dist_canvas.clearPlt()
            # file = list(self.filenameToPathDict.values())[0]
            file = list(self.filenameToPathDict.values())[-1]
            column_list_index = self.comboBox_column_name.currentIndex()
            column = self.column_list[column_list_index]
            self.dist_canvas.dist_plot(file, column)
            self.dist_canvas.addVerticalLines(self.quantile_calculation()[0], self.quantile_calculation()[1],
                                              self.quantile_calculation()[2])
            self.groupBox_map_properties.setEnabled(False)

    def pushButton_dataproperties_reset_listener(self):
        self.setIsoLineSlider1()
        self.setIsoLineSlider2()
        self.setIsoLineSlider3()
        self.pushButton_dataproperties_apply_listener()

    def pushButton_dataproperties_apply_listener(self):
        self.dist_canvas.clearPlt()
        self.setPushButton_distrbution()
        worker = Worker(self.setHorizontalMagnitude)
        worker.signals.finished.connect(self.thread_complete)
        self.threadpool.start(worker)

    def thread_complete(self):
        print("THREAD COMPLETE!")

    def setCombox_Quadtree(self):
        self.comboBox_Quadtree.addItems(self.quadtree_depth)

    def checkedBoxes_init_(self):
        self.checkBox_content_vectors.setChecked(True)
        self.checkBox_content_basemap.setChecked(True)
        self.checkBox_direction_negative.setChecked(True)
        self.checkBox_direction_positive.setChecked(True)
        self.checkBox_fill_contours.setChecked(True)

    def setHorizontalMagnitude(self):
        self.processSelectedData()
        # magnitude1 = self.select_column_weighted_graph('mag_pos')
        # magnitude2 = self.select_column_weighted_graph('mag_neg')
        magnitude_HL = self.select_column_weighted_graph('mag_HL')
        magnitude_LH = self.select_column_weighted_graph('mag_LH')

        # magnitude_max = max(max(magnitude1), max(magnitude2))
        # magnitude_min = min(min(magnitude1), min(magnitude2))
        # self.horizontalSlider_magnitude.setMinimum(magnitude_min)
        # self.horizontalSlider_magnitude.setMaximum(magnitude_max)
        self.horizontalSlider_magnitude.setMinimum(max(0, min(int(magnitude_HL[np.isfinite(magnitude_HL)].min()),
                                                              int(magnitude_LH[np.isfinite(magnitude_LH)].min()))))
        self.horizontalSlider_magnitude.setMaximum(min(1000, max(int(magnitude_HL[np.isfinite(magnitude_HL)].max()),
                                                                 int(magnitude_LH[np.isfinite(magnitude_LH)].max()))))

        self.horizontalSlider_magnitude.setTickInterval(1)

        # self.horizontalSlider_magnitude.setValue(magnitude_min)
        self.horizontalSlider_magnitude.setValue(0)

        self.label__magnitude_value.setText(str(self.horizontalSlider_magnitude.value()))
        self.groupBox_map_properties.setEnabled(True)

    def magnitudeValueChanged(self):
        self.horizontalSlider_magnitude.valueChanged.connect(self.setHorizontalMagnitudeListener)

    def setHorizontalMagnitudeListener(self):
        value = self.horizontalSlider_magnitude.value() / 100
        self.label__magnitude_value.setText(str(value))

    def directionCheckBoxHandler(self):
        """direction flag for checkbox
        * 0 for both
        * 1 for positive vector only
        * 2 for negative vector only
        * -1 if none selected
        """
        flag = -1
        positive_vector = self.checkBox_direction_positive.isChecked()
        negative_vector = self.checkBox_direction_negative.isChecked()
        if (positive_vector and negative_vector):
            flag = 0

        elif (positive_vector):
            flag = 1

        elif negative_vector:
            flag = 2

        else:
            flag = -1
        return flag

    def contentCheckBoxHandler(self):
        """content flag for checkbox
        * 0 for both
        * 1 for basemap only
        * 2 for vector only
        * -1 if none selected
        """
        flag = -1
        contour = self.checkBox_content_basemap.isChecked()
        vector = self.checkBox_content_vectors.isChecked()
        if (contour and vector):
            flag = 0

        elif (contour):
            flag = 1

        elif vector:
            flag = 2

        else:
            flag = -1
        return flag

    def setMapProperties(self):pass

    def pushButton_mapproperties_apply_listener(self):
        if(self.filenameToPathDict):
            time2 = time.time()
            self.filter_weighted_graph()
            print('done filtering weighted graph in appUI')
            self.draw_diagram()
            print("Time 2: %s seconds" % (time.time() - time2))

    def draw_diagram(self):
        final_diagram = time.time()

        alpha = self.horizontalSlider_opacity_contours.value() / 100
        column_index = self.comboBox_column_name.currentIndex()
        column = self.column_list[column_index]
        magnitude_val = self.horizontalSlider_magnitude.value()
        cmap_index = self.comboBox_colormap_contours.currentIndex()
        cmap = self.combox_color_map_values[cmap_index - 1]
        cline_index = self.comboBox_color_lines.currentIndex()
        cline = self.comboBox_color_lines_values[cline_index - 1]
        arrowscale_index = self.comboBox_arrow_scale.currentIndex()
        arrowscale = self.comboBox_arrow_scale_values[arrowscale_index]
        cvector_high2low_index = self.comboBox_arrow_color_2.currentIndex()
        cvector_high2low = self.comboBox_color_vector_high2low_values[cvector_high2low_index]
        cvector_low2high_index = self.comboBox_arrow_color.currentIndex()
        cvector_low2high = self.comboBox_color_vector_low2high_values[cvector_low2high_index]
        line_alpha = self.horizontalSlider_opacity_lines.value() / 100
        line_width = self.horizontalSlider_line_width.value()/10

        # self.canvas.generate_images(self.filtered_graph,self.data,self.quantile_calculation(),column,alpha,self.directionCheckBoxHandler(),
        #                             self.contentCheckBoxHandler(),magnitude_val,cmap,cline,arrowscale,cvector_high2low,cvector_low2high,line_alpha,line_width)

        self.canvas.generate_images(self.filtered_graph, list(self.filenameToPathDict.values())[0], self.data,
                                    self.quantile_calculation(), column, alpha, self.directionCheckBoxHandler(),
                                    self.contentCheckBoxHandler(), magnitude_val,
                                    cmap, cline, arrowscale, cvector_high2low, cvector_low2high, line_alpha, line_width)

        print("For creating final diagram %s seconds" % (time.time() - final_diagram))

    def setColorMapCombobox(self):
        self.comboBox_colormap_contours.addItems(self.combox_color_map_values)
        self.comboBox_colormap_contours.currentIndexChanged.connect(self.colorcombox_change_listener)

    def colorcombox_change_listener(self):
        if(self.filenameToPathDict):
            worker = Worker(self.draw_diagram)
            worker.signals.finished.connect(self.thread_complete)
            self.threadpool.start(worker)

    def setArrowScaleComboBox(self):
        self.comboBox_arrow_scale.addItems(self.comboBox_arrow_scale_values)
        self.comboBox_arrow_scale.currentIndexChanged.connect(self.arrowscale_change_listener)

    def arrowscale_change_listener(self):
                if (self.filenameToPathDict):
                    worker = Worker(self.draw_diagram)
                    worker.signals.finished.connect(self.thread_complete)
                    self.threadpool.start(worker)

    def setVectorColorHigh2Low(self):
        self.comboBox_arrow_color_2.addItems(self.comboBox_color_vector_high2low_values)
        self.comboBox_arrow_color_2.currentIndexChanged.connect(self.color_vector_high2low_change_listener)

    def color_vector_high2low_change_listener(self):
        if (self.filenameToPathDict):
            worker = Worker(self.draw_diagram)
            worker.signals.finished.connect(self.thread_complete)
            self.threadpool.start(worker)

    def setVectorColorLow2High(self):
        self.comboBox_arrow_color.addItems(self.comboBox_color_vector_low2high_values)
        self.comboBox_arrow_color.currentIndexChanged.connect(self.color_vector_low2high_change_listener)

    def color_vector_low2high_change_listener(self):
        if (self.filenameToPathDict):
            worker = Worker(self.draw_diagram)
            worker.signals.finished.connect(self.thread_complete)
            self.threadpool.start(worker)

    def setHorizontalSliderOpacityContours(self):
        self.horizontalSlider_opacity_contours.setMinimum(0)
        self.horizontalSlider_opacity_contours.setMaximum(100)
        self.horizontalSlider_opacity_contours.setTickInterval(1)

        # self.horizontalSlider_opacity_contours.setValue(75)
        self.horizontalSlider_opacity_contours.setValue(100)

        self.label_opacity_contours_value.setText(str(self.horizontalSlider_opacity_contours.value() / 100))
        self.horizontalSlider_opacity_contours.valueChanged.connect(self.setHorizontalSliderValueChanged)
        self.horizontalSlider_opacity_contours.sliderReleased.connect(self.setHorizontalSliderListener)

    def setHorizontalSliderValueChanged(self):
        value = self.horizontalSlider_opacity_contours.value() / 100
        self.label_opacity_contours_value.setText(str(value))

    def setHorizontalSliderListener(self):
        value = self.horizontalSlider_opacity_contours.value() / 100
        worker = Worker(self.draw_diagram)
        worker.signals.finished.connect(self.thread_complete)
        self.threadpool.start(worker)

    def setLineColor(self):
        self.comboBox_color_lines.addItems(self.comboBox_color_lines_values)
        self.comboBox_color_lines.currentIndexChanged.connect(self.color_line_combox_change_listener)

    def color_line_combox_change_listener(self):
        if(self.filenameToPathDict):
            worker = Worker(self.draw_diagram)
            worker.signals.finished.connect(self.thread_complete)
            self.threadpool.start(worker)

    def setContourLineOppacity(self):
        self.horizontalSlider_opacity_lines.setMinimum(0)
        self.horizontalSlider_opacity_lines.setMaximum(100)
        self.horizontalSlider_opacity_lines.setTickInterval(1)
        self.horizontalSlider_opacity_lines.setValue(75)

        self.label_opacity_lines_value.setText(str(self.horizontalSlider_opacity_lines.value() / 100))
        self.horizontalSlider_opacity_lines.valueChanged.connect(self.setContourLineOpacityChanged)
        self.horizontalSlider_opacity_lines.sliderReleased.connect(self.setContourLineOpacitySliderReleased)

    def setContourLineOpacityChanged(self):
        value = self.horizontalSlider_opacity_lines.value() / 100
        self.label_opacity_lines_value.setText(str(value))

    def setContourLineOpacitySliderReleased(self):
        if (self.filenameToPathDict):
            worker = Worker(self.draw_diagram)
            worker.signals.finished.connect(self.thread_complete)
            self.threadpool.start(worker)

    def setLineWidthSlider(self):
        self.horizontalSlider_line_width.setMinimum(0)
        self.horizontalSlider_line_width.setMaximum(30)
        self.horizontalSlider_line_width.setTickInterval(1)

        # self.horizontalSlider_line_width.setValue(15)
        self.horizontalSlider_line_width.setValue(5)

        self.label_line_width_value.setText(str(self.horizontalSlider_line_width.value()/10))
        self.horizontalSlider_line_width.valueChanged.connect(self.setContourLineWidthChanged)
        self.horizontalSlider_line_width.sliderReleased.connect(self.contourLineWidthSliderReleased)

    def setContourLineWidthChanged(self):
        value = self.horizontalSlider_line_width.value()/10
        self.label_line_width_value.setText(str(value))

    def contourLineWidthSliderReleased(self):
        if (self.filenameToPathDict):
            worker = Worker(self.draw_diagram)
            worker.signals.finished.connect(self.thread_complete)
            self.threadpool.start(worker)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.initialize(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())




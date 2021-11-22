# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 14:11:26 2020

@author: UTILISATEUR
"""
import sys
from PyQt5 import QtWidgets, QtCore
import numpy as np

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from numba import njit
import time

@njit(fastmath = True)
def hsv_to_rgb(h, s, v):
    if s == 0.0: return (v, v, v)
    i = int(h*6.)
    f = (h*6.)-i; p,q,t = v*(1.-s), v*(1.-s*f), v*(1.-s*(1.-f)); i%=6
    if i == 0: return (v*255, t*255, p*255)
    if i == 1: return (q*255, v*255, p*255)
    if i == 2: return (p*255, v*255, t*255)
    if i == 3: return (p*255, q*255, v*255)
    if i == 4: return (t*255, p*255, v*255)
    if i == 5: return (v*255, p*255, q*255)

@njit(fastmath = True)
def fastMendel(x,y):
    p = np.sqrt((x - 1/4)**2 + y*y)
    return x < p - 2*p*p + 1/4 or (x+1)**2 + y*y < 1/16

@njit(fastmath = True)
def make_mandelbrot(width, height, max_iterations, coeflog, slopex, Ox, slopey, Oy):
    result = np.zeros((height, width, 3))

    for iy in np.arange(height):
        for ix in np.arange(width):

            x0 = ix*slopex/width + Ox
            y0 = iy*slopey/height + Oy

            if fastMendel(x0,y0):
                color = max_iterations
            else:
                x = 0.0
                y = 0.0
                for iteration in range(max_iterations):
                    x_new = x*x - y*y + x0
                    y = 2*x*y + y0
                    x = x_new

                    if x*x + y*y > 4.0:
                        color = iteration + 1 - coeflog*(np.log2(np.log2(x*x + y*y)))
                        break
                else:
                    color = max_iterations
            hue = color / max_iterations
            value = 1 if color < max_iterations else 0
            result[iy, ix] = hsv_to_rgb(hue, 1, value)

    return result
class MainWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        # a figure instance to plot on
        self.setGeometry(75,25,1800,1000)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.Ax = self.figure.add_subplot(111)

        self.button = QtWidgets.QPushButton('False')
        self.reset = QtWidgets.QPushButton('Reset')
        self.highRes = False
        self.baseResW = QtWidgets.QLineEdit("600")
        self.baseResH = QtWidgets.QLineEdit("400")
        self.iterMax = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.iterMax.setMaximum(10000)
        self.iterMax.setMinimum(40)
        self.iterMaxLabel = QtWidgets.QLabel(str(self.iterMax.value()))
        self.coeflog = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.coeflog.setMaximum(100)
        self.coeflog.setTickInterval(500)
        self.coeflog.setMinimum(1)
        self.coeflogLabel = QtWidgets.QLabel(str(self.coeflog.value()/10))
        self.interpolation = QtWidgets.QComboBox()
        self.interpolation.addItem("None")
        self.interpolation.addItem("bilinear")

        #Connects
        self.canvas.mpl_connect('scroll_event', self.onScroll)
        self.iterMax.valueChanged.connect(self.plot)
        self.coeflog.valueChanged.connect(self.plot)
        self.button.clicked.connect(self.onhighRes)
        self.reset.clicked.connect(self.onReset)
        self.interpolation.currentIndexChanged.connect(self.plot)
        self.groupBox = QtWidgets.QGroupBox("Settings")
        self.groupBox.setMinimumWidth(220)
        self.groupBox.setMaximumWidth(220)
        boxLayout = QtWidgets.QVBoxLayout()

        layoutreset = self.creatPara("Reset :",self.reset)
        layoutW = self.creatPara("Base W :",self.baseResW)
        layoutH = self.creatPara("Base H :",self.baseResH)
        layout1 = self.creatPara("High resolution :",self.button)
        layout2 = self.creatPara("Iter max :",self.iterMax)
        layout2.addWidget(self.iterMaxLabel)
        layout3 = self.creatPara("coef log :",self.coeflog)
        layout4 = self.creatPara("interpolation :",self.interpolation)
        layout3.addWidget(self.coeflogLabel)
        boxLayout.addLayout(layoutreset)
        boxLayout.addLayout(layoutW)
        boxLayout.addLayout(layoutH)
        boxLayout.addLayout(layout1)
        boxLayout.addLayout(layout2)
        boxLayout.addLayout(layout3)
        boxLayout.addLayout(layout4)
        verticalSpacer = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        boxLayout.addItem(verticalSpacer)
        self.groupBox.setLayout(boxLayout)
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.groupBox)
        layout.addWidget(self.canvas)

        self.setLayout(layout)
        self.img = None

        self.xlim = [-2,1]
        self.ylim = [-1,1]
        self.zoom = 1
        self.slopex, self.Ox = self.xlim[1] - self.xlim[0], self.xlim[0]
        self.slopey, self.Oy = self.ylim[1] - self.ylim[0], self.ylim[0]
        self.plot()

    def plot(self):
        t = time.perf_counter()
        maxIter = self.iterMax.value()
        self.iterMaxLabel.setText(str(maxIter))
        coeflog = self.coeflog.value()/10
        self.coeflogLabel.setText(str(coeflog))
        w,h = int(self.baseResW.text()),int(self.baseResH.text())
        interpolation = self.interpolation.currentText()
        if not self.highRes:
            mandelbrot = make_mandelbrot(w, h, maxIter,coeflog, self.slopex, self.Ox, self.slopey, self.Oy).astype(np.uint8)
        else:
             mandelbrot = make_mandelbrot(3000, 2000, maxIter,coeflog,self.slopex, self.Ox, self.slopey, self.Oy).astype(np.uint8)
        if self.img is None:
            self.img = self.Ax.imshow(mandelbrot, interpolation= interpolation )
        else:
            self.img.set_data(mandelbrot)
            self.img.set_interpolation(interpolation)
        self.canvas.draw()
        temps = (time.perf_counter() - t)*1000
        print("zoom :", self.zoom, "| time :",round(temps,3),"ms")
        if temps > 150:
            self.baseResW.setText(str(w - 1)),
            self.baseResH.setText(str(h -1))

    def onhighRes(self):
        self.highRes = not self.highRes
        self.button.setText(str(self.highRes))
        self.plot()

    def onReset(self):
        self.xlim = [-2,1]
        self.ylim = [-1,1]
        self.slopex, self.Ox = self.xlim[1] - self.xlim[0], self.xlim[0]
        self.slopey, self.Oy = self.ylim[1] - self.ylim[0], self.ylim[0]

    def creatPara(self,name, widget):
        layout =  QtWidgets.QHBoxLayout()
        layout.addWidget(QtWidgets.QLabel(name))
        layout.addWidget(widget)
        return layout

    def onScroll(self, event):
          self.highRes = False
          self.button.setText("False")
          xdata = event.xdata
          ydata = event.ydata
          xdata = xdata*self.slopex/600 + self.Ox
          ydata = ydata*self.slopey/400 + self.Oy

          cur_xlim = self.xlim
          cur_ylim = self.ylim

          base_scale = 1.5
          if event.button == 'up':
            scale_factor = 1 / base_scale
            self.iterMax.setValue(self.iterMax.value()+ 10)
          elif event.button == 'down':
            self.iterMax.setValue(self.iterMax.value() - 10)
            scale_factor = base_scale
          else:
            scale_factor = 1

          self.xlim,self.ylim = scrollCompute(cur_xlim,cur_ylim,xdata,ydata,scale_factor)
          self.zoom = 3/(self.xlim[1] - self.xlim [0])


          self.slopex, self.Ox = self.xlim[1] - self.xlim[0], self.xlim[0]
          self.slopey, self.Oy = self.ylim[1] - self.ylim[0], self.ylim[0]
          self.plot()

    def getTxt(self):
        path = "D:/JLP/Python/VSC/MendelbrotTxt.txt"
        with open(path) as f:
            return np.array([line.rstrip().split(",") for line in f])

@njit(fastmath = True)
def scrollCompute(cur_xlim,cur_ylim,xdata,ydata,scale_factor):
    new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
    new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

    relx = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
    rely = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])

    return [xdata - new_width * (1-relx), xdata + new_width * (relx)],[ydata - new_height * (1-rely), ydata + new_height * (rely)]
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    main = MainWindow()
    main.show()

    sys.exit(app.exec_())
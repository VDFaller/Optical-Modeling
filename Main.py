'''
Demo to show use of the engineering Formatter.
'''
from PyQt5 import QtGui, QtWidgets, QtCore
import sys
import random
from numpy import arange, sin, pi
import MainUI
import matplotlib.pyplot as plt

from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg, NavigationToolbar2QT)
from matplotlib.figure import Figure

import xlrd
import tmm
from numpy.core.numeric import inf


class MPlibWidget(QtWidgets.QWidget):
    def __init__(self, parent = None):
        super(MPlibWidget, self).__init__(parent)
        
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setParent(self)
        
        self.mpl_toolbar = NavigationToolbar2QT(self.canvas, self)
        
        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        self.axis = self.figure.add_subplot(111)
        self.axis.hold(False)

        self.compute_initial_figure()
        
        self.layoutVertical = QtWidgets.QVBoxLayout(self)
        self.layoutVertical.addWidget(self.canvas)
        self.layoutVertical.addWidget(self.mpl_toolbar)
        
    def on_key_press(self, event):
        print('you pressed', event.key)
        # implement the default mpl key press events described at
        # http://matplotlib.org/users/navigation_toolbar.html#navigation-keyboard-shortcuts
        key_press_handler(event, self.canvas, self.mpl_toolbar)    
        
    def compute_initial_figure(self):
        pass 
        
        
class MplCanvas(MPlibWidget):
    """Simple canvas with a sine plot."""
    def compute_initial_figure(self):
        t = arange(0.0, 3.0, 0.01)
        s = sin(2*pi*t)
        self.axis.plot(t, s) 
    
    def update_figure(self):
        # Build a list of 4 random integers between 0 and 10 (both inclusive)
        l = [random.randint(0, 10) for i in range(4)]

        self.axis.plot([0, 1, 2, 3], l, 'r')
        self.canvas.draw()


class MW(QtWidgets.QMainWindow, MainUI.Ui_MainWindow):
    def __init__(self, parent = None):
        super(MW, self).__init__(parent)
        self.setupUi(self)
        
        self.static = MplCanvas(self.GraphFrame)
        
        self.verticalLayout.addWidget(self.static)
        self.plot.clicked.connect(self.plot_clicked)
        book = xlrd.open_workbook('H:/Perrysburg Users/VFaller/Public/Tasks/VF059 - Optical Modeling/modeling.xlsx')
        self.sh = book.sheet_by_index(0)
        self.wavelength = self.sh.col_values(0, 1)
    
    def get_column(self, header):
        for col_index in range(self.sh.ncols):
            if self.sh.cell(0, col_index).value == header:
                return(col_index)
    
    def plot_clicked(self):
        
        layers = [15, 14, 13, 12]
        layer_map = [map(complex, self.sh.col_values(i, 1)) for i in layers]       
        thkCdTe = int(self.CdTeThickness.text())
        thkCdSe = int(self.CdSeThickness.text())
        
        d_list = [inf, thkCdSe, thkCdTe, inf]
        theta0 = 0
        n_list = zip(*layer_map)
        
        R = [tmm.unpolarized_RT(next(n_list), d_list, theta0, wv)['R'] for wv in self.wavelength]
        
        self.static.axis.plot(self.wavelength, R)
        self.static.canvas.draw()
        
    
        
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    form = MW()
    form.show()
    app.exec_()



# fig, ax = plt.subplots()
# ax.set_xscale('log')
# formatter = EngFormatter(unit='Hz', places=1)
# ax.xaxis.set_major_formatter(formatter)
# 
# xs = np.logspace(1, 9, 100)
# ys = (0.8 + 0.4 * np.random.uniform(size=100)) * np.log10(xs)**2
# ax.plot(xs, ys)
# 
# plt.plot()
"""
UI Program to create an optical modeler
"""
import sys
import random
from math import floor, ceil

from PyQt5 import QtWidgets
import pandas as pd
from numpy import arange, sin, pi, interp
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")  # required currently because matplotlib uses PyQt4
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg, NavigationToolbar2QT)
from matplotlib.figure import Figure
# import xlrd
import tmm_core as tmm

from numpy.core.numeric import inf

import MainUI

print("boop")


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
        self.low_wavelength = 300
        self.high_wavelength = 1700
        self.increment = .2

        self.static = MplCanvas(self.GraphFrame)

        self.verticalLayout.addWidget(self.static)
        self.plot.clicked.connect(self.plot_clicked)
        # book = xlrd.open_workbook('H:/Perrysburg Users/VFaller/Public/Tasks/VF059 - Optical Modeling/modeling.xlsx')
        # self.sh = book.sheet_by_index(0)

        # self.wavelength = self.sh.col_values(0, 1)
        self.materials = {}

    def get_column(self, header):
        for col_index in range(self.sh.ncols):
            if self.sh.cell(0, col_index).value == header:
                return(col_index)

    def add_material(self, film, path):
        if film not in self.materials:
            self.materials[film] =  material(path)

    def set_wavelength(self, low, high):
        self.low_wavelength = low
        self.high_wavelength = high

    def plot_clicked(self):

        self.add_material("TCO", './Materials/Semiconductor/TCO.csv')
        self.add_material("CdSe", './Materials/Semiconductor/CdSe.csv')
        self.add_material("CdTe", './Materials/Semiconductor/CdTe.csv')
        self.add_material("Air", './Materials/Semiconductor/Air.csv')
        layers = ["Air", "CdTe", "CdSe", "TCO"]
        index_array = np.array([self.materials["Air"].nc,self.materials["CdTe"].nc,self.materials["CdSe"].nc,self.materials["TCO"].nc]).T
        self.wavelength = np.array(self.materials["TCO"].wv_raw)
        # self.add_material("Sapphire", 'C:/Writing Programs/Optical Modeling/Materials/Dielectric/Sapphire.csv')
        # layers = ["Sapphire"]
        # self.wavelength = self.materials["Sapphire"].wv_raw
        # wavelength = frange(self.low_wavelength, self.high_wavelength, self.increment)

        # layer_map = [map(complex, self.materials[i].n_raw, self.materials[i].k_raw) for i in layers]


        thkcdte = int(self.CdTeThickness.text())
        thkcdse = int(self.CdSeThickness.text())

        d_list = [inf, thkcdse, thkcdte, inf]
        theta0 = 0
        # n_list = zip(*layer_map)
        # n_list = map(interp(wv, ))

        # R = [tmm.unpolarized_RT(next(n_list), d_list, theta0, wv)['R'] for wv in self.wavelength]
        # T = [tmm.unpolarized_RT(next(n_list), d_list, theta0, wv)['T'] for wv in self.wavelength]

        data = tmm.unpolarized_RT(index_array, d_list, theta0, self.wavelength)

        r = data['R']
        t = data['T']
        a = 1-t-r

        plots = self.static.axis.plot(self.wavelength, r, self.wavelength, t, self.wavelength, a)


        self.static.figure.legend(plots, ['R', 'T', 'A'])
        self.static.canvas.draw()


class material():
    def __init__(self, path):
        f = pd.read_csv(path)
        self.wv_raw = f.wv
        self.nc = f.n+f.k*1j  #complex refractive index
        #self.n_raw = f.n
        #self.k_raw = f.k;



    def interpolate(self, start_wavelength, end_wavelength, increment = .2):
        start = max(floor(self.wv[0]/increment)*increment, start_wavelength)
        end = min(ceil(self.wv[-1]/increment)*increment, end_wavelength)
        wvrange = self.frange(start, end, increment)
        self.wv = []
        self.n = []
        self.k = []
        for i in wvrange:
            self.wv.append(self.wvrange[i])
            self.n.append(interp(i, self.wv_raw, self.n_raw))
            self.k.append(interp(i, self.wv_raw, self.k_raw))


def frange(x, y, jump):
        while x < y:
            yield(x)
            x += jump


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
"""
UI Program to create an optical modeler
"""
import sys
import random
import MainUI
import tmm_core as tmm
from math import floor, ceil

from PyQt5 import QtWidgets
import pandas as pd
import numpy as np
import matplotlib
from scipy.interpolate import interp1d
matplotlib.use("Qt5Agg")  # required currently because matplotlib uses PyQt4
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg, NavigationToolbar2QT)
from matplotlib.figure import Figure
from numpy.core.numeric import inf


class MPLibWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(MPLibWidget, self).__init__(parent)
        
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


class MW(QtWidgets.QMainWindow, MainUI.Ui_MainWindow):
    def __init__(self, parent=None):
        super(MW, self).__init__(parent)
        self.setupUi(self)
        self.wavelength = np.arange(300, 1700, 1)
        self.increment = .2
        self.index_array = np.array([])

        self.widget = MPLibWidget(self.GraphFrame)

        self.verticalLayout.addWidget(self.widget)
        self.plot.clicked.connect(self.plot_clicked)
        self.materials = []
        self.mat_df = pd.DataFrame([], self.wavelength)

        self.add_material("TCO", './Materials/Semiconductor/TCO.csv')
        self.add_material("CdSe", './Materials/Semiconductor/CdSe.csv')
        self.add_material("CdTe", './Materials/Semiconductor/CdTe.csv')
        self.add_material("Air", './Materials/Semiconductor/Air.csv')

    def add_material(self, film, path):
        if film not in self.mat_df:
            mat = Material(path, film, self.wavelength)
            self.materials.append(mat)
            self.mat_df = self.mat_df.join(mat.df1)

    def set_wavelength(self, low, high, interval):
        self.wavelength = np.arange(low, high, interval)
        df = self.mat_df.reindex(self.wavelength)
        df = df.interpolate('spline', order=3)
        self.mat_df = df

    def build_stack_array(self, layers):
        self.index_array = np.array(self.mat_df[layers])

    def plot_clicked(self):
        layers = ["Air", "CdTe", "CdSe", "TCO"]
        self.build_stack_array(layers)

        thkcdte = int(self.CdTeThickness.text())
        thkcdse = int(self.CdSeThickness.text())

        d_list = [inf, thkcdse, thkcdte, inf]
        theta0 = 0

        data = tmm.unpolarized_RT(self.index_array, d_list, theta0, self.wavelength)

        r = data['R']
        t = data['T']
        a = 1-t-r

        plots = self.widget.axis.plot(self.wavelength, r, self.wavelength, t, self.wavelength, a)
        self.widget.figure.legend(plots, ['R', 'T', 'A'])
        self.widget.canvas.draw()


class Material:
    def __init__(self, path, name, wavelengths):
        f = pd.read_csv(path)
        wv_raw = np.array(f.wv)
        nc = np.array(f.n+f.k*1j)  # complex refractive index
        self.name = name
        self.f = interp1d(wv_raw, nc, kind='cubic')
        self.min_wv = min(wv_raw)
        self.max_wv = max(wv_raw)
        self.df = pd.DataFrame(nc, wv_raw, [name])
        self.df = self.df.reindex(wavelengths)
        self.df = self.df1.interpolate('spline', order=3)

    def interp(self, wavelengths):
        wavelengths = wavelengths[np.nonzero(np.logical_and(wavelengths > self.min_wv, wavelengths < self.max_wv))]
        nc = self.f(wavelengths)
        self.df = pd.DataFrame(nc,wavelengths, [self.name])

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
"""
UI Program to create an optical modeler
"""
import sys
import os
import random
import MainUI
import tmm_core as tmm
from math import floor, ceil

from PyQt5 import QtWidgets
import pandas as pd
import numpy as np
import scipy as sp
import scipy.optimize as so
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

        pathdict = {}
        for d in os.listdir("C:/Writing Programs/Optical-Modeling/Materials/"):
            pathdict[d]=[]
            for file in os.listdir("C:/Writing Programs/Optical-Modeling/Materials/"+d):
                split = os.path.splitext(file)
                if split[1] == ".csv":
                    pathdict[d].append(split[0])
        self.lwDielectric.addItems(pathdict['Dielectric'])
        self.lwMetal.addItems(pathdict['Metal'])
        self.lwOther.addItems(pathdict['Other'])
        self.lWSemiconductors.addItems(pathdict['Semiconductor'])

        self.widget = MPLibWidget(self.GraphFrame)
        self.verticalLayout.addWidget(self.widget)
        self.btnAddLayer.clicked.connect(self.add_layer)
        self.btnRemoveLayer.clicked.connect(self.remove_layer)
        self.btnSwapLayer.clicked.connect(self.swap_layer)
        self.btnPlot.clicked.connect(self.plot_clicked)
        self.model = Model()

    def swap_layer(self):
        selected = self.tableWidget.selectionModel().selectedRows()[0].row()
        ctab = self.tab_Library.currentIndex()
        name = self.tab_Library.tabText(ctab)
        list_widget = self.tab_Library.widget(ctab).findChild(QtWidgets.QListWidget)
        txt = list_widget.currentItem().text()
        if txt not in self.model.mat_df:
            self.model.add_material(txt, './Materials/' + name + '/' + txt + '.csv')
        txt = QtWidgets.QTableWidgetItem(txt)
        self.tableWidget.setItem(selected, 1, txt)

    def add_layer(self):
        self.tableWidget.insertRow(0)
        rows = self.tableWidget.rowCount()-1
        rows = QtWidgets.QTableWidgetItem(str(rows))
        self.tableWidget.setItem(0, 0, rows)
        current_tab = self.tab_Library.currentIndex()
        mat_type = self.tab_Library.tabText(current_tab)
        list_widget = self.tab_Library.widget(current_tab).findChild(QtWidgets.QListWidget)
        txt = list_widget.currentItem().text()
        if txt not in self.model.mat_df:
            self.model.add_material(txt, './Materials/' + mat_type + '/' + txt + '.csv')
        txt = QtWidgets.QTableWidgetItem(txt)
        self.tableWidget.setItem(0, 1, txt)
        self.tableWidget.setItem(0, 2, QtWidgets.QTableWidgetItem(str(100)))

    def remove_layer(self):
        selected = self.tableWidget.selectionModel().selectedRows()
        for i in selected:
            self.tableWidget.removeRow(i.row())

    def plot_clicked(self):
        layers = ['Air']
        d_list = [inf]

        rows = self.tableWidget.rowCount()
        for i in range(rows):
            layers.append(self.tableWidget.item(i, 1).text())
            d_list.append(int(self.tableWidget.item(i, 2).text()))
        d_list[-1] = inf
        theta = float(self.le_Theta0.text())*sp.pi/180

        self.model.run(layers, d_list, theta)
        r = self.model.data['R']
        t = self.model.data['T']
        a = 1-t-r

        plots = self.widget.axis.plot(self.model.wavelength, r, self.model.wavelength, t, self.model.wavelength, a)
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
        # self.df = self.df.reindex(wavelengths)
        # self.df = self.df.interpolate('spline', order=3)
        self.interp(wavelengths)

    def interp(self, wavelengths):
        wavelengths = wavelengths[np.nonzero(np.logical_and(wavelengths > self.min_wv, wavelengths < self.max_wv))]
        nc = self.f(wavelengths)
        self.df = pd.DataFrame(nc, wavelengths, [self.name])


class Model:
    def __init__(self):
        self.wavelength = np.arange(300, 1700, 1)
        self.increment = .2
        self.index_array = np.array([])
        self.materials = []
        self.mat_df = pd.DataFrame(1+0j, self.wavelength, ['Air'])
        self.data = {}
        self.add_material("TCO", './Materials/Semiconductor/TCO.csv')

    def add_material(self, film, path):
        if film not in self.mat_df:
            mat = Material(path, film, self.wavelength)
            self.materials.append(mat)
            self.mat_df = self.mat_df.join(mat.df)

    def set_wavelength(self, low, high, interval):
        self.wavelength = np.arange(low, high, interval)
        df = self.mat_df.reindex(self.wavelength)
        df = df.interpolate('spline', order=3)
        self.mat_df = df

    def better_bruggeman(self, n1, n2, percent_included):
        p = n1/n2
        b = .25*((3*percent_included-1)*(1/p-p)+p)
        z = b + (b**2 + .5)**0.5
        e = z*n1*n2
        return {"e": e, "n": e**0.5, 'conc': percent_included, "n1": n1, 'n2': n2}

    def run(self, layers, thicknesses, theta0):
        self.index_array = np.array(self.mat_df[layers])
        self.data = tmm.unpolarized_RT(self.index_array, thicknesses, theta0, self.wavelength)


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
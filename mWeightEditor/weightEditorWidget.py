from __future__ import print_function
from __future__ import absolute_import
from .Qt import QtGui, QtCore, QtWidgets, QtCompat

from functools import partial
from maya import cmds, OpenMaya
import os
import re
import difflib
import weakref
from six.moves import zip

try:
    from blurdev.gui import Window
except ImportError:
    from .Qt.QtWidgets import QMainWindow as Window

from .weightTools.skinData import DataOfSkin
from .weightTools.abstractData import DataQuickSet
from .weightTools.weightMapsData import DataOfBlendShape, DataOfDeformers

from .weightTools.tableWidget import FastTableView, TableModel
from .weightTools.spinnerSlider import ValueSettingWE, ButtonWithValue
from .weightTools.utils import (
    GlobalContext,
    addNameChangedCallback,
    removeNameChangedCallback,
    toggleBlockSignals,
    SettingWithRedraw,
    SettingVariable,
    ToggleHeaderVisibility,
)
from six.moves import range

# -----------------------------------------------------------------
# styleSheet and icons --------------------------------------------
# -----------------------------------------------------------------
def getIcon(iconNm):
    fileVar = os.path.realpath(__file__)
    uiFolder, filename = os.path.split(fileVar)
    iconPth = os.path.join(uiFolder, "img", iconNm + ".png")
    return QtGui.QIcon(iconPth)


def loadUndoPlugin():
    fileVar = os.path.realpath(__file__)
    uiFolder, filename = os.path.split(fileVar)
    plugPth = os.path.join(uiFolder, "weightTools", "undoPlug.py")
    cmds.loadPlugin(plugPth)


def getUiFile(fileVar, subFolder="ui", uiName=None):
    uiFolder, filename = os.path.split(fileVar)
    if uiName is None:
        uiName = os.path.splitext(filename)[0]
    if subFolder:
        uiFile = os.path.join(uiFolder, subFolder, uiName+".ui")
    return uiFile


_icons = {
    "lock": getIcon("lock-48"),
    "unlock": getIcon("unlock-48"),
    "refresh": getIcon("arrow-circle-045-left"),
    "clearText": getIcon("clearText"),
    "unlockJnts": getIcon("unlockJnts"),
    "lockJnts": getIcon("lockJnts"),
    "zeroOn": getIcon("zeroOn"),
    "zeroOff": getIcon("zeroOff"),
    "option": getIcon("option"),
}


class SkinWeightWin(Window):
    colWidth = 30
    maxWidthCentralWidget = 340

    def __init__(self, parent=None):
        super(SkinWeightWin, self).__init__(parent)
        if not cmds.pluginInfo("blurSkin", query=True, loaded=True):
            cmds.loadPlugin("blurSkin")

        uiPath = getUiFile(__file__)
        QtCompat.loadUi(uiPath, self)

        if not cmds.pluginInfo("undoPlug", query=True, loaded=True):
            loadUndoPlugin()
        self.getOptionVars()
        self.buildRCMenu()

        self.dataOfDeformer = DataOfSkin(
            useShortestNames=self.useShortestNames,
            hideZeroColumn=self.hideZeroColumn,
            mainWindow=self,
            createDisplayLocator=self.useDisplayLocator,
        )

        self.get_data_frame()
        self.createWindow()
        styleSheet = open(os.path.join(os.path.dirname(__file__), "xsi.css"), 'r').read()
        self.setStyleSheet(styleSheet)

        self.addCallBacks()
        self.setWindowDisplay()
        self.applyDisplayColumnsFilters(None)
        self.refreshCurrentSelectionOrder()

    # -------------------------------------------------------------
    # window events -----------------------------------------------
    # -------------------------------------------------------------
    def showEvent(self, event):
        super(SkinWeightWin, self).showEvent(event)
        self.getOptionVars()

    def closeEvent(self, event):
        self.deleteCallBacks()

        pos = self.pos()
        size = self.size()
        cmds.optionVar(clearArray="SkinWeightWindow")
        for el in pos.x(), pos.y(), size.width(), size.height():
            cmds.optionVar(intValueAppend=("SkinWeightWindow", el))
        self._tv.deleteLater()
        super(SkinWeightWin, self).closeEvent(event)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MidButton:
            self.resizeToMinimum()
        elif event.button() == QtCore.Qt.LeftButton:
            self._tv.clearSelection()
        super(SkinWeightWin, self).mousePressEvent(event)

    # -------------------------------------------------------------
    # widget creation/edition  ------------------------------------
    # -------------------------------------------------------------
    def addMinButton(self):
        self.setWindowFlags(QtCore.Qt.Window)

    def setWindowDisplay(self):
        self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.Tool)
        self.setWindowTitle("Weight Editor")
        self.refreshPosition()
        self.show()

    def resizeToMinimum(self):
        wdth = self._tv.VHeaderView.width() + self._tv.viewportSizeHint().width()
        self.resize(wdth, self.height())

    def addButtonsDirectSet(self, lstBtns):
        theCarryWidget = QtWidgets.QWidget()

        carryWidgLayoutlayout = QtWidgets.QHBoxLayout(theCarryWidget)
        carryWidgLayoutlayout.setContentsMargins(40, 0, 0, 0)
        carryWidgLayoutlayout.setSpacing(0)

        for theVal in lstBtns:
            newBtn = QtWidgets.QPushButton("{0:.0f}".format(theVal))

            newBtn.clicked.connect(self.prepareToSetValue)
            newBtn.clicked.connect(partial(self.doAddValue, theVal / 100.0))
            newBtn.clicked.connect(self.postSetValue)

            carryWidgLayoutlayout.addWidget(newBtn)
        theCarryWidget.setMaximumSize(self.maxWidthCentralWidget, 14)

        return theCarryWidget

    def changeLock(self, val):
        if val:
            self.lockBTN.setIcon(_icons["lock"])
        else:
            self.lockBTN.setIcon(_icons["unlock"])
        self.unLock = not val

    def changeDisplayZero(self, val):
        if val:
            self.zeroCol_BTN.setIcon(_icons["zeroOn"])
        else:
            self.zeroCol_BTN.setIcon(_icons["zeroOff"])
        self.toggleZeroColumn(val)

    def changeDisplayLock(self, val):
        if val:
            self.locked_BTN.setIcon(_icons["lockJnts"])
        else:
            self.locked_BTN.setIcon(_icons["unlockJnts"])
        self.toggleDisplayLockColumn(val)

    def changeAddAbs(self, checked):
        self.widgetAbs.setVisible(False)
        self.widgetAdd.setVisible(False)
        self.widgetAbs.setVisible(checked)
        self.widgetAdd.setVisible(not checked)
        self.valueSetter.setAddMode(not checked)

    def changeAddPerc(self, checked):
        self.addPercentage = checked

    def deferredBtns(self):
        for nm in ["abs", "add", "addPerc"]:
            theBtn = self.findChild(QtWidgets.QPushButton, nm + "BTN")
            if theBtn:
                theBtn.setAutoExclusive(True)

        self.addBTN.setChecked(True)
        self.reassignLocallyBTN.setMinimumHeight(24)
        self.averageBTN.setMinimumHeight(24)

    def createWindow(self):
        theLayout = self.mainLayout

        theLayout.setContentsMargins(10, 10, 10, 2)
        theLayout.setSpacing(3)
        self.addPercentage = False

        self._tm = TableModel(self)
        self._tm.update(self.dataOfDeformer)

        self._tv = FastTableView(self, colWidth=self.colWidth)
        self._tv.setModel(self._tm)
        self.unLock = True
        self.lockBTN.setIcon(_icons["unlock"])
        self.lockBTN.setMaximumSize(24, 24)
        self.lockBTN.setCheckable(True)
        self.lockBTN.setChecked(False)
        self.lockBTN.toggled.connect(self.changeLock)
        self.lockBTN.setText("")

        self.valueSetter = ValueSettingWE(self)

        Hlayout = QtWidgets.QHBoxLayout(self)
        Hlayout.setContentsMargins(0, 0, 0, 0)
        Hlayout.setSpacing(0)
        Hlayout.addWidget(self.valueSetter)
        self.valueSetter.setMaximumWidth(self.maxWidthCentralWidget)

        self.widgetAbs = self.addButtonsDirectSet(
            [0, 10, 25, 100.0 / 3, 50, 200 / 3.0, 75, 90, 100]
        )
        self.widgetAdd = self.addButtonsDirectSet(
            [-100, -75, -200 / 3.0, -50, -100.0 / 3, -25, 25, 100.0 / 3, 50, 200 / 3.0, 75, 100]
        )

        Hlayout2 = QtWidgets.QHBoxLayout(self)
        Hlayout2.setContentsMargins(0, 0, 0, 0)
        Hlayout2.setSpacing(0)
        Hlayout2.addWidget(self.widgetAbs)
        Hlayout2.addWidget(self.widgetAdd)

        self.carryWidgetLAY.addLayout(Hlayout2)
        self.carryWidgetLAY.addLayout(Hlayout)
        self.widgetAbs.hide()

        theLayout.addWidget(self._tv)

        self.setColumnVisSize()

        self.pruneWghtBTN = ButtonWithValue(
            self, usePow=True, name="Prune", minimumValue=-1, defaultValue=2
        )
        self.botLayout.insertWidget(2, self.pruneWghtBTN)

        self.smoothBTN = ButtonWithValue(
            self, usePow=False, name="Smooth", minimumValue=1, defaultValue=3
        )
        self.botLayout.insertWidget(4, self.smoothBTN)

        self.percentBTN = ButtonWithValue(
            self,
            usePow=False,
            name="%",
            minimumValue=0,
            maximumValue=1.0,
            defaultValue=1.0,
            step=0.1,
            clickable=False,
        )
        self.botLayout.insertWidget(5, self.percentBTN)
        self.percentBTN.setMaximumWidth(30)

        self.problemVertsBTN = ButtonWithValue(
            self, usePow=False, name="problemVerts", minimumValue=2, defaultValue=4
        )
        self.topLayout.addWidget(self.problemVertsBTN)

        self.problemVertsBTN.clicked.connect(self.selProbVerts)
        self.problemVerts_btn.deleteLater()

        self.uiSetFromUVsBTN.clicked.connect(self.setUsingUvs)
        for el in self.uiUVsSettingWDG.children():
            el.setEnabled(True)

        averageBTN = ButtonWithValue(
            self,
            usePow=False,
            name="Average",
            minimumValue=0,
            maximumValue=1.0,
            defaultValue=1.0,
            step=0.1,
            clickable=True,
        )
        self.botLayout.replaceWidget(self.averageBTN, averageBTN)
        self.averageBTN.hide()
        self.averageBTN = averageBTN

        for nm in ["swap"]:
            theBtn = self.findChild(QtWidgets.QPushButton, nm + "BTN")
            if theBtn:
                theBtn.setEnabled(False)
                theBtn.hide()

        self.refreshBTN.clicked.connect(self.refreshBtn)
        self.refreshBTN.setIcon(_icons["refresh"])
        self.refreshBTN.setText("")

        self.smoothBTN.clicked.connect(self.smooth)
        self.smoothBTN.clicked.connect(self.refreshBtn)

        self.absBTN.toggled.connect(self.changeAddAbs)
        self.addPercBTN.toggled.connect(self.changeAddPerc)
        self.pruneWghtBTN.clicked.connect(self.pruneWeights)
        self.normalizeBTN.clicked.connect(self.doNormalize)

        self.averageBTN.clicked.connect(self.doAverage)
        self.reassignLocallyBTN.clicked.connect(self.reassignLocally)

        self.listInputs_CB.currentIndexChanged.connect(self.changeTypeOfData)
        self.orderType_CB.currentTextChanged.connect(self.changeOrder)
        self.nbColumns_CB.currentTextChanged.connect(self.maxColumnsDisplay)

        for uiName in [
            "reassignLocallyBTN",
            "widgetAbs",
            "widgetAdd",
            "valueSetter",
        ]:  # "averageBTN",

            theUI = self.findChild(QtWidgets.QWidget, uiName)
            if theUI:
                theUI.setEnabled(False)
                self._tv.selEmptied.connect(theUI.setEnabled)
        for btn in [self.exportBTN, self.importBTN]:
            btn.setEnabled(False)
        self.exportBTN.clicked.connect(self.exportAction)
        self.importBTN.clicked.connect(self.importAction)

        self._tv.selEmptied.connect(self.exportButtonsVis)
        cmds.evalDeferred(self.deferredBtns)

        # display the list of paintable attributes
        with toggleBlockSignals([self.listInputs_CB]):
            self.listInputs_CB.addItems(["skinCluster", "blendShape", "deformers"])
        self.cancelImportBTN.clicked.connect(self.importQueryFrame.hide)
        self.doImportXmlBTN.clicked.connect(self.doImportXmlCouples)
        self.doImportXmlBTN.clicked.connect(self.importQueryFrame.hide)

        self.importQueryFrame.hide()

        self.searchInfluences_le.textChanged.connect(self.filterInfluences)
        self.clearWildCardBTN.clicked.connect(lambda: self.searchInfluences_le.setText(""))
        self.clearWildCardBTN.setIcon(_icons["clearText"])
        self.clearWildCardBTN.setText("")

        self.zeroCol_BTN.setIcon(_icons["zeroOff"])
        self.zeroCol_BTN.setText("")
        self.zeroCol_BTN.setCheckable(True)
        self.zeroCol_BTN.setChecked(self.hideZeroColumn)
        self.zeroCol_BTN.toggled.connect(self.changeDisplayZero)

        self.locked_BTN.setIcon(_icons["unlockJnts"])
        self.locked_BTN.setText("")
        self.locked_BTN.setCheckable(True)
        self.locked_BTN.setChecked(self.hideLockColumn)
        self.locked_BTN.toggled.connect(self.changeDisplayLock)

        self.option_BTN.setIcon(_icons["option"])
        self.option_BTN.setText("")
        self.option_BTN.mousePressEvent = self.showRightClickMenu

        # ------ Copy / Paste -----------------------------------------------
        self.copyBTN.clicked.connect(self.doCopyArray)
        self.pasteBTN.clicked.connect(self.doPasteArray)

    # -------------------------------------------------------------
    # export import  ----------------------------------------------
    # -------------------------------------------------------------
    def exportAction(self):
        colIndices = self._tv.HHeaderView.getSelectedColumns()
        self.dataOfDeformer.exportColumns(colIndices)

    def importAction(self):
        colIndices = self._tv.HHeaderView.getSelectedColumns()
        resultImport = self.dataOfDeformer.importColumns(colIndices)
        if resultImport is not None:
            self.associationXml_tbl.lstComboxes = []
            self.associationXml_tbl.clear()
            self.associationXml_tbl.setColumnWidth(0, 150)

            colNames, filesPath = resultImport
            shortFilePaths = [pth.split("/")[-1] for pth in filesPath]
            self.dicNmFilePath = dict(zip(shortFilePaths, filesPath))

            shortFilePaths.insert(0, "")
            self.importQueryFrame.show()
            for nm in self.dataOfDeformer.shortColumnsNames:
                associationItem = QtWidgets.QTreeWidgetItem()
                associationItem.setText(0, str(nm))
                associationItem.setFlags(
                    associationItem.flags()
                    | QtCore.Qt.ItemIsEditable
                    | QtCore.Qt.ItemIsUserCheckable
                )
                self.associationXml_tbl.addTopLevelItem(associationItem)
                comboB = QtWidgets.QComboBox()
                comboB.addItems(shortFilePaths)
                self.associationXml_tbl.setItemWidget(associationItem, 1, comboB)
                matchNames = difflib.get_close_matches(nm, shortFilePaths, 1, 0.2) or []
                if matchNames:
                    ind = shortFilePaths.index(matchNames[0])
                    comboB.setCurrentIndex(ind)
                else:
                    comboB.setCurrentIndex(0)
                self.associationXml_tbl.lstComboxes.append(comboB)
            self.fileImportPths = filesPath
        else:
            self.refresh(force=True)

    def doImportXmlCouples(self):
        for inCol, comboB in enumerate(self.associationXml_tbl.lstComboxes):
            currentText = comboB.currentText()
            if currentText in self.dicNmFilePath:
                self.dataOfDeformer.doImport(self.dicNmFilePath[currentText], inCol)
        self.refresh(force=True)

    def exportButtonsVis(self, val):
        for btn in [self.exportBTN, self.importBTN]:
            btn.setEnabled(
                not self.dataOfDeformer.isSkinData and val
            )

    # -------------------------------------------------------------
    # callBacks ---------------------------------------------------
    # -------------------------------------------------------------
    def renameCB(self, oldName, newName):
        if self.dataOfDeformer:
            self.dataOfDeformer.renameCB(oldName, newName)

    def deleteCB(self, nodeName):
        print("to be Deleted ", nodeName)

    def addCallBacks(self):
        self.refreshSJ = cmds.scriptJob(event=["SelectionChanged", self.refresh])
        self.renameCallBack = addNameChangedCallback(self.renameCB)

        sceneUpdateCallback = OpenMaya.MSceneMessage.addCallback(
            OpenMaya.MSceneMessage.kBeforeNew, self.deselectAll
        )  # kSceneUpdate
        self.close_callback = [sceneUpdateCallback]
        self.close_callback.append(
            OpenMaya.MSceneMessage.addCallback(OpenMaya.MSceneMessage.kBeforeOpen, self.deselectAll)
        )

    def deleteCallBacks(self):
        removeNameChangedCallback(self.renameCallBack)
        self.dataOfDeformer.deleteDisplayLocator()
        cmds.scriptJob(kill=self.refreshSJ, force=True)
        for callBck in self.close_callback:
            OpenMaya.MSceneMessage.removeCallback(callBck)

    def deselectAll(self, *args):
        self._tm.beginResetModel()
        self.dataOfDeformer.clearData()
        self._tm.endResetModel()

    # -------------------------------------------------------------
    # right click menu --------------------------------------------
    # -------------------------------------------------------------
    def buildRCMenu(self):
        self.popMenu = QtWidgets.QMenu(self)

        resizeAction = self.popMenu.addAction("resize to minimum(MiddleClick)")
        resizeAction.triggered.connect(self.resizeToMinimum)

        chbox = QtWidgets.QCheckBox("auto Prune", self.popMenu)
        chbox.setChecked(self.autoPrune)
        chbox.toggled.connect(self.autoPruneChecked)
        checkableAction = QtWidgets.QWidgetAction(self.popMenu)
        checkableAction.setDefaultWidget(chbox)
        self.popMenu.addAction(checkableAction)

        chbox = QtWidgets.QCheckBox("shortest name", self.popMenu)
        chbox.setChecked(self.useShortestNames)
        chbox.toggled.connect(self.useShortestNameChecked)
        checkableAction = QtWidgets.QWidgetAction(self.popMenu)
        checkableAction.setDefaultWidget(chbox)
        self.popMenu.addAction(checkableAction)

        chbox = QtWidgets.QCheckBox("display points", self.popMenu)
        chbox.setChecked(self.useDisplayLocator)
        chbox.toggled.connect(self.useDisplayLocatorChecked)
        checkableAction = QtWidgets.QWidgetAction(self.popMenu)
        checkableAction.setDefaultWidget(chbox)
        self.popMenu.addAction(checkableAction)

        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.showMenu)

    def showRightClickMenu(self, event):
        self.popMenu.exec_(event.globalPos())

    def showMenu(self, pos):
        child = self.childAt(pos)
        widgetName = child.objectName()
        if widgetName in ["centralwidget", "topButtonsWidget", "carryWidget", "widgetAdd", "widgetAbs"]:
            self.popMenu.exec_(self.mapToGlobal(pos))

    # -------------------------------------------------------------
    # optionVars --------------------------------------------------
    # -------------------------------------------------------------
    def getOptionVars(self):
        self.autoPrune = (
            cmds.optionVar(q="autoPrune") if cmds.optionVar(exists="autoPrune") else False
        )
        self.autoPruneValue = (
            cmds.optionVar(q="autoPruneValue")
            if cmds.optionVar(exists="autoPruneValue")
            else 0.0001
        )
        self.useShortestNames = (
            cmds.optionVar(q="useShortestNames")
            if cmds.optionVar(exists="useShortestNames")
            else True
        )
        self.hideZeroColumn = (
            cmds.optionVar(q="hideZeroColumn") if cmds.optionVar(exists="hideZeroColumn") else False
        )
        self.hideLockColumn = (
            cmds.optionVar(q="hideLockColumn") if cmds.optionVar(exists="hideLockColumn") else False
        )
        self.useDisplayLocator = (
            cmds.optionVar(q="useDisplayLocator")
            if cmds.optionVar(exists="useDisplayLocator")
            else True
        )

    def maxColumnsDisplay(self, nb):
        self.applyDisplayColumnsFilters(None)

    def sort_human(self, lst):
        def convert(ltrs):
            return float(ltrs) if ltrs.isdigit() else ltrs

        def alphanum(key):
            return [convert(c) for c in re.split(r'([-+]?[0-9]*\.?[0-9]*)', key)]

        lst.sort(key=alphanum)
        return lst

    def changeOrder(self, orderType):
        HH = self._tv.HHeaderView

        with ToggleHeaderVisibility(HH):
            if orderType == "Default":
                self.produceOrder(HH, self.dataOfDeformer.driverNames)
            if orderType == "Alphabetical":
                newOrderDriverNames = sorted(self.dataOfDeformer.driverNames)
                self.produceOrder(HH, newOrderDriverNames)
            elif orderType == "Side Alphabetical":
                allNewNames = []
                for el in self.dataOfDeformer.driverNames:
                    spl = el.split("_")
                    if len(spl) > 2:
                        spl.append(spl.pop(1))
                    newName = "_".join(spl)
                    allNewNames.append((newName, el))
                newOrderDriverNames = [el for newel, el in sorted(allNewNames)]
                self.produceOrder(HH, newOrderDriverNames)
            elif orderType == "Value":
                self.produceOrder(HH, self.dataOfDeformer.getNamesHighestColumns())
        self.applyDisplayColumnsFilters(None)

    def produceOrder(self, HH, newOrderDriverNames):
        ind = len(newOrderDriverNames)
        for destInd, element in enumerate(newOrderDriverNames):
            currentInd = self.currentSectionsOrder[element]
            elementAtDestInd = self.currentSectionsOrderReverse[destInd]
            if ind != currentInd:
                HH.swapSections(destInd, currentInd)
                self.currentSectionsOrderReverse[destInd] = element
                self.currentSectionsOrderReverse[currentInd] = elementAtDestInd
                self.currentSectionsOrder[elementAtDestInd] = currentInd
                self.currentSectionsOrder[element] = destInd

    def applyDisplayColumnsFilters(self, newText):
        displayColumns = [True] * self.dataOfDeformer.columnCount
        # first apply the Text ---------------------
        if newText is None:
            newText = self.searchInfluences_le.text()
        if newText:
            newTexts = newText.split(" ")
            while "" in newTexts:
                newTexts.remove("")
            newTexts = [txt.replace("?", ".").replace("*", ".*") for txt in newTexts]
            for ind, nameInfluence in enumerate(self.dataOfDeformer.columnsNames):
                foundText = False
                for txt in newTexts:
                    foundText = re.search(txt, nameInfluence, re.IGNORECASE) is not None
                    if foundText:
                        break
                displayColumns[ind] = foundText
        # then apply the Zero Colums: -----------------------------
        if self.hideZeroColumn:
            for ind in self.dataOfDeformer.hideColumnIndices:
                displayColumns[ind] = False
        # then apply the Lock Colums: -----------------------------
        if self.hideLockColumn:
            for ind, isLocked in enumerate(self.dataOfDeformer.lockedColumns):
                if isLocked:
                    displayColumns[ind] = False
        # now apply how many to show: -----------------------------
        if self.dataOfDeformer.isSkinData:
            nbToShow = self.nbColumns_CB.currentText()
            if nbToShow != "All":
                nbToShow = int(nbToShow)
                for i in range(self.dataOfDeformer.nbDrivers):
                    driverName = self.currentSectionsOrderReverse[i]
                    columnIndex = self.dataOfDeformer.driverNames.index(driverName)
                    if displayColumns[columnIndex]:
                        if nbToShow > 0:
                            nbToShow -= 1
                        else:
                            displayColumns[columnIndex] = False
        with ToggleHeaderVisibility(self._tv.HHeaderView):
            # now do the hidding --------------------------------------------
            for ind, isVisible in enumerate(displayColumns):
                if isVisible:
                    self._tv.showColumn(ind)
                else:
                    self._tv.hideColumn(ind)
            if self.dataOfDeformer.isSkinData:  # show the sum column, always
                self._tv.showColumn(self.dataOfDeformer.columnCount + 1)

    def toggleDisplayLockColumn(self, checked):
        cmds.optionVar(intValue=["hideLockColumn", checked])
        self.hideLockColumn = checked
        self.applyDisplayColumnsFilters(None)

    def toggleZeroColumn(self, checked):
        cmds.optionVar(intValue=["hideZeroColumn", checked])
        self.hideZeroColumn = checked
        self.applyDisplayColumnsFilters(None)
        self.zeroCol_BTN.setChecked(checked)

    def filterInfluences(self, newText):
        self.applyDisplayColumnsFilters(newText)

    def autoPruneChecked(self, checked):
        cmds.optionVar(intValue=["autoPrune", checked])
        self.autoPrune = checked
        self.popMenu.close()

    def useShortestNameChecked(self, checked):
        cmds.optionVar(intValue=["useShortestNames", checked])
        self.useShortestNames = checked
        self.dataOfDeformer.useShortestNames = checked
        if self.dataOfDeformer.isSkinData:
            self.dataOfDeformer.getDriversShortNames()
        self.popMenu.close()

    def useDisplayLocatorChecked(self, checked):
        cmds.optionVar(intValue=["useDisplayLocator", checked])
        self.useDisplayLocator = checked
        if checked:
            with SettingVariable(self, "unLock", valueOn=False, valueOut=True):
                self.dataOfDeformer.createDisplayLocator(forceSelection=True)
        else:
            self.dataOfDeformer.removeDisplayLocator()
        self.popMenu.close()

    # -------------------------------------------------------------
    # Refresh -----------------------------------------------------
    # -------------------------------------------------------------
    def refreshPosition(self):
        vals = cmds.optionVar(q="SkinWeightWindow")
        if vals:
            self.move(vals[0], vals[1])
            self.resize(vals[2], vals[3])

    def refreshBtn(self):
        with SettingWithRedraw(self):
            self.refresh(force=True)

    @staticmethod
    def refreshPaintEditor(self):
        try:
            import mPaintEditor
        except ImportError:
            return

        if mPaintEditor.PAINT_EDITOR in QtWidgets.QApplication.instance().topLevelWidgets():
            mPaintEditor.PAINT_EDITOR.refreshColorsAndLocks()

    def refreshSkinDisplay(self):  # call by skinBrush
        self._tm.beginResetModel()
        self.dataOfDeformer.rebuildRawSkin()
        self.dataOfDeformer.convertRawSkinToNumpyArray()
        self._tm.endResetModel()
        self._tv.repaint()

    def refresh(self, force=False):
        if self.unLock or force:
            if self.dataOfDeformer.isSkinData:
                self.changeOrder("Default")
            self._tm.beginResetModel()
            for ind in range(self.dataOfDeformer.columnCount):
                self._tv.showColumn(ind)

            with GlobalContext(message="weightEdtior getAllData", doPrint=False):
                self.dataOfDeformer.updateDisplayVerts([])
                resultData = self.dataOfDeformer.getAllData(force=force)
                doForce = not resultData
                doForce = (
                    doForce
                    and cmds.objExists(self.dataOfDeformer.deformedShape)
                    and self.dataOfDeformer.theDeformer == ""
                )
                doForce = doForce and cmds.nodeType(self.dataOfDeformer.deformedShape) in [
                    "mesh",
                    "nurbsSurface",
                ]
                if doForce:
                    self.dataOfDeformer.clearData()

            self.dataOfDeformer.getLocksInfo()
            self._tm.endResetModel()
            self.setColumnVisSize()
            self.applyDisplayColumnsFilters(None)
            if not resultData and self.dataOfDeformer.isSkinData:
                self.highlightSelectedDeformers()
            self._tv.selEmptied.emit(False)
            self._tv.repaint()

            if self.dataOfDeformer.isSkinData:
                self.refreshCurrentSelectionOrder()
                self.changeOrder(self.orderType_CB.currentText())

    def refreshCurrentSelectionOrder(self):
        if self.dataOfDeformer.isSkinData:
            self.currentSectionsOrder = dict(
                [(el, ind) for ind, el in enumerate(self.dataOfDeformer.driverNames)]
            )
            self.currentSectionsOrderReverse = dict(
                [(ind, el) for ind, el in enumerate(self.dataOfDeformer.driverNames)]
            )
        else:
            self.currentSectionsOrder, self.currentSectionsOrderReverse = {}, {}

    # -------------------------------------------------------------
    # Functions ---------------------------------------------------
    # -------------------------------------------------------------
    def reassignLocally(self):
        chunks = self.getRowColumnsSelected()
        if chunks:
            actualyVisibleColumns = [
                indCol
                for indCol in self.dataOfDeformer.hideColumnIndices
                if not self._tv.HHeaderView.isSectionHidden(indCol)
            ]
            with SettingWithRedraw(self):
                self.dataOfDeformer.preSettingValuesFn(chunks, actualyVisibleColumns)
                self.dataOfDeformer.reassignLocally()
                self.postSetValue()

    def pruneWeights(self):
        chunks = self.getRowColumnsSelected()
        if not chunks:
            chunks = [(0, self.dataOfDeformer.rowCount - 1, 0, self.dataOfDeformer.columnCount - 1)]
        actualyVisibleColumns = []

        with SettingWithRedraw(self):
            self.dataOfDeformer.preSettingValuesFn(chunks, actualyVisibleColumns)
            self.dataOfDeformer.pruneWeights(self.pruneWghtBTN.precisionValue / 100.0)
            self.postSetValue()

    def doNormalize(self):
        chunks = self.getRowColumnsSelected()
        if not chunks:
            chunks = [(0, self.dataOfDeformer.rowCount - 1, 0, self.dataOfDeformer.columnCount - 1)]
        actualyVisibleColumns = []

        with SettingWithRedraw(self):
            self.dataOfDeformer.preSettingValuesFn(chunks, actualyVisibleColumns)
            self.dataOfDeformer.normalize()
            self.postSetValue()

    def doCopyArray(self):
        self.prepareToSetValue(selectAllIfNothing=True)
        self.dataOfDeformer.copyArray()
        self.pasteBTN.setEnabled(True)

    def doPasteArray(self):
        with SettingWithRedraw(self):
            self.prepareToSetValue(selectAllIfNothing=True)
            result = self.dataOfDeformer.pasteArray()
            if result:
                self.postSetValue()
            else:
                cmds.confirmDialog(m="Not same number of deformers\nFAILED", t="can't paste")

    def doAverage(self):
        self.prepareToSetValue(selectAllIfNothing=True)
        self.doAddValue(self.averageBTN.precision, forceAbsolute=False, average=True)
        self.postSetValue()

    def smooth(self):
        with SettingWithRedraw(self):
            if self.dataOfDeformer.isSkinData:
                chunks = self.getRowColumnsSelected()
                if not chunks:
                    chunks = [
                        (
                            0,
                            self.dataOfDeformer.rowCount - 1,
                            0,
                            self.dataOfDeformer.columnCount - 1,
                        )
                    ]
                self.dataOfDeformer.getConnectedBlurskinDisplay(disconnectWeightList=True)
                # convert to vertices or get the vertices
                self.dataOfDeformer.smoothSkin(
                    chunks, repeat=self.smoothBTN.precision, percentMvt=self.percentBTN.precision
                )
                if self.dataOfDeformer.blurSkinNode and cmds.objExists(
                    self.dataOfDeformer.blurSkinNode
                ):
                    cmds.delete(self.dataOfDeformer.blurSkinNode)
            else:
                success = self.prepareToSetValue()
                if success:
                    self.dataOfDeformer.smoothVertices(iteration=self.smoothBTN.precision)
                    self.postSetValue()

    def selProbVerts(self):
        vtx = self.dataOfDeformer.fixAroundVertices(tolerance=self.problemVertsBTN.precision)
        selVertices = self.dataOfDeformer.orderMelList(vtx)
        inList = [
            "{1}.vtx[{0}]".format(el, self.dataOfDeformer.deformedShape) for el in selVertices
        ]
        cmds.select(inList)

    def setUsingUvs(self):
        using_U = self.uiURBTN.isChecked()
        normalize = self.uiNormalizeUVsCBOX.isChecked()
        opposite = self.uiOppositeUVsCBOX.isChecked()
        with SettingWithRedraw(self):
            success = self.prepareToSetValue(selectAllIfNothing=True)
            if success:
                self.dataOfDeformer.setUsingUVs(using_U, normalize, opposite)
                self.postSetValue()

    # -------------------------------------------------------------
    # Basic set Values --------------------------------------------
    # -------------------------------------------------------------
    def prepareToSetValue(self, selectAllIfNothing=False):
        chunks = self.getRowColumnsSelected()
        actualyVisibleColumns = [
            indCol
            for indCol in self.dataOfDeformer.hideColumnIndices
            if not self._tv.HHeaderView.isSectionHidden(indCol)
        ]
        if selectAllIfNothing and not chunks:
            chunks = [(0, self.dataOfDeformer.rowCount - 1, 0, self.dataOfDeformer.columnCount - 1)]
        if chunks:
            self.dataOfDeformer.preSettingValuesFn(chunks, actualyVisibleColumns)
            return True
        return False

    def postSetValue(self):
        if self.dataOfDeformer.isSkinData:
            self.dataOfDeformer.postSkinSet()
            undoArgs = (self.dataOfDeformer.undoValues,)
            redoArgs = (self.dataOfDeformer.redoValues,)
            self.dataOfDeformer.undoDic["mainWindow"] = self

            newClass = DataQuickSet(undoArgs, redoArgs, **self.dataOfDeformer.undoDic)
            cmds.pythonCommand(hex(id(newClass)))
        else:
            # return
            undoArgs = (self.dataOfDeformer.undoValues,)
            redoArgs = (self.dataOfDeformer.redoValues,)
            newClass = DataQuickSet(undoArgs, redoArgs, mainWindow=self)
            cmds.pythonCommand(hex(id(newClass)))

    def doAddValue(self, val, forceAbsolute=False, average=False):
        with SettingWithRedraw(self):
            if self.valueSetter.addMode and not forceAbsolute:
                if self.dataOfDeformer.isSkinData:
                    self.dataOfDeformer.setSkinData(
                        val, percent=self.addPercentage, autoPrune=self.autoPrune, average=average
                    )
                else:
                    self.dataOfDeformer.doAdd(
                        val, percent=self.addPercentage, autoPrune=self.autoPrune, average=average
                    )
            else:
                self.dataOfDeformer.absoluteVal(val)

    # -------------------------------------------------------------
    # Selection ---------------------------------------------------
    # -------------------------------------------------------------
    def storeSelection(self):
        selection = self._tv.selectionModel().selection()
        self.topLeftBotRightSel = [
            (item.top(), item.left(), item.bottom(), item.right()) for item in selection
        ]

    def retrieveSelection(self):
        self._tv.ignoreReselect = True
        newSel = self._tv.selectionModel().selection()
        somethingSelected = False
        for top, left, bottom, right in self.topLeftBotRightSel:
            somethingSelected = True
            newSel.select(self._tm.index(top, left), self._tm.index(bottom, right))
        self._tv.selectionModel().select(newSel, QtCore.QItemSelectionModel.ClearAndSelect)
        self._tv.ignoreReselect = False
        self._tv.selEmptied.emit(somethingSelected)

    def highlightSelectedDeformers(self):
        selection = cmds.ls(sl=True)
        selection = set(cmds.ls(sl=True))
        intersect = selection.intersection(self.dataOfDeformer.driverNames)
        if intersect:
            nbRows = self._tm.rowCount()
            indices = [self.dataOfDeformer.driverNames.index(colName) for colName in intersect]
            # print intersect, indices
            newSel = self._tv.selectionModel().selection()
            newSel.clear()
            for index in indices:
                newSel.select(self._tm.index(0, index), self._tm.index(nbRows - 1, index))
                self._tv.showColumn(index)
            self._tv.selectionModel().select(newSel, QtCore.QItemSelectionModel.ClearAndSelect)
        else:
            self.dataOfDeformer.updateDisplayVerts([])

    def getRowColumnsSelected(self):
        sel = self._tv.selectionModel().selection()
        chunks = []
        for item in sel:
            chunks.append((item.top(), item.bottom(), item.left(), item.right()))
        return chunks

    # -------------------------------------------------------------
    # Mesh Paintable ----------------------------------------------
    # -------------------------------------------------------------
    def displayInfoPaintAttr(self, displayName):
        if displayName in self.dicDisplayNames:
            print(self.dicDisplayNames[displayName])

    # -------------------------------------------------------------
    # Misc --------------------------------------------------------
    # -------------------------------------------------------------
    def changeTypeOfData(self, ind):
        UvsEnabled = False
        if ind == 0:  # skinCluster
            self.dataOfDeformer = DataOfSkin(
                useShortestNames=self.useShortestNames,
                hideZeroColumn=self.hideZeroColumn,
                mainWindow=self,
                createDisplayLocator=self.useDisplayLocator,
            )
            self.problemVertsBTN.setEnabled(True)
        elif ind == 1:  # blendShape
            self.dataOfDeformer = DataOfBlendShape(
                mainWindow=self, createDisplayLocator=self.useDisplayLocator
            )
            self.problemVertsBTN.setEnabled(False)
        elif ind == 2:  # deformers
            self.dataOfDeformer = DataOfDeformers(
                mainWindow=self, createDisplayLocator=self.useDisplayLocator
            )
            self.problemVertsBTN.setEnabled(False)
            UvsEnabled = True
        self._tm.update(self.dataOfDeformer)
        self.uiUVsSettingWDG.setEnabled(UvsEnabled)

    def get_data_frame(self):
        with GlobalContext(message="get_data_frame"):
            self.dataOfDeformer.getAllData()
        return self.dataOfDeformer

    # --------------------------------------------------------------
    # Table UI functions  ------------------------------------------
    # --------------------------------------------------------------
    def setColumnVisSize(self):
        if self.dataOfDeformer.columnCount:
            for i in range(self.dataOfDeformer.columnCount):
                self._tv.setColumnWidth(i, self.colWidth)
            self._tv.setColumnWidth(i + 1, self.colWidth + 10)  # sum column
        self.hideColumns()

    def hideColumns(self):
        if self.hideZeroColumn:
            for ind in self.dataOfDeformer.hideColumnIndices:
                self._tv.hideColumn(ind)

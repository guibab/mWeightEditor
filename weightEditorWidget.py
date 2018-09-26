"""
import __main__
self = __main__.weightEditor
"""
from Qt import QtGui, QtCore, QtWidgets

# import shiboken2 as shiboken
from functools import partial
from maya import cmds, OpenMaya
import blurdev

from studio.gui.resource import Icons
from tools.skinData import DataOfSkin
from tools.tableWidget import TableView, TableModel
from tools.spinnerSlider import ValueSettingWE, ButtonWithValue
from tools.utils import GlobalContext

_icons = {
    "lock": Icons.getIcon(r"icons8\Android_L\PNG\48\Very_Basic\lock-48"),
    "unlock": Icons.getIcon(r"icons8\Android_L\PNG\48\Very_Basic\unlock-48"),
    "refresh": Icons.getIcon("refresh"),
}

styleSheet = """
QWidget {
    background:  #aba8a6;
    color:black;
    selection-background-color: #a0a0ff;
}
QCheckBox:hover
{
  background:rgb(120, 120, 120); 
}
QMenu::item:disabled {
    color:grey;
    font: italic;
}
QMenu::item:selected  {
    background-color:rgb(120, 120, 120);  
}
QPushButton {
    color:  black;
}
QPushButton:checked{
    background-color: rgb(100, 100, 100);
    color:white;
    border: none; 
}
QPushButton:hover{  
    background-color: grey; 
    border-style: outset;  
}
QPushButton:pressed {
    background-color: rgb(130, 130, 130);
    color:white;
    border-style: inset;
}
QPushButton:disabled {
    font:italic;
    color:grey;
    }
TableView {
     selection-background-color: #a0a0ff;
     background : #aba8a6;
     color: black;
     selection-color: black;
     border : 0px;
 }
QTableView QTableCornerButton::section {
    background:  transparent;
    border : 1px solid black;
}
 
TableView::section {
    background-color: #878787;
    color: black;
    border : 1px solid black;
}
QHeaderView::section {
    background-color: #878787;
    color: black;
    border : 1px solid black;
}
VertHeaderView{
    color: black;
    border : 0px solid black;
}
HorizHeaderView{
    color: black;
    border : 0px solid black;
}
"""
###################################################################################
#
#   the window
#
###################################################################################


class SkinWeightWin(QtWidgets.QDialog):
    """
    A simple test widget to contain and own the model and table.
    """

    colWidth = 30
    maxWidthCentralWidget = 340

    def addMinButton(self):
        # self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowMinimizeButtonHint)
        self.setWindowFlags(QtCore.Qt.Window)

    def __init__(self, parent=None):
        super(SkinWeightWin, self).__init__(parent)
        """
        self.setFloating (True)
        self.setAllowedAreas( QtCore.Qt.DockWidgetAreas ())
        self.isDockable = False
        """
        import __main__

        __main__.__dict__["weightEditor"] = self

        if not cmds.pluginInfo("blurSkin", query=True, loaded=True):
            cmds.loadPlugin("blurSkin")
        blurdev.gui.loadUi(__file__, self)

        # QtWidgets.QWidget.__init__(self, parent)
        self.buildRCMenu()

        self.dataOfSkin = DataOfSkin(
            useShortestNames=self.useShortestNames, hideZeroColumn=self.hideZeroColumn
        )
        self.get_data_frame()
        self.createWindow()
        self.setStyleSheet(styleSheet)

        self.addCallBacks()
        self.setWindowDisplay()

    def addCallBacks(self):
        self.refreshSJ = cmds.scriptJob(event=["SelectionChanged", self.refresh])
        # self.listJobEvents =[refreshSJ]
        sceneUpdateCallback = OpenMaya.MSceneMessage.addCallback(
            OpenMaya.MSceneMessage.kBeforeNew, self.deselectAll
        )  # kSceneUpdate
        self.close_callback = [sceneUpdateCallback]
        self.close_callback.append(
            OpenMaya.MSceneMessage.addCallback(OpenMaya.MSceneMessage.kBeforeOpen, self.deselectAll)
        )

    def deleteCallBacks(self):
        # for jobNum in self.listJobEvents   : cmds.scriptJob( kill=jobNum, force=True)
        self.dataOfSkin.deleteDisplayLocator()
        cmds.scriptJob(kill=self.refreshSJ, force=True)
        for callBck in self.close_callback:
            OpenMaya.MSceneMessage.removeCallback(callBck)

    def deselectAll(self, *args):
        self._tm.beginResetModel()
        self.dataOfSkin.clearData()
        self._tm.endResetModel()

    def toggleZeroColumn(self, checked):
        cmds.optionVar(intValue=["hideZeroColumn", checked])
        self.hideZeroColumn = checked
        for ind in self.dataOfSkin.hideColumnIndices:
            if self.hideZeroColumn:
                self._tv.hideColumn(ind)
            else:
                self._tv.showColumn(ind)

    def buildRCMenu(self):
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
            cmds.optionVar(q="hideZeroColumn") if cmds.optionVar(exists="hideZeroColumn") else True
        )
        # -------------------
        self.popMenu = QtWidgets.QMenu(self)

        resizeAction = self.popMenu.addAction("resize to minimum (MiddleClick)")
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

        # autoPruneAction = self.popMenu.addAction("auto Prune")
        # autoPruneAction.setCheckable (True)
        # autoPruneAction.setChecked ( True )
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.showMenu)

    def autoPruneChecked(self, checked):
        cmds.optionVar(intValue=["autoPrune", checked])
        self.autoPrune = checked
        self.popMenu.close()

    def useShortestNameChecked(self, checked):
        cmds.optionVar(intValue=["useShortestNames", checked])
        self.useShortestNames = checked
        self.dataOfSkin.useShortestNames = checked
        self.dataOfSkin.getShortNames()
        self.popMenu.close()

    def showMenu(self, pos):
        chd = self.childAt(pos)
        for (
            widgetName,
            widg,
        ) in self.__dict__.iteritems():  # for name, age in list.items():  (for Python 3.x)
            if widg == chd:
                # print widgetName
                break
        # print widgetName
        if widgetName == "topButtonsWidget":
            self.popMenu.exec_(self.mapToGlobal(pos))

    def setWindowDisplay(self):
        self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.Tool)
        self.setWindowTitle("Weight Editor")
        self.refreshPosition()
        self.show()

    def resizeToMinimum(self):
        nbShown = 0
        for ind in range(self._tv.HHeaderView.count()):
            if not self._tv.HHeaderView.isSectionHidden(ind):
                nbShown += 1
        wdth = self._tv.VHeaderView.width() + nbShown * self.colWidth + 50
        self.resize(wdth, self.height())

    def mousePressEvent(self, event):
        # print "click"
        if event.button() == QtCore.Qt.MidButton:
            self.resizeToMinimum()
        elif event.button() == QtCore.Qt.LeftButton:
            self._tv.clearSelection()
        super(SkinWeightWin, self).mousePressEvent(event)

    def refreshPosition(self):
        vals = cmds.optionVar(q="SkinWeightWindow")
        if vals:
            self.move(vals[0], vals[1])
            self.resize(vals[2], vals[3])

    def closeEvent(self, event):
        self.deleteCallBacks()

        pos = self.pos()
        size = self.size()
        cmds.optionVar(clearArray="SkinWeightWindow")
        for el in pos.x(), pos.y(), size.width(), size.height():
            cmds.optionVar(intValueAppend=("SkinWeightWindow", el))
        self._tv.deleteLater()
        # self.headerView.deleteLater()
        super(SkinWeightWin, self).closeEvent(event)

    def keyPressEvent(self, event):
        theKeyPressed = event.key()
        ctrlPressed = event.modifiers() == QtCore.Qt.ControlModifier

        if ctrlPressed and event.key() == QtCore.Qt.Key_Z:
            self.storeSelection()
            self._tm.beginResetModel()

            self.dataOfSkin.callUndo()

            self._tm.endResetModel()
            self.retrieveSelection()

            # super(SkinWeightWin, self).keyPressEvent(event)
            return
        super(SkinWeightWin, self).keyPressEvent(event)

    def addButtonsDirectSet(self, lstBtns):
        theCarryWidget = QtWidgets.QWidget()

        carryWidgLayoutlayout = QtWidgets.QHBoxLayout(theCarryWidget)
        carryWidgLayoutlayout.setContentsMargins(40, 0, 0, 0)
        carryWidgLayoutlayout.setSpacing(0)

        for theVal in lstBtns:
            newBtn = QtWidgets.QPushButton("{0:.0f}".format(theVal))

            newBtn.clicked.connect(self.prepareToSetValue)
            newBtn.clicked.connect(partial(self.doAddValue, theVal / 100.0))
            newBtn.clicked.connect(self.dataOfSkin.postSkinSet)

            carryWidgLayoutlayout.addWidget(newBtn)
        theCarryWidget.setMaximumSize(self.maxWidthCentralWidget, 14)

        return theCarryWidget

    def changeLock(self, val):
        if val:
            self.lockBTN.setIcon(_icons["lock"])
        else:
            self.lockBTN.setIcon(_icons["unlock"])
        self.unLock = not val

    def createWindow(self):
        theLayout = self.layout()  # QtWidgets.QVBoxLayout(self)
        theLayout.setContentsMargins(10, 10, 10, 10)
        theLayout.setSpacing(3)
        self.addPercentage = False

        topButtonsLay = self.topButtonsWidget.layout()

        self._tm = TableModel(self)
        self._tm.update(self.dataOfSkin)

        self._tv = TableView(self, colWidth=self.colWidth)
        self._tv.setModel(self._tm)
        # self._tm._tv = self._tv
        self.unLock = True
        self.lockBTN.setIcon(_icons["unlock"])
        self.lockBTN.setMaximumSize(24, 24)
        self.lockBTN.setCheckable(True)
        self.lockBTN.setChecked(False)
        self.lockBTN.toggled.connect(self.changeLock)
        self.lockBTN.setText("")

        self.valueSetter = ValueSettingWE(self)  # ProgressItem("BlendShape", szrad = 0, value = 0)

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

        topButtonsLay.addSpacing(10)
        topButtonsLay.addLayout(Hlayout2)
        self.widgetAbs.hide()
        topButtonsLay.addLayout(Hlayout)
        topButtonsLay.addSpacing(10)

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

        for nm in ["copy", "paste", "swap"]:
            self.__dict__[nm + "BTN"].setEnabled(False)
            self.__dict__[nm + "BTN"].hide()
        # -----------------------------------------------------------
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

        for uiName in ["reassignLocallyBTN", "averageBTN", "widgetAbs", "widgetAdd", "valueSetter"]:
            theUI = self.__dict__[uiName]
            theUI.setEnabled(False)
            self._tv.selEmptied.connect(theUI.setEnabled)
        cmds.evalDeferred(self.deferredBtns)

    def selProbVerts(self):
        vtx = self.dataOfSkin.fixAroundVertices(tolerance=self.problemVertsBTN.precision)
        selVertices = self.dataOfSkin.orderMelList(vtx)
        inList = ["{1}.vtx[{0}]".format(el, self.dataOfSkin.deformedShape) for el in selVertices]
        cmds.select(inList)

    def deferredBtns(self):
        for nm in ["abs", "add", "addPerc"]:
            self.__dict__[nm + "BTN"].setAutoExclusive(True)
        self.addBTN.setChecked(True)
        self.reassignLocallyBTN.setMinimumHeight(24)
        self.averageBTN.setMinimumHeight(24)
        # self.addPercBTN.setEnabled(False)

    def reassignLocally(self):
        chunks = self.getRowColumnsSelected()
        if chunks:
            actualyVisibleColumns = [
                indCol
                for indCol in self.dataOfSkin.hideColumnIndices
                if not self._tv.HHeaderView.isSectionHidden(indCol)
            ]
            self.storeSelection()
            self._tm.beginResetModel()

            self.dataOfSkin.prepareValuesforSetSkinData(chunks, actualyVisibleColumns)
            self.dataOfSkin.reassignLocally()
            self.dataOfSkin.postSkinSet()

            self._tm.endResetModel()
            self.retrieveSelection()

    def pruneWeights(self):
        chunks = self.getRowColumnsSelected()
        if not chunks:
            chunks = [(0, self.dataOfSkin.rowCount - 1, 0, self.dataOfSkin.columnCount - 1)]
        actualyVisibleColumns = []

        self.storeSelection()
        self._tm.beginResetModel()

        self.dataOfSkin.prepareValuesforSetSkinData(chunks, actualyVisibleColumns)
        self.dataOfSkin.pruneWeights(self.pruneWghtBTN.precisionValue / 100.0)
        self.dataOfSkin.postSkinSet()

        self._tm.endResetModel()
        self.retrieveSelection()

    def doNormalize(self):
        chunks = self.getRowColumnsSelected()
        if not chunks:
            chunks = [(0, self.dataOfSkin.rowCount - 1, 0, self.dataOfSkin.columnCount - 1)]
        actualyVisibleColumns = []

        self.storeSelection()
        self._tm.beginResetModel()

        self.dataOfSkin.prepareValuesforSetSkinData(chunks, actualyVisibleColumns)
        self.dataOfSkin.normalize()
        self.dataOfSkin.postSkinSet()

        self._tm.endResetModel()
        self.retrieveSelection()

    def doAverage(self):
        self.prepareToSetValue()
        self.doAddValue(0, forceAbsolute=False, average=True)
        self.dataOfSkin.postSkinSet()

    def changeAddAbs(self, checked):
        self.widgetAbs.setVisible(False)
        self.widgetAdd.setVisible(False)
        self.widgetAbs.setVisible(checked)
        self.widgetAdd.setVisible(not checked)
        self.valueSetter.setAddMode(not checked)

    def changeAddPerc(self, checked):
        self.addPercentage = checked

    def smooth(self):
        cmds.blurSkinCmd(
            command="smooth", repeat=self.smoothBTN.precision, percentMvt=self.percentBTN.precision
        )

    def prepareToSetValue(self):
        # with GlobalContext (message = "prepareValuesforSetSkinData"):
        chunks = self.getRowColumnsSelected()

        actualyVisibleColumns = [
            indCol
            for indCol in self.dataOfSkin.hideColumnIndices
            if not self._tv.HHeaderView.isSectionHidden(indCol)
        ]
        if chunks:
            self.dataOfSkin.prepareValuesforSetSkinData(chunks, actualyVisibleColumns)
            return True
        return False

    def storeSelection(self):
        selection = self._tv.selectionModel().selection()
        self.topLeftBotRightSel = [
            (item.top(), item.left(), item.bottom(), item.right()) for item in selection
        ]

    def retrieveSelection(self):
        self._tv.ignoreReselect = True
        newSel = self._tv.selectionModel().selection()
        for top, left, bottom, right in self.topLeftBotRightSel:
            newSel.select(self._tm.index(top, left), self._tm.index(bottom, right))
        self._tv.selectionModel().select(newSel, QtCore.QItemSelectionModel.ClearAndSelect)
        self._tv.ignoreReselect = False

    def doAddValue(self, val, forceAbsolute=False, average=False):
        self.storeSelection()
        self._tm.beginResetModel()

        if self.valueSetter.addMode and not forceAbsolute:
            self.dataOfSkin.setSkinData(
                val, percent=self.addPercentage, autoPrune=self.autoPrune, average=average
            )
        else:
            self.dataOfSkin.absoluteVal(val)

        self._tm.endResetModel()
        self.retrieveSelection()

    def getRowColumnsSelected(self):
        sel = self._tv.selectionModel().selection()
        chunks = []
        for item in sel:
            chunks.append((item.top(), item.bottom(), item.left(), item.right()))
        return chunks

    def refreshBtn(self):
        self.storeSelection()
        self.refresh(force=True)
        self.retrieveSelection()

    def highlightSelectedDeformers(self):
        selection = cmds.ls(sl=True)
        selection = set(cmds.ls(sl=True))
        intersect = selection.intersection(self.dataOfSkin.driverNames)
        if intersect:
            nbRows = self._tm.rowCount()
            indices = [self.dataOfSkin.driverNames.index(colName) for colName in intersect]
            # print intersect, indices
            newSel = self._tv.selectionModel().selection()
            newSel.clear()
            for index in indices:
                newSel.select(self._tm.index(0, index), self._tm.index(nbRows - 1, index))
                self._tv.showColumn(index)
            self._tv.selectionModel().select(newSel, QtCore.QItemSelectionModel.ClearAndSelect)

    def refresh(self, force=False):
        if self.unLock or force:
            self._tm.beginResetModel()
            for ind in self.dataOfSkin.hideColumnIndices:
                self._tv.showColumn(ind)
            with GlobalContext(message="weightEdtior getAllData", doPrint=False):
                resultData = self.dataOfSkin.getAllData(force=force)
            self._tm.endResetModel()
            self.setColumnVisSize()
            if not resultData:
                self.highlightSelectedDeformers()
            self._tv.selEmptied.emit(False)
            self._tv.repaint()

    def setColumnVisSize(self):
        if self.dataOfSkin.columnCount:
            for i in range(self.dataOfSkin.columnCount):
                self._tv.setColumnWidth(i, self.colWidth)
            self._tv.setColumnWidth(i + 1, self.colWidth + 10)  # sum column
        self.hideColumns()

    def hideColumns(self):
        # self.dataOfSkin.getZeroColumns ()
        if self.hideZeroColumn:
            for ind in self.dataOfSkin.hideColumnIndices:
                self._tv.hideColumn(ind)
        # self._tv.headerView.setMaximumWidth(self.colWidth*len (self.dataOfSkin.usedDeformersIndices))

    def get_data_frame(self):
        with GlobalContext(message="get_data_frame"):
            self.dataOfSkin.getAllData()
        return self.dataOfSkin

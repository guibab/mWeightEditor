"""
import __main__
self = __main__.weightEditor
"""
from Qt import QtGui, QtCore, QtWidgets, QtCompat

# import shiboken2 as shiboken
from functools import partial
from maya import cmds, OpenMaya
import os
import blurdev
from blurdev.gui import Window

from studio.gui.resource import Icons
from tools.skinData import DataOfSkin
from tools.weightMapsData import DataOfBlendShape

from tools.tableWidget import TableView, TableModel
from tools.spinnerSlider import ValueSettingWE, ButtonWithValue
from tools.utils import (
    GlobalContext,
    addNameChangedCallback,
    removeNameChangedCallback,
    toggleBlockSignals,
)

# -------------------------------------------------------------------------------------------
# styleSheet and icons ---------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
def getIcon(iconNm):
    fileVar = os.path.realpath(__file__)
    uiFolder, filename = os.path.split(fileVar)
    iconPth = os.path.join(uiFolder, "img", iconNm + ".png")
    return QtGui.QIcon(iconPth)


_icons = {
    "lock": getIcon("lock-48"),
    "unlock": getIcon("unlock-48"),
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
    QComboBox {
        color:  black;
        border : 1px solid grey;
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
class SkinWeightWin(Window):
    colWidth = 30
    maxWidthCentralWidget = 340  # top widget max size -----------

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
        self.getOptionVars()
        self.buildRCMenu()

        # self.dataOfDeformer = DataOfBlendShape ()
        self.dataOfDeformer = DataOfSkin(
            useShortestNames=self.useShortestNames, hideZeroColumn=self.hideZeroColumn
        )

        self.get_data_frame()
        self.createWindow()
        self.setStyleSheet(styleSheet)

        self.addCallBacks()
        self.setWindowDisplay()

    # -----------------------------------------------------------------------------------------------------------
    # window events --------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------
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
        # self.headerView.deleteLater()
        super(SkinWeightWin, self).closeEvent(event)

    def keyPressEvent(self, event):
        theKeyPressed = event.key()
        ctrlPressed = event.modifiers() == QtCore.Qt.ControlModifier

        if ctrlPressed and event.key() == QtCore.Qt.Key_Z:
            self.storeSelection()
            self._tm.beginResetModel()

            self.dataOfDeformer.callUndo()

            self._tm.endResetModel()
            self.retrieveSelection()

            # super(SkinWeightWin, self).keyPressEvent(event)
            return
        super(SkinWeightWin, self).keyPressEvent(event)

    def mousePressEvent(self, event):
        # print "click"
        if event.button() == QtCore.Qt.MidButton:
            self.resizeToMinimum()
        elif event.button() == QtCore.Qt.LeftButton:
            self._tv.clearSelection()
        super(SkinWeightWin, self).mousePressEvent(event)

    # -----------------------------------------------------------------------------------------------------------
    # widget creation/edition  ---------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------
    def addMinButton(self):
        # self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowMinimizeButtonHint)
        self.setWindowFlags(QtCore.Qt.Window)

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

    def addButtonsDirectSet(self, lstBtns):
        theCarryWidget = QtWidgets.QWidget()

        carryWidgLayoutlayout = QtWidgets.QHBoxLayout(theCarryWidget)
        carryWidgLayoutlayout.setContentsMargins(40, 0, 0, 0)
        carryWidgLayoutlayout.setSpacing(0)

        for theVal in lstBtns:
            newBtn = QtWidgets.QPushButton("{0:.0f}".format(theVal))

            newBtn.clicked.connect(self.prepareToSetValue)
            newBtn.clicked.connect(partial(self.doAddValue, theVal / 100.0))
            newBtn.clicked.connect(self.dataOfDeformer.postSkinSet)

            carryWidgLayoutlayout.addWidget(newBtn)
        theCarryWidget.setMaximumSize(self.maxWidthCentralWidget, 14)

        return theCarryWidget

    def changeLock(self, val):
        if val:
            self.lockBTN.setIcon(_icons["lock"])
        else:
            self.lockBTN.setIcon(_icons["unlock"])
        self.unLock = not val

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
            self.__dict__[nm + "BTN"].setAutoExclusive(True)
        self.addBTN.setChecked(True)
        self.reassignLocallyBTN.setMinimumHeight(24)
        self.averageBTN.setMinimumHeight(24)
        # self.addPercBTN.setEnabled(False)

    def createWindow(self):
        theLayout = self.mainLayout

        theLayout.setContentsMargins(10, 10, 10, 10)
        theLayout.setSpacing(3)
        self.addPercentage = False

        topButtonsLay = self.topButtonsWidget.layout()

        self._tm = TableModel(self)
        self._tm.update(self.dataOfDeformer)

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

        # self.listInputs_CB.currentTextChanged.connect (self.displayInfoPaintAttr)
        self.listInputs_CB.currentIndexChanged.connect(self.changeTypeOfData)

        for uiName in ["reassignLocallyBTN", "averageBTN", "widgetAbs", "widgetAdd", "valueSetter"]:
            theUI = self.__dict__[uiName]
            theUI.setEnabled(False)
            self._tv.selEmptied.connect(theUI.setEnabled)
        cmds.evalDeferred(self.deferredBtns)

        # display the list of paintable attributes

        with toggleBlockSignals([self.listInputs_CB]):
            self.listInputs_CB.addItems(["skinCluster", "blendShape"])
        """
        if self.dataOfDeformer.deformedShape : 
            self.getListPaintableAttributes (self.dataOfDeformer.deformedShape)
        """

    # -----------------------------------------------------------------------------------------------------------
    # callBacks ------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------
    def renameCB(self, oldName, newName):
        if self.dataOfDeformer:
            self.dataOfDeformer.renameCB(oldName, newName)

    def addCallBacks(self):
        self.refreshSJ = cmds.scriptJob(event=["SelectionChanged", self.refresh])
        self.renameCallBack = addNameChangedCallback(self.renameCB)

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
        removeNameChangedCallback(self.renameCallBack)
        self.dataOfDeformer.deleteDisplayLocator()
        cmds.scriptJob(kill=self.refreshSJ, force=True)
        for callBck in self.close_callback:
            OpenMaya.MSceneMessage.removeCallback(callBck)

    def deselectAll(self, *args):
        # print "DESELECTALL"
        self._tm.beginResetModel()
        self.dataOfDeformer.clearData()
        self._tm.endResetModel()

    # -----------------------------------------------------------------------------------------------------------
    # right click menu -----------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------
    def buildRCMenu(self):
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

    # -----------------------------------------------------------------------------------------------------------
    # optionVars -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------
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

    def toggleZeroColumn(self, checked):
        cmds.optionVar(intValue=["hideZeroColumn", checked])
        self.hideZeroColumn = checked
        for ind in self.dataOfDeformer.hideColumnIndices:
            if self.hideZeroColumn:
                self._tv.hideColumn(ind)
            else:
                self._tv.showColumn(ind)

    def autoPruneChecked(self, checked):
        cmds.optionVar(intValue=["autoPrune", checked])
        self.autoPrune = checked
        self.popMenu.close()

    def useShortestNameChecked(self, checked):
        cmds.optionVar(intValue=["useShortestNames", checked])
        self.useShortestNames = checked
        self.dataOfDeformer.useShortestNames = checked
        self.dataOfDeformer.getShortNames()
        self.popMenu.close()

    # -----------------------------------------------------------------------------------------------------------
    # Refresh --------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------
    def refreshPosition(self):
        vals = cmds.optionVar(q="SkinWeightWindow")
        if vals:
            self.move(vals[0], vals[1])
            self.resize(vals[2], vals[3])

    def refreshBtn(self):
        self.storeSelection()
        self.refresh(force=True)
        self.retrieveSelection()

    def refreshPaintEditor(self):
        import __main__

        if (
            hasattr(__main__, "paintEditor")
            and __main__.paintEditor in QtWidgets.QApplication.instance().topLevelWidgets()
        ):
            __main__.paintEditor.refreshColorsAndLocks()

    def refresh(self, force=False):
        if self.unLock or force:
            self._tm.beginResetModel()
            for ind in self.dataOfDeformer.hideColumnIndices:
                self._tv.showColumn(ind)
            with GlobalContext(message="weightEdtior getAllData", doPrint=False):
                self.dataOfDeformer.updateDisplayVerts([])
                resultData = self.dataOfDeformer.getAllData(force=force)
            """
            sel = cmds.ls (sl=True, tr=True)
            if sel : 
                theSel = sel [0]
                theShape = cmds.listRelatives( theSel, noIntermediate=True, shapes = True) [0]
                self.getListPaintableAttributes (theShape)
            else : 
                self.listInputs_CB.clear()
            """

            self._tm.endResetModel()
            self.setColumnVisSize()
            if not resultData and self.dataOfDeformer.isSkinData:
                self.highlightSelectedDeformers()
            self._tv.selEmptied.emit(False)
            self._tv.repaint()

    # -----------------------------------------------------------------------------------------------------------
    # Functions ------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------
    def reassignLocally(self):
        chunks = self.getRowColumnsSelected()
        if chunks:
            actualyVisibleColumns = [
                indCol
                for indCol in self.dataOfDeformer.hideColumnIndices
                if not self._tv.HHeaderView.isSectionHidden(indCol)
            ]
            self.storeSelection()
            self._tm.beginResetModel()

            self.dataOfDeformer.preSettingValuesFn(chunks, actualyVisibleColumns)
            self.dataOfDeformer.reassignLocally()
            self.dataOfDeformer.postSkinSet()

            self._tm.endResetModel()
            self.retrieveSelection()

    def pruneWeights(self):
        chunks = self.getRowColumnsSelected()
        if not chunks:
            chunks = [(0, self.dataOfDeformer.rowCount - 1, 0, self.dataOfDeformer.columnCount - 1)]
        actualyVisibleColumns = []

        self.storeSelection()
        self._tm.beginResetModel()

        self.dataOfDeformer.preSettingValuesFn(chunks, actualyVisibleColumns)
        self.dataOfDeformer.pruneWeights(self.pruneWghtBTN.precisionValue / 100.0)
        self.dataOfDeformer.postSkinSet()

        self._tm.endResetModel()
        self.retrieveSelection()

    def doNormalize(self):
        chunks = self.getRowColumnsSelected()
        if not chunks:
            chunks = [(0, self.dataOfDeformer.rowCount - 1, 0, self.dataOfDeformer.columnCount - 1)]
        actualyVisibleColumns = []

        self.storeSelection()
        self._tm.beginResetModel()

        self.dataOfDeformer.preSettingValuesFn(chunks, actualyVisibleColumns)
        self.dataOfDeformer.normalize()
        self.dataOfDeformer.postSkinSet()

        self._tm.endResetModel()
        self.retrieveSelection()

    def doAverage(self):
        self.prepareToSetValue()
        self.doAddValue(0, forceAbsolute=False, average=True)
        self.dataOfDeformer.postSkinSet()

    def smooth(self):
        chunks = self.getRowColumnsSelected()
        if not chunks:
            chunks = [(0, self.dataOfDeformer.rowCount - 1, 0, self.dataOfDeformer.columnCount - 1)]
        self.dataOfDeformer.getConnectedBlurskinDisplay(disconnectWeightList=True)
        # convert to vertices or get the vertices
        self.dataOfDeformer.smoothSkin(
            chunks, repeat=self.smoothBTN.precision, percentMvt=self.percentBTN.precision
        )
        # cmds.blurSkinCmd (command = "smooth", repeat = self.smoothBTN.precision, percentMvt = self.percentBTN.precision)
        if self.dataOfDeformer.blurSkinNode and cmds.objExists(self.dataOfDeformer.blurSkinNode):
            cmds.delete(self.dataOfDeformer.blurSkinNode)

    def selProbVerts(self):
        vtx = self.dataOfDeformer.fixAroundVertices(tolerance=self.problemVertsBTN.precision)
        selVertices = self.dataOfDeformer.orderMelList(vtx)
        inList = [
            "{1}.vtx[{0}]".format(el, self.dataOfDeformer.deformedShape) for el in selVertices
        ]
        cmds.select(inList)

    # -----------------------------------------------------------------------------------------------------------
    # Basic set Values -----------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------
    def prepareToSetValue(self):
        # with GlobalContext (message = "preSettingValuesFn"):
        chunks = self.getRowColumnsSelected()
        actualyVisibleColumns = [
            indCol
            for indCol in self.dataOfDeformer.hideColumnIndices
            if not self._tv.HHeaderView.isSectionHidden(indCol)
        ]
        if chunks:
            self.dataOfDeformer.preSettingValuesFn(chunks, actualyVisibleColumns)
            return True
        return False

    def postSetValue(self):
        if self.dataOfDeformer.isSkinData:
            return self.dataOfDeformer.postSkinSet()

    def doAddValue(self, val, forceAbsolute=False, average=False):
        self.storeSelection()
        self._tm.beginResetModel()

        if self.valueSetter.addMode and not forceAbsolute:
            if self.dataOfDeformer.isSkinData:
                self.dataOfDeformer.setSkinData(
                    val, percent=self.addPercentage, autoPrune=self.autoPrune, average=average
                )
            else:
                self.dataOfDeformer.doAdd(
                    val, percent=self.addPercentage, autoPrune=self.autoPrune, average=average
                )
                # print "to implement Add, use Absolute"
        else:
            self.dataOfDeformer.absoluteVal(val)

        self._tm.endResetModel()
        self.retrieveSelection()

    # -----------------------------------------------------------------------------------------------------------
    # Selection ------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------
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

    # -----------------------------------------------------------------------------------------------------------
    # Mesh Paintable -------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------
    def getListPaintableAttributes(self, theNodeShape):
        listDeformersTypes = cmds.nodeType("geometryFilter", derived=True, isTypeName=True)
        listShapesTypes = cmds.nodeType("shape", derived=True, isTypeName=True)

        paintableItems = cmds.artBuildPaintMenu(theNodeShape).split(" ")

        lstDeformers = []
        lstShapes = []
        lstOthers = []

        blendShapes = set()
        self.dicDisplayNames = {}
        toSel = ""
        for itemToPaint in paintableItems:
            if not itemToPaint:
                continue
            splt = itemToPaint.split(".")
            nodeType, nodeName, attr = splt[:3]
            nodeNameShort = nodeName.split("|")[-1]
            displayName = "-".join([nodeNameShort, attr])

            if nodeType == "skinCluster":
                toSel = displayName
            if nodeType == "blendShape":
                blendShapes.add(nodeName)
                continue
            self.dicDisplayNames[displayName] = nodeName + "." + attr
            if nodeType in listDeformersTypes:
                lstDeformers.append(displayName)
            elif nodeType in lstShapes:
                lstShapes.append(displayName)
            else:
                lstOthers.append(displayName)
        lstBlendShapes = []
        # deal with blendShaps
        for BSnode in blendShapes:
            lsGeomsOrig = cmds.blendShape(BSnode, q=True, geometry=True)
            lsGeomsIndicesOrig = cmds.blendShape(BSnode, q=True, geometryIndices=True)
            if theNodeShape in lsGeomsOrig:
                # get the index of the node in the blendShape
                ind = lsGeomsIndicesOrig[lsGeomsOrig.index(theNodeShape)]

                displayName = "-".join([BSnode, "baseWeights"])
                self.dicDisplayNames[displayName] = "{}.inputTarget[{}].baseWeights".format(
                    BSnode, ind
                )
                lstBlendShapes.append(displayName)

                # get the alias
                listAlias = cmds.aliasAttr(BSnode, q=True)
                if listAlias == None:
                    listAliasIndices = cmds.getAttr(
                        BSnode + ".inputTarget[{}].inputTargetGroup".format(ind), mi=True
                    )
                    for channelIndex in listAliasIndices:
                        displayName = "-".join([BSnode, "targetWeights_{}".format(channelIndex)])
                        self.dicDisplayNames[
                            displayName
                        ] = "{}.inputTarget[{}].inputTargetGroup[{}].targetWeights".format(
                            BSnode, ind, channelIndex
                        )
                        lstBlendShapes.append(displayName)
                else:
                    listAlias_names = sorted(
                        [
                            (int(listAlias[i * 2 + 1].split("[")[1][:-1]), listAlias[i * 2])
                            for i in range(len(listAlias) / 2)
                        ]
                    )
                    for channelIndex, nmAlias in listAlias_names:
                        displayName = "-".join([BSnode, nmAlias])
                        self.dicDisplayNames[
                            displayName
                        ] = "{}.inputTarget[{}].inputTargetGroup[{}].targetWeights".format(
                            BSnode, ind, channelIndex
                        )
                        lstBlendShapes.append(displayName)
        with toggleBlockSignals([self.listInputs_CB]):
            self.listInputs_CB.addItems(["skinCluster", "blendShape"])
            """
            self.listInputs_CB.clear()
            met = QtGui.QFontMetrics (self.listInputs_CB.font())
            longest = 0
            for displayName in lstBlendShapes+lstDeformers+lstOthers+lstShapes :
                width = met.width(displayName)
                if width>longest : longest = width

                newItem = self.listInputs_CB.addItem (displayName)
                #newItem.setItemData ()
            self.listInputs_CB.view().setMinimumWidth(longest + 10)

            if toSel : self.listInputs_CB.setCurrentText (toSel)
            else : self.listInputs_CB.setCurrentIndex(-1)
            """

    def displayInfoPaintAttr(self, displayName):
        if displayName in self.dicDisplayNames:
            print self.dicDisplayNames[displayName]

    # -----------------------------------------------------------------------------------------------------------
    # Misc -----------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------
    def changeTypeOfData(self, ind):
        if ind == 0:  # skinCluster
            self.dataOfDeformer = DataOfSkin(
                useShortestNames=self.useShortestNames, hideZeroColumn=self.hideZeroColumn
            )
            self.problemVertsBTN.setEnabled(True)
        elif ind == 1:  # blendShape
            self.dataOfDeformer = DataOfBlendShape()
            self.problemVertsBTN.setEnabled(False)
        self._tm.update(self.dataOfDeformer)

    def get_data_frame(self):
        with GlobalContext(message="get_data_frame"):
            self.dataOfDeformer.getAllData()
        return self.dataOfDeformer

    # -----------------------------------------------------------------------------------------------------------
    # Table UI functions  --------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------
    def setColumnVisSize(self):
        if self.dataOfDeformer.columnCount:
            for i in range(self.dataOfDeformer.columnCount):
                self._tv.setColumnWidth(i, self.colWidth)
            self._tv.setColumnWidth(i + 1, self.colWidth + 10)  # sum column
        self.hideColumns()

    def hideColumns(self):
        # self.dataOfDeformer.getZeroColumns ()
        if self.hideZeroColumn:
            for ind in self.dataOfDeformer.hideColumnIndices:
                self._tv.hideColumn(ind)
        # self._tv.headerView.setMaximumWidth(self.colWidth*len (self.dataOfDeformer.usedDeformersIndices))

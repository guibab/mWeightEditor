from Qt import QtGui, QtCore, QtWidgets

# import shiboken2 as shiboken
from maya import cmds
import blurdev

from tools.skinData import DataOfSkin
from tools.tableWidget import TableView, TableModel
from tools.spinnerSlider import ValueSetting
from tools.utils import GlobalContext

styleSheet = """
QWidget {
    background:  #aba8a6;
    color:black;
}

QMenu::item:disabled {
    color:grey;
    font: italic;
}
QPushButton {
    color:  black;
}
QPushButton:checked{
    background-color: rgb(80, 80, 80);
    color:white;
    border: none; 
}
QPushButton:hover{  
    background-color: grey; 
    border-style: outset;  
}  

TableView {
     selection-background-color: #a0a0ff;
     background : #aba8a6;
     color: black;
     selection-color: black;
     border : 0px;
 }
QTableView QTableCornerButton::section {
    background:  #878787;
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
MyHeaderView{
    color: black;
    border : 0px solid black;
}
"""
"""
MyHeaderView{
    background-color: #878787;
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

    def __init__(self, parent=None):
        super(SkinWeightWin, self).__init__(parent)
        """
        self.setFloating (True)
        self.setAllowedAreas( QtCore.Qt.DockWidgetAreas ())
        self.isDockable = False
        """
        import __main__

        __main__.__dict__["weightEditor"] = self

        blurdev.gui.loadUi(__file__, self)

        # QtWidgets.QWidget.__init__(self, parent)
        self.dataOfSkin = DataOfSkin()
        self.get_data_frame()
        self.createWindow()
        self.setStyleSheet(styleSheet)

        refreshSJ = cmds.scriptJob(event=["SelectionChanged", self.refresh])
        self.listJobEvents = [refreshSJ]

        self.setWindowDisplay()

    def setWindowDisplay(self):
        self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.Tool)
        self.setWindowTitle("Weight Editor")
        self.refreshPosition()
        self.show()

    def mousePressEvent(self, event):
        # print "click"
        self._tv.clearSelection()
        super(SkinWeightWin, self).mousePressEvent(event)

    def refreshPosition(self):
        vals = cmds.optionVar(q="SkinWeightWindow")
        if vals:
            self.move(vals[0], vals[1])
            self.resize(vals[2], vals[3])

    def closeEvent(self, event):
        for jobNum in self.listJobEvents:
            cmds.scriptJob(kill=jobNum, force=True)
        pos = self.pos()
        size = self.size()
        cmds.optionVar(clearArray="SkinWeightWindow")
        for el in pos.x(), pos.y(), size.width(), size.height():
            cmds.optionVar(intValueAppend=("SkinWeightWindow", el))
        # self.headerView.deleteLater()
        super(SkinWeightWin, self).closeEvent(event)

    def keyPressEvent(self, event):
        theKeyPressed = event.key()
        ctrlPressed = event.modifiers() == QtCore.Qt.ControlModifier

        if ctrlPressed and event.key() == QtCore.Qt.Key_Z:
            self.dataOfSkin.callUndo()
            # super(SkinWeightWin, self).keyPressEvent(event)
            return
        super(SkinWeightWin, self).keyPressEvent(event)

    def createWindow(self):
        theLayout = self.layout()  # QtWidgets.QVBoxLayout(self)
        theLayout.setContentsMargins(10, 10, 10, 10)
        theLayout.setSpacing(3)

        self._tm = TableModel(self)
        self._tm.update(self.dataOfSkin)

        self._tv = TableView(self, colWidth=self.colWidth)
        self._tv.setModel(self._tm)
        # self._tm._tv = self._tv

        self.refreshBTN.clicked.connect(self.refreshBtn)

        self.valueSetter = ValueSetting(self)  # ProgressItem("BlendShape", szrad = 0, value = 0)
        Hlayout = QtWidgets.QHBoxLayout(self)
        Hlayout.setContentsMargins(0, 0, 0, 0)
        Hlayout.setSpacing(0)
        Hlayout.addWidget(self.valueSetter)
        self.valueSetter.setMaximumWidth(300)

        theLayout.addLayout(Hlayout)
        theLayout.addWidget(self._tv)

        for i in range(self.dataOfSkin.columnCount):
            self._tv.setColumnWidth(i, self.colWidth)
        self.hideColumns()

    def prepareToSetValue(self):
        # with GlobalContext (message = "prepareValuesforSetSkinData"):
        chunks = self.getRowColumnsSelected()
        if chunks:
            self.dataOfSkin.prepareValuesforSetSkinData(chunks)
            return True
        return False

    def storeSelection(self):
        selection = self._tv.selectionModel().selection()
        self.topLeftBotRightSel = [
            (item.top(), item.left(), item.bottom(), item.right()) for item in selection
        ]

    def retrieveSelection(self):
        newSel = self._tv.selectionModel().selection()
        for top, left, bottom, right in self.topLeftBotRightSel:
            newSel.select(self._tm.index(top, left), self._tm.index(bottom, right))
        self._tv.selectionModel().select(newSel, QtCore.QItemSelectionModel.ClearAndSelect)

    def doAddValue(self, val):
        self.storeSelection()
        self._tm.beginResetModel()

        self.dataOfSkin.setSkinData(val)

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
        self.refresh()
        self.retrieveSelection()

    def refresh(self):
        self._tm.beginResetModel()
        for ind in self.dataOfSkin.hideColumnIndices:
            self._tv.showColumn(ind)
        self.dataOfSkin.getAllData()
        self._tm.endResetModel()
        for i in range(self.dataOfSkin.columnCount):
            self._tv.setColumnWidth(i, self.colWidth)
        self.hideColumns()

    def hideColumns(self):
        # self.dataOfSkin.getZeroColumns ()
        for ind in self.dataOfSkin.hideColumnIndices:
            self._tv.hideColumn(ind)
        # self._tv.headerView.setMaximumWidth(self.colWidth*len (self.dataOfSkin.usedDeformersIndices))

    def get_data_frame(self):
        with GlobalContext(message="get_data_frame"):
            self.dataOfSkin.getAllData()
        return self.dataOfSkin

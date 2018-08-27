from Qt import QtGui, QtCore, QtWidgets
from functools import partial
from maya import cmds
import numpy as np


class TableModel(QtCore.QAbstractTableModel):
    def __init__(self, parent=None, *args):
        super(TableModel, self).__init__(parent)
        self.datatable = None
        self.brownBrush = QtGui.QBrush(QtGui.QColor(130, 130, 90))
        self.greyBrush = QtGui.QBrush(QtGui.QColor(140, 140, 140))
        self.greyDarkerBrush = QtGui.QBrush(QtGui.QColor(80, 80, 80))
        self.sumBrush = QtGui.QBrush(QtGui.QColor(100, 100, 100))
        self.redBrush = QtGui.QBrush(QtGui.QColor(150, 100, 100))
        self.whiteBrush = QtGui.QBrush(QtGui.QColor(200, 200, 200))

    def update(self, dataIn):
        print "Updating Model"
        self.datatable = dataIn
        # print 'Datatable : {0}'.format(self.datatable)

    def rowCount(self, parent=QtCore.QModelIndex()):
        return self.datatable.rowCount

    def columnCount(self, parent=QtCore.QModelIndex()):
        return self.datatable.columnCount + 1

    def columnNames(self):
        return self.datatable.shortDriverNames

    def fullColumnNames(self):
        return self.datatable.driverNames

    def data(self, index, role=QtCore.Qt.DisplayRole):
        # print 'Data Call'
        # print index.column(), index.row()
        if role == QtCore.Qt.DisplayRole:
            # return QtCore.QVariant(str(self.datatable.iget_value(i, j)))
            # return '{0:.2f}'.format(self.realData(index))
            return round(self.realData(index) * 100, 1)
        elif role == QtCore.Qt.EditRole:
            ff = self.realData(index) * 100
            return "{0:.3f}".format(ff).rstrip("0") + "0"[0 : (ff % 1 == 0)]
        elif role == QtCore.Qt.TextAlignmentRole:
            return QtCore.Qt.AlignCenter  # | QtCore.Qt.AlignVCenter)
        elif role == QtCore.Qt.BackgroundRole:
            if self.isSumColumn(index):
                return (
                    self.sumBrush if round(self.realData(index) * 100, 1) == 100 else self.redBrush
                )
            elif self.isLocked(index):
                return self.greyBrush
            elif self.realData(index) != 0.0:
                return self.brownBrush
        elif role == QtCore.Qt.ForegroundRole:
            if self.isSumColumn(index):
                return self.whiteBrush
            elif self.isLocked(index):
                return self.greyDarkerBrush
        else:
            return None

    def setData(self, index, value, role=QtCore.Qt.EditRole):
        # print "SET DATA 3"
        # super(TableModel, self).setData( index, value, role)
        row = index.row()
        column = index.column()
        print row, column, value

        # now set the value
        self.parent().prepareToSetValue()
        self.parent().doAddValue(value / 100.0, forceAbsolute=True)
        self.datatable.postSkinSet()
        return True

    def isLocked(self, index):
        row = index.row()
        column = index.column()
        return self.datatable.isLocked(row, column)

    def realData(self, index):
        row = index.row()
        column = index.column()
        return self.datatable.getValue(row, column)

    def isSumColumn(self, index):
        column = index.column()
        return column >= self.datatable.nbDrivers

    def headerData(self, col, orientation, role):
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return self.datatable.driverNames[col]
            else:
                return self.datatable.rowText[col]
        elif role == QtCore.Qt.TextAlignmentRole:
            return QtCore.Qt.AlignCenter  # | QtCore.Qt.AlignVCenter)
        else:
            return None

    def getColumnText(self, col):
        try:
            return self.datatable.shortDriverNames[col]
        except:
            return "total"

    def getRowText(self, row):
        return self.datatable.rowText[row]

    def getColumnSide(self, col):
        try:
            driverName = self.datatable.driverNames[col]
            for letter in "LRM":
                if "_{0}_".format(letter) in driverName:
                    return letter
            return "X"
        except:
            return "X"

    def flags(self, index):
        if not index.isValid():
            return QtCore.Qt.ItemIsEnabled
        # sresult = super(TableModel,self).flags(index)
        # result = sresult | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable
        column = index.column()
        if column == self.datatable.nbDrivers:  # sum column
            result = QtCore.Qt.ItemIsEnabled
        elif self.isLocked(index):
            result = QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
        else:
            result = QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable
        return QtCore.Qt.ItemFlags(result)


class HighlightDelegate(QtWidgets.QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        # print "createEditor"
        editor = QtWidgets.QDoubleSpinBox(parent)
        editor.setStyleSheet("QDoubleSpinBox { background-color: yellow; color : black; }")
        editor.setMaximum(100)
        editor.setMinimum(0)
        editor.setMinimumWidth(50)
        editor.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        return editor

    """
    def setEditorData(self, editor, index):
        print "setEditorData"
        model = index.model()
        realData = model.realData (index)        
        editor.setText (str(realData))
        editor.selectAll() 
        print realData
        super(HighlightDelegate, self).setEditorData(editor, index)
    
    def closeEditor(self, editor, hint=None):
        print "closeEditor"
        super(HighlightDelegate, self).closeEditor(editor, hint)

    def commitData(self, editor):
        print "commitData"
        super(HighlightDelegate, self).commitData(editor)
    """

    def paint(self, painter, rawOption, index):
        if not index.isValid():
            return super(HighlightDelegate, self).paint(painter, rawOption, index)
        model = index.model()
        realData = model.realData(index)
        # theData = model.data(index)
        if realData == 0.00:
            return super(HighlightDelegate, self).paint(painter, rawOption, index)
        option = QtWidgets.QStyleOptionViewItem(rawOption)
        pal = option.palette
        # bg = model.data(index, QtCore.Qt.BackgroundRole)
        pal.setColor(pal.currentColorGroup(), QtGui.QPalette.Highlight, QtGui.QColor(140, 140, 235))
        # pal.setColor(pal.currentColorGroup(), QtGui.QPalette.HighlightedText, pal.text().color())

        return super(HighlightDelegate, self).paint(painter, option, index)


class VertHeaderView(QtWidgets.QHeaderView):
    def __init__(self, parent=None):
        super(VertHeaderView, self).__init__(QtCore.Qt.Vertical, parent)
        self.setMinimumWidth(20)

        self.setSectionsClickable(True)
        self.setHighlightSections(True)
        self.setSectionResizeMode(QtWidgets.QHeaderView.Fixed)

        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.showMenu)

        self.regularBG = QtGui.QBrush(QtGui.QColor(130, 130, 130))
        self.greyBG = QtGui.QBrush(QtGui.QColor(100, 100, 100))

    def showMenu(self, pos):
        popMenu = QtWidgets.QMenu(self)
        selectionIsEmpty = self.selectionModel().selection().isEmpty()

        selAction = popMenu.addAction("select vertices")
        selAction.triggered.connect(self.selectVerts)
        selAction.setEnabled(not selectionIsEmpty)
        popMenu.addSeparator()

        lockAction = popMenu.addAction("lock selected")
        lockAction.triggered.connect(self.lockSelectedRows)
        lockAction.setEnabled(not selectionIsEmpty)

        lockAllButSelAction = popMenu.addAction("lock all but selected")
        lockAllButSelAction.triggered.connect(self.lockAllButSelectedRows)
        lockAllButSelAction.setEnabled(not selectionIsEmpty)

        unlockAction = popMenu.addAction("unlock selected")
        unlockAction.triggered.connect(self.unlockSelectedRows)
        unlockAction.setEnabled(not selectionIsEmpty)

        clearLocksAction = popMenu.addAction("clear all Locks")
        clearLocksAction.triggered.connect(self.clearLocks)
        popMenu.exec_(self.mapToGlobal(pos))

    def paintSection(self, painter, rect, index):
        theBGBrush = self.greyBG if self.model().datatable.isRowLocked(index) else self.regularBG
        if self.model().datatable.isRowLocked(index):
            text = self.model().getRowText(index)
            painter.save()
            painter.setBrush(self.greyBG)
            pen = QtGui.QPen(QtGui.QColor(0, 0, 0))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRect(rect)
            painter.restore()
            painter.drawText(rect, QtCore.Qt.AlignCenter, text)
            return
        else:
            return super(VertHeaderView, self).paintSection(painter, rect, index)

    def getSelectedRows(self):
        """
        RowsSelected = self.selectionModel ().selectedRows()
        selectedIndices = [ modelIndex.column() for modelIndex in RowsSelected if not self.isSectionHidden(modelIndex.column()) ]
        return selectedIndices
        """
        sel = self.selectionModel().selection()
        chunks = np.array([], dtype=int)
        for item in sel:
            chunks = np.union1d(chunks, range(item.top(), item.bottom() + 1))

        # selectedIndices = [indRow for indRow in chunks if not self.isSectionHidden(indRow) ]
        return chunks

    def selectVerts(self):
        selectedIndices = self.getSelectedRows()
        self.model().datatable.selectVerts(selectedIndices)

    def lockSelectedRows(self):
        selectedIndices = self.getSelectedRows()
        self.model().datatable.lockRows(selectedIndices)

    def lockAllButSelectedRows(self):
        selectedIndices = set(range(self.count()))
        selectedIndices.difference_update(self.getSelectedRows())
        self.model().datatable.lockRows(selectedIndices)

    def unlockSelectedRows(self):
        selectedIndices = self.getSelectedRows()
        self.model().datatable.unLockRows(selectedIndices)

    def clearLocks(self):
        self.model().datatable.unLockRows(range(self.count()))


class HorizHeaderView(QtWidgets.QHeaderView):
    """
    _colors = [(161,105,48), (159,161,48), (104,161,48), (48,161,93),
                (48,161,161), (48,103,161), (111,48,161), (161,48,105)]
    """

    def getColors(self):
        self._colors = []
        for i in xrange(1, 9):
            col = cmds.displayRGBColor("userDefined{0}".format(i), q=True)
            self._colors.append([int(el * 255) for el in col])

    def __init__(self, colWidth=10, parent=None):
        super(HorizHeaderView, self).__init__(QtCore.Qt.Horizontal, parent)
        self.getColors()
        self.colWidth = colWidth
        self._font = QtGui.QFont("Myriad Pro", 10)
        self._font.setBold(False)
        self._metrics = QtGui.QFontMetrics(self._font)
        self._descent = self._metrics.descent()
        self._margin = 5
        self._colorDrawHeight = 20
        self.setSectionsClickable(True)
        self.setHighlightSections(True)
        self.setSectionResizeMode(QtWidgets.QHeaderView.Fixed)

        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.showMenu)

        self.regularBG = QtGui.QBrush(QtGui.QColor(130, 130, 130))
        self.greyBG = QtGui.QBrush(QtGui.QColor(100, 100, 100))

        self.blueBG = QtGui.QBrush(QtGui.QColor(112, 124, 137))
        self.redBG = QtGui.QBrush(QtGui.QColor(134, 119, 127))
        self.yellowBG = QtGui.QBrush(QtGui.QColor(144, 144, 122))

    def mousePressEvent(self, event):
        nbShown = 0
        for ind in range(self.count()):
            if not self.isSectionHidden(ind):
                nbShown += 1
        outClick = event.pos().x() > self.colWidth * nbShown
        if outClick:
            if event.button() == QtCore.Qt.MidButton:
                self.parent().parent().reizeToMinimum()
            elif event.button() == QtCore.Qt.LeftButton:
                self.parent().clearSelection()
        elif self.height() - event.pos().y() < 20:
            index = self.visualIndexAt(event.pos().x())
            self.setColor(event.pos(), index)
        else:
            super(HorizHeaderView, self).mousePressEvent(event)

    def color(self, ind):
        return self._colors[cmds.getAttr(self.model().fullColumnNames()[ind] + ".objectColor")]

    def setColor(self, pos, index):
        menu = ColorMenu(self)
        pos = self.mapToGlobal(pos)  # + QtCore.QPoint(30,310)
        menu.exec_(pos)
        color = menu.color()
        if color is None:
            return
        else:
            cmds.setAttr(self.model().fullColumnNames()[index] + ".objectColor", color)

    def getSelectedColumns(self):
        """
        columnsSelected = self.selectionModel ().selectedColumns()
        selectedIndices = [ modelIndex.column() for modelIndex in columnsSelected if not self.isSectionHidden(modelIndex.column()) ]
        return selectedIndices
        """

        sel = self.selectionModel().selection()
        chunks = np.array([], dtype=int)
        for item in sel:
            chunks = np.union1d(chunks, range(item.left(), item.right() + 1))

        selectedIndices = [indCol for indCol in chunks if not self.isSectionHidden(indCol)]
        return selectedIndices

    def lockSelectedColumns(self):
        selectedIndices = self.getSelectedColumns()
        self.model().datatable.lockColumns(selectedIndices)

    def lockAllButSelectedColumns(self):
        selectedIndices = set(range(self.count() - 1))
        selectedIndices.difference_update(self.getSelectedColumns())
        self.model().datatable.lockColumns(selectedIndices)

    def unlockSelectedColumns(self):
        selectedIndices = self.getSelectedColumns()
        self.model().datatable.unLockColumns(selectedIndices)

    def selectDeformers(self):
        selectedIndices = self.getSelectedColumns()
        self.model().datatable.selectDeformers(selectedIndices)

    def clearLocks(self):
        self.model().datatable.unLockColumns(range(self.count() - 1))

    def showMenu(self, pos):
        popMenu = QtWidgets.QMenu(self)
        selectionIsEmpty = self.selectionModel().selection().isEmpty()
        # columnsSelected = self.selectionModel ().selectedColumns()

        selAction = popMenu.addAction("select deformers")
        selAction.triggered.connect(self.selectDeformers)
        selAction.setEnabled(not selectionIsEmpty)
        popMenu.addSeparator()

        lockAction = popMenu.addAction("lock selected")
        lockAction.triggered.connect(self.lockSelectedColumns)
        lockAction.setEnabled(not selectionIsEmpty)

        lockAllButSelAction = popMenu.addAction("lock all but selected")
        lockAllButSelAction.triggered.connect(self.lockAllButSelectedColumns)
        lockAllButSelAction.setEnabled(not selectionIsEmpty)

        unlockAction = popMenu.addAction("unlock selected")
        unlockAction.triggered.connect(self.unlockSelectedColumns)
        unlockAction.setEnabled(not selectionIsEmpty)

        clearLocksAction = popMenu.addAction("clear all Locks")
        clearLocksAction.triggered.connect(self.clearLocks)

        # newAction .setCheckable (True)
        # newAction .setChecked (self.isDockable)
        model = self.model()
        hideColumnIndices = model.datatable.hideColumnIndices
        if len(hideColumnIndices) > 0:
            columnNames = model.columnNames()
            popMenu.addSeparator()
            subMenuFollow = popMenu.addMenu("show Columns")
            for ind in hideColumnIndices:
                # newAction = subMenuFollow .addAction(columnNames [ind])
                # newAction.setCheckable (True)
                # newAction.setChecked (False)
                chbox = QtWidgets.QCheckBox(columnNames[ind], subMenuFollow)

                chbox.setChecked(not self.isSectionHidden(ind))
                chbox.toggled.connect(partial(self.toggledColumn, ind, columnNames[ind]))

                checkableAction = QtWidgets.QWidgetAction(subMenuFollow)
                checkableAction.setDefaultWidget(chbox)
                subMenuFollow.addAction(checkableAction)
        popMenu.exec_(self.mapToGlobal(pos))

    def toggledColumn(self, ind, ColumnName, checked):
        print checked, ind, ColumnName
        # theShow -----------------
        if not checked:
            self.parent().hideColumn(ind)
        else:
            self.parent().showColumn(ind)

    def paintSection(self, painter, rect, index):
        # https://github.com/openwebos/qt/blob/master/src/gui/itemviews/qheaderview.cpp
        if not rect.isValid():
            return
        isLastColumn = index >= self.model().datatable.nbDrivers
        data = self._get_data(index)

        if isLastColumn:
            painter.save()
            painter.setBrush(self.greyBG)
            pen = QtGui.QPen(QtGui.QColor(0, 0, 0))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRect(rect)
            painter.restore()

            painter.setPen(QtGui.QColor(200, 200, 200))
            painter.drawText(rect, QtCore.Qt.AlignCenter, data)
        else:
            isBold = False
            sel = self.parent().selectionModel().selection()
            for item in sel:
                isBold = item.left() <= index <= item.right()
                if isBold:
                    break
            self._font.setBold(isBold)
            # painter.setPen (QtGui.QColor (0,0,0))
            painter.setFont(self._font)
            painter.rotate(-90)
            x = -rect.height()
            y = rect.left()

            side = self.model().getColumnSide(index)
            defaultBGInd = "RLMX".index(side)
            defaultBG = [self.blueBG, self.redBG, self.yellowBG, self.regularBG][defaultBGInd]

            theBGBrush = self.greyBG if self.model().datatable.isColumnLocked(index) else defaultBG
            # theBGBrush = self.regularBG

            painter.setBrush(theBGBrush)
            painter.drawRect(x + 1, y - 1, rect.height() - 1, rect.width())

            theColor = self.color(index)
            painter.setBrush(QtGui.QBrush(QtGui.QColor(*theColor)))
            painter.drawRect(x + 1, y - 1, 20, rect.width())

            painter.drawText(
                -rect.height() + self._margin + self._colorDrawHeight,
                rect.left() + (rect.width() + self._descent) / 2,
                data,
            )

    def sizeHint(self):
        return QtCore.QSize(10, self._get_text_width() + 2 * self._margin + self._colorDrawHeight)

    def _get_text_width(self):
        allMetrics = [self._metrics.width(colName) for colName in self.model().columnNames()]
        if allMetrics:
            return max(allMetrics) + 15
        else:
            return 50

    def _get_data(self, index):
        return self.model().getColumnText(index)


class TableView(QtWidgets.QTableView):
    """
    A simple table to demonstrate the QComboBox delegate.
    """

    selEmptied = QtCore.Signal(bool, name="selEmptied")

    def __init__(self, *args, **kwargs):
        colWidth = kwargs.pop("colWidth", None)
        QtWidgets.QTableView.__init__(self, *args, **kwargs)
        # self.sizeHintForRow = QtCore.QSize (0,10)
        self._hd = HighlightDelegate(self)
        self.setItemDelegate(self._hd)
        self.HHeaderView = HorizHeaderView(colWidth)
        self.VHeaderView = VertHeaderView()
        self.setHorizontalHeader(self.HHeaderView)
        self.setVerticalHeader(self.VHeaderView)

        self.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Fixed)

        # self.setUniformRowHeights (True)
        # btn = QtWidgets.QPushButton("Hi")
        # self.setCornerWidget(btn)

        # self.Icon = self.createIcon ()

        self._font = QtGui.QFont("Myriad Pro", 10)
        self._font.setBold(False)
        self._metrics = QtGui.QFontMetrics(self._font)
        self._descent = self._metrics.descent()
        self._margin = 5
        self._colorDrawHeight = 20
        self.regularBG = QtGui.QBrush(QtGui.QColor(130, 130, 130))

        self.__nw_heading = "Vtx"
        self.addRedrawButton()

    def rmvRedrawButton(self):
        btn = self.findChild(QtWidgets.QAbstractButton)
        btn.removeEventFilter(self)
        self.repaint()

    def addRedrawButton(self):
        btn = self.findChild(QtWidgets.QAbstractButton)
        btn.setText(self.__nw_heading)
        btn.setToolTip("Toggle selecting all table cells")
        btn.installEventFilter(self)
        opt = QtWidgets.QStyleOptionHeader()
        opt.text = btn.text()
        s = QtCore.QSize(
            btn.style()
            .sizeFromContents(QtWidgets.QStyle.CT_HeaderSection, opt, QtCore.QSize(), btn)
            .expandedTo(QtWidgets.QApplication.globalStrut())
        )

        if s.isValid():
            self.verticalHeader().setMinimumWidth(s.width())
        self.repaint()

    def selectionChanged(self, selected, deselected):
        super(TableView, self).selectionChanged(selected, deselected)

        sel = self.selectionModel().selection()
        rowsSel = []
        for item in sel:
            rowsSel += range(item.top(), item.bottom() + 1)
        self.model().datatable.updateDisplayVerts(rowsSel)

        self.selEmptied.emit(not sel.isEmpty())

    def createPixMap(self, rect):
        # Icon = QtGui.QIcon()
        thePixmap = QtGui.QPixmap(180, 180)
        painter = QtGui.QPainter()
        painter.begin(thePixmap)

        painter.rotate(-90)
        x = -rect.height()
        y = rect.left()

        data = self.model().datatable.shapeShortName

        painter.setBrush(self.regularBG)
        painter.setFont(self._font)
        painter.drawRect(x + 1, y - 1, rect.height() - 1, rect.width())

        painter.drawText(
            -rect.height() + self._margin, rect.left() + (rect.width() + self._descent) / 2, data
        )
        painter.end()
        # Icon.addPixmap(thePixmap, QtGui.QIcon.Normal, QtGui.QIcon.Off)

        return thePixmap

    def eventFilter(self, obj, event):
        if event.type() != QtCore.QEvent.Paint or not isinstance(obj, QtWidgets.QAbstractButton):
            return False
        # Paint by hand (borrowed from QTableCornerButton)
        opt = QtWidgets.QStyleOptionHeader()
        opt.initFrom(obj)
        styleState = QtWidgets.QStyle.State_None
        if obj.isEnabled():
            styleState |= QtWidgets.QStyle.State_Enabled
        if obj.isActiveWindow():
            styleState |= QtWidgets.QStyle.State_Active
        if obj.isDown():
            styleState |= QtWidgets.QStyle.State_Sunken
        opt.state = styleState
        opt.rect = obj.rect()
        # This line is the only difference to QTableCornerButton
        # opt.text = obj.text()
        # opt.textAlignment = QtCore.Qt.AlignBottom | QtCore.Qt.AlignHCenter
        opt.position = QtWidgets.QStyleOptionHeader.OnlyOneSection
        painter = QtWidgets.QStylePainter(obj)
        painter.drawItemPixmap(opt.rect, 1, self.createPixMap(opt.rect))
        painter.drawControl(QtWidgets.QStyle.CE_Header, opt)

        return True


# -------------------------------------------------------------------------------
# COLOR
# -------------------------------------------------------------------------------
class ColorMenu(QtWidgets.QMenu):
    """
    _colors = [(161,105,48), (159,161,48), (104,161,48), (48,161,93),
                (48,161,161), (48,103,161), (111,48,161), (161,48,105)]
    """

    def __init__(self, parent):
        super(ColorMenu, self).__init__(parent)
        self.getColors()
        self._color = None

        self.setFixedWidth(20)

        for index, color in enumerate(self._colors):
            pixmap = QtGui.QPixmap(12, 12)
            pixmap.fill(QtGui.QColor(*color))
            act = self.addAction("")
            act.setIcon(QtGui.QIcon(pixmap))
            act.triggered.connect(partial(self.pickColor, index))

    def getColors(self):
        self._colors = []
        for i in xrange(1, 9):
            col = cmds.displayRGBColor("userDefined{0}".format(i), q=True)
            self._colors.append([int(el * 255) for el in col])
        # cmds.displayRGBColor( "userDefined1", 0, 1, 1, create=True)

    def pickColor(self, index):
        self._color = index

    def color(self):
        return self._color

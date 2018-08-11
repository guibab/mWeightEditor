from Qt import QtGui, QtCore, QtWidgets


class TableModel(QtCore.QAbstractTableModel):
    def __init__(self, parent=None, *args):
        super(TableModel, self).__init__()
        self.datatable = None
        brown = QtGui.QColor(130, 130, 90)
        self.brownBrush = QtGui.QBrush(brown)

    def update(self, dataIn):
        print "Updating Model"
        self.datatable = dataIn
        # print 'Datatable : {0}'.format(self.datatable)

    def rowCount(self, parent=QtCore.QModelIndex()):
        return self.datatable.rowCount

    def columnCount(self, parent=QtCore.QModelIndex()):
        return self.datatable.columnCount

    def columnNames(self):
        return self.datatable.driverNames

    def data(self, index, role=QtCore.Qt.DisplayRole):
        # print 'Data Call'
        # print index.column(), index.row()
        if role == QtCore.Qt.DisplayRole:
            # return QtCore.QVariant(str(self.datatable.iget_value(i, j)))
            # return '{0:.2f}'.format(self.realData(index))
            return round(self.realData(index) * 100, 1)
        elif role == QtCore.Qt.EditRole:
            ff = self.realData(index)
            return "{0:.5f}".format(ff).rstrip("0") + "0"[0 : (ff % 1 == 0)]
        elif role == QtCore.Qt.TextAlignmentRole:
            return QtCore.Qt.AlignCenter  # | QtCore.Qt.AlignVCenter)
        elif role == QtCore.Qt.BackgroundRole:
            if self.realData(index) != 0.0:
                return self.brownBrush
        else:
            return None

    def setData(self, index, value, role=QtCore.Qt.EditRole):
        row = index.row()
        column = index.column()
        self.datatable.setValue(row, column, value)

        self.beginResetModel()
        self.datatable.rebuildRawSkin()
        self.endResetModel()

        super(TableModel, self).setData(index, value, role)
        """
        try:
            self.items[index.row()] = value 
            left = self.createIndex(index.row(), 0)
            right = self.createIndex(index.row(), self.columnCount())
            self.dataChanged.emit(left, right)
            return True
        except:
            pass
        return False             
        """

    def realData(self, index):
        row = index.row()
        column = index.column()
        return self.datatable.getValue(row, column)

    def headerData(self, col, orientation, role):
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return self.datatable.driverNames[col]
            else:
                return self.datatable.vertices[col]
        elif role == QtCore.Qt.TextAlignmentRole:
            return QtCore.Qt.AlignCenter  # | QtCore.Qt.AlignVCenter)
        else:
            return None

    def getColumnText(self, col):
        return self.datatable.driverNames[col]

    def flags(self, index):
        if not index.isValid():
            return QtCore.Qt.ItemIsEnabled
        # sresult = super(TableModel,self).flags(index)
        # result = sresult | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable
        result = QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable
        return QtCore.Qt.ItemFlags(result)

        # return QtCore.Qt.ItemIsEnabled


class HighlightDelegate(QtWidgets.QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        # print "createEditor"
        editor = QtWidgets.QLineEdit(parent)
        editor.setStyleSheet("QLineEdit { background-color: yellow; color : black; }")
        validator = QtGui.QDoubleValidator()
        editor.setValidator(validator)
        editor.setMinimumWidth(50)
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


class MyHeaderView(QtWidgets.QHeaderView):
    def __init__(self, parent=None):
        super(MyHeaderView, self).__init__(QtCore.Qt.Horizontal, parent)
        self._font = QtGui.QFont("Myriad Pro", 10)
        self._font.setBold(False)
        self._metrics = QtGui.QFontMetrics(self._font)
        self._descent = self._metrics.descent()
        self._margin = 5
        self.setSectionsClickable(True)
        self.setHighlightSections(True)
        self.setSectionResizeMode(QtWidgets.QHeaderView.Fixed)

        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.showMenu)

    def showMenu(self, pos):
        popMenu = QtWidgets.QMenu(self)
        newAction = popMenu.addAction("hello")
        # newAction.triggered.connect (self.doSetDockable)
        # newAction .setCheckable (True)
        # newAction .setChecked (self.isDockable)
        model = self.model()
        hideColumnIndices = model.datatable.hideColumnIndices
        if len(hideColumnIndices) > 0:
            columnNames = model.columnNames()
            subMenuFollow = popMenu.addMenu("show Columns")
            for ind in hideColumnIndices:
                newAction = subMenuFollow.addAction(columnNames[ind])
                newAction.setCheckable(True)
                newAction.setChecked(False)
        popMenu.exec_(self.mapToGlobal(pos))

    def paintSection(self, painter, rect, index):
        # https://github.com/openwebos/qt/blob/master/src/gui/itemviews/qheaderview.cpp
        if not rect.isValid():
            return
        isBold = False
        sel = self.parent().selectionModel().selection()
        for item in sel:
            isBold = item.left() <= index <= item.right()
            if isBold:
                break
        self._font.setBold(isBold)
        data = self._get_data(index)
        # painter.setPen (QtGui.QColor (0,0,0))
        painter.setFont(self._font)
        painter.rotate(-90)
        x = -rect.height()
        y = rect.left()

        painter.setBrush(QtGui.QBrush(QtGui.QColor(130, 130, 130)))
        painter.drawRect(x + 1, y - 1, rect.height() - 1, rect.width())
        painter.drawText(
            -rect.height() + self._margin, rect.left() + (rect.width() + self._descent) / 2, data
        )

    def sizeHint(self):
        return QtCore.QSize(10, self._get_text_width() + 2 * self._margin)

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

    def __init__(self, *args, **kwargs):
        QtWidgets.QTableView.__init__(self, *args, **kwargs)
        # self.sizeHintForRow = QtCore.QSize (0,10)
        self._hd = HighlightDelegate(self)
        self.setItemDelegate(self._hd)
        self.headerView = MyHeaderView()
        self.setHorizontalHeader(self.headerView)
        self.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Fixed)

        # self.setUniformRowHeights (True)

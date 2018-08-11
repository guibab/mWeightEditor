# https://github.com/chadmv/cmt/blob/master/scripts/cmt/deform/skinio.py
"""
fullPath   = r'H:\public\guillaume\Code\maya\python\skinWeightTableNew.py'
execfile(fullPath  , globals(), globals())

self = SkinWeightWin()
self.show()
"""
"""

with GlobalContext (message = "prepareValuesforSetSkinData"):        
    self.dataOfSkin.prepareValuesforSetSkinData (self.getRowColumnsSelected())

    
with GlobalContext (message = "prepareValuesforSetSkinData"):           
    self.storeSelection ()
    self._tm.beginResetModel()    
    self.dataOfSkin.setSkinDataExample ()    
    self._tm.endResetModel()
    self.retrieveSelection ()
    
"""

"""
a = np.arange(0.,27.,3.).reshape(3,3)
row_sums = a.sum(axis=1)
new_matrix = a / row_sums[:, np.newaxis]
"""
import blurdev
from blurdev.gui import DockWidget

from maya import OpenMayaUI, OpenMaya, OpenMayaAnim
from Qt import QtGui, QtCore, QtWidgets

# import shiboken2 as shiboken
import time, datetime

from ctypes import c_double

from maya import cmds
from pymel.core import getMelGlobal
from functools import partial

import numpy as np

###################################################################################
#
#   SKIN FUNCTIONS
#
###################################################################################
class DataOfSkin(object):
    def __init__(self):
        self.AllWght = []
        self.usedDeformersIndices = []
        self.theSkinCluster = ""
        self.vertices = []
        self.driverNames = []
        self.skinningMethod = ""
        self.normalizeWeights = []
        self.rowCount = 0
        self.columnCount = 0

        self.UNDOstack = []

    def orderMelList(self, listInd, onlyStr=True):
        # listInd = [49, 60, 61, 62, 80, 81, 82, 83, 100, 101, 102, 103, 113, 119, 120, 121, 138, 139, 140, 158, 159, 178, 179, 198, 230, 231, 250, 251, 252, 270, 271, 272, 273, 274, 291, 292, 293, 319, 320, 321, 360,361,362]
        listInds = []
        listIndString = []
        # listIndStringAndCount = []

        it = iter(listInd)
        currentValue = it.next()
        while True:
            try:
                firstVal = currentValue
                theVal = firstVal
                while currentValue == theVal:
                    currentValue = it.next()
                    theVal += 1
                theVal -= 1
                if firstVal != theVal:
                    toAppend = [firstVal, theVal]
                else:
                    toAppend = [firstVal]
                if onlyStr:
                    listIndString.append(":".join(map(unicode, toAppend)))
                else:
                    listInds.append(toAppend)
                # listIndStringAndCount .append ((theStr,theVal - firstVal + 1))
            except StopIteration:
                if firstVal != theVal:
                    toAppend = [firstVal, theVal]
                else:
                    toAppend = [firstVal]
                if onlyStr:
                    listIndString.append(":".join(map(unicode, toAppend)))
                else:
                    listInds.append(toAppend)
                # listIndStringAndCount .append ((theStr,theVal - firstVal + 1))
                break
        if onlyStr:
            return listIndString
        else:
            return listInds

    def getIndicesFromSelection(self, sel):
        indices = [el.split(".vtx[")[-1][:-1] for el in sel if ".vtx" in el]
        allIndices = set()
        for index in indices:
            if ":" in index:
                nmbs = map(int, index.split(":"))
                allIndices.update(range(nmbs[0], nmbs[1] + 1))
            else:
                allIndices.add(int(index))
        return sorted(list(allIndices))

    def getMObject(self, nodeName, returnDagPath=True):
        # We expect here the fullPath of a shape mesh
        selList = OpenMaya.MSelectionList()
        OpenMaya.MGlobal.getSelectionListByName(nodeName, selList)
        depNode = OpenMaya.MObject()
        selList.getDependNode(0, depNode)

        if returnDagPath == False:
            return depNode
        mshPath = OpenMaya.MDagPath()
        selList.getDagPath(0, mshPath, depNode)
        return mshPath

    def prepareValuesforSetSkinData(self, chunks):
        chunk = chunks[0]
        top, bottom, left, right = chunk

        selectedRows = xrange(top, bottom + 1)
        indicesVertices = [self.vertices[indRow] for indRow in selectedRows]

        # GET the sub ARRAY -------------------------------------------------------
        # self.sub2DArrayToSet = self.raw2dArray [top:bottom+1,left:right+1]
        self.sub2DArrayToSet = self.raw2dArray[
            top : bottom + 1,
        ]
        self.orig2dArray = self.sub2DArrayToSet.copy()

        # MAKING the SUB Array ----------------------------------------------------
        selectedColumns = xrange(left, right + 1)
        condArray = isin(
            range(self.nbDrivers), self.usedDeformersIndices, assume_unique=True, invert=False
        )
        condArray[:left] = False
        condArray[right + 1 :] = False

        # theNamesOfSelectedDeformers = [self.driverNames [ind] for ind, val in enumerate(condArray) if val]
        # print theNamesOfSelectedDeformers

        self.maskArray = np.tile(condArray, (len(selectedRows), 1))

        # NOW Prepare for settingSkin Cluster -------------------------------------
        self.influenceIndices = OpenMaya.MIntArray()
        self.influenceIndices.setLength(self.nbDrivers)
        for i in xrange(self.nbDrivers):
            self.influenceIndices.set(i, i)

        componentType = OpenMaya.MFn.kMeshVertComponent
        fnComponent = OpenMaya.MFnSingleIndexedComponent()
        self.userComponents = fnComponent.create(componentType)
        for ind in indicesVertices:
            fnComponent.addElement(ind)

        lengthArray = self.nbDrivers * (bottom - top + 1)

        self.newArray = OpenMaya.MDoubleArray()
        self.newArray.setLength(lengthArray)

        # set normalize FALSE --------------------------------------------------------
        cmds.setAttr(self.theSkinCluster + ".normalizeWeights", 0)

    def setSkinData(self, val):
        with GlobalContext(message="prepareSkinfn numpy"):
            selectedValues = np.copy(self.orig2dArray)
            np.putmask(selectedValues, ~self.maskArray, 0)

            remainingValues = np.copy(self.orig2dArray)
            np.putmask(remainingValues, self.maskArray, 0)

            row_sumsSelected = selectedValues.sum(axis=1)
            row_sumsRemaining = remainingValues.sum(axis=1)

            # --------------------------------------------------------------------
            # with GlobalContext (message = "prepareSkinfn numpy"):
            new2dArray = np.copy(self.orig2dArray)
            np.putmask(new2dArray, self.maskArray, new2dArray + val)

            # get number selected columns ------------------------------------
            nbColumnsSelected = np.count_nonzero(self.maskArray[0])
            # get total value added       ------------------------------------
            totalAdded = nbColumnsSelected * val
            # remove from non zero other columns -----------------------------
            # self.maskArray.
            # np.putmask(new2dArray ,~self.maskArray , new2dArray-subVal)
            # clamp 0 1 --------------------------------------------------
            new2dArray = new2dArray.clip(min=0, max=1.0)
            # normalize ------------------------------------------------
            row_sums = new2dArray.sum(axis=1)
            new2dArrayDiv = new2dArray / row_sums[:, np.newaxis]
        # set Value ------------------------------------------------
        self.actuallySetValue(
            new2dArrayDiv,
            self.sub2DArrayToSet,
            self.userComponents,
            self.influenceIndices,
            self.shapePath,
            self.sknFn,
        )

    def postSkinSet(self):
        cmds.setAttr(self.theSkinCluster + ".normalizeWeights", self.normalizeWeights)
        self.storeUndoStack()

    def storeUndoStack(self):
        # add to the Undo stack -----------------------------------------
        undoArray = np.copy(self.orig2dArray)
        self.UNDOstack.append(
            (
                undoArray,
                self.sub2DArrayToSet,
                self.userComponents,
                self.influenceIndices,
                self.shapePath,
                self.sknFn,
            )
        )

    def actuallySetValue(
        self, new2dArrayDiv, sub2DArrayToSet, userComponents, influenceIndices, shapePath, sknFn
    ):
        """
        # quick settingWeights ------------------------------
        doubles = new2dArrayDiv.flatten()
        count = new2dArrayDiv.size
        util = OpenMaya.MScriptUtil()
        util.createFromList(doubles, count)
        doublePtr = util.asDoublePtr()
        newArray = OpenMaya.MDoubleArray(doublePtr, count)
        """
        # way slower -------------------------------
        # for ind, el in enumerate(new2dArrayDiv.flat) : self.newArray.set( el, ind)

        # WAY WAY slower -------------------------------
        """
        with GlobalContext (message = "prepareValuesforSetSkinData"):        
            new2dArrayDiv = np.copy (self.dataOfSkin.orig2dArray   )
            count = new2dArrayDiv.size    
            #util = OpenMaya.MScriptUtil()
            res = OpenMaya.MScriptUtil( self.dataOfSkin.newArray)
            doublePtr = res.asDoublePtr()
            for i, val in enumerate(new2dArrayDiv.flat): res.setDoubleArray (doublePtr, i, val )
            newArray = OpenMaya.MDoubleArray(doublePtr, count)

        """
        arrayForSetting = np.copy(new2dArrayDiv)
        doubles = arrayForSetting.flatten()
        count = doubles.size
        tempArrayForSize = OpenMaya.MDoubleArray()
        tempArrayForSize.setLength(count)

        res = OpenMaya.MScriptUtil(tempArrayForSize)
        ptr = res.asDoublePtr()

        # Cast the swig double pointer to a ctypes array
        cta = (c_double * count).from_address(int(ptr))
        out = np.ctypeslib.as_array(cta)
        np.copyto(out, doubles)
        newArray = OpenMaya.MDoubleArray(ptr, count)

        # with GlobalContext (message = "sknFn.setWeights"):
        normalize = False
        UndoValues = OpenMaya.MDoubleArray()
        sknFn.setWeights(
            shapePath, userComponents, influenceIndices, newArray, normalize, UndoValues
        )

        np.put(sub2DArrayToSet, xrange(sub2DArrayToSet.size), new2dArrayDiv)

    def callUndo(self):
        if self.UNDOstack:
            print "UNDO"
            undoArgs = self.UNDOstack.pop()
            self.actuallySetValue(*undoArgs)
        else:
            print "No more undo"

    """
    def setSkinDataOld (self, val) :
        newArray = OpenMaya.MDoubleArray()
        newArray.copy (self.OrigWeights)
        nbColumns = len (self.usedDeformersIndices)

        #for indVtx, vtx in enumerate (self.indicesVertices) : 
        for indVtx, indVtxInRawArray in enumerate (self.selectedRows) : 
            sumVal = 0
            prevSumVal = 0
            for selectedCol in self.userSelectedColumnsIndices : 
                selCol = int(selectedCol)
                indInArray = indVtx*nbColumns+selCol

                origVal = self.OrigWeights [indInArray]
                newVal = np.round (origVal+val, 5)
                if newVal < 0. : newVal = 0.0
                #newVal = np.clip(origVal+val,0,1)
                prevSumVal += origVal
                sumVal += newVal
                newArray.set (newVal, indInArray)
            
            # Normalize -------------------------------------
            if sumVal>1.0 : 
                toMult = 1./sumVal
                for selectedCol in self.userSelectedColumnsIndices : 
                    selCol = int(selectedCol)
                    indInArray = indVtx*nbColumns+selCol

                    newVal = newArray [indInArray]
                    newArray.set (newVal * toMult, indInArray)

                for selectedCol in self.notSelColumnsIndices : 
                    selCol = int(selectedCol)
                    indInArray = indVtx*nbColumns+selCol
                    newArray.set (0, indInArray)
            else : 
                prevToDiv = 1.-prevSumVal
                if prevToDiv :
                    toMult = (1.-sumVal) / prevToDiv
                    globalSum = sumVal
                    for selectedCol in self.notSelColumnsIndices : 
                        selCol = int(selectedCol)
                        indInArray = indVtx*nbColumns+selCol

                        origVal = self.OrigWeights [indInArray]
                        newVal = np.round (origVal*toMult, 5)

                        if ( globalSum + newVal ) >1.0 : newVal = 1.0 - globalSum
                        globalSum += newVal
                        newArray.set (newVal, indInArray)

            # set rawSkinValues ------------------------
            for selectedCol, colBigArr in enumerate (self.usedDeformersIndices) : 
                indInBigArray = indVtxInRawArray*self.nbDrivers+int(colBigArr)
                indInArray = indVtx*nbColumns+int(selectedCol)

                newVal = newArray [indInArray]
                self.rawSkinValues.set (newVal, indInBigArray)
        # store old values
        oldValues  = OpenMaya.MDoubleArray()
        normalize = True
        self.sknFn.setWeights( self.shapePath              ,
                               self.userComponents         ,
                               self.influenceIndices       ,
                               newArray                    ,
                               normalize                   ,
                               oldValues                   ) 

        return
    """

    def exposeSkinData(self, inputSkinCluster, indices=[]):
        self.skinClusterObj = self.getMObject(inputSkinCluster, returnDagPath=False)
        self.sknFn = OpenMayaAnim.MFnSkinCluster(self.skinClusterObj)

        jointPaths = OpenMaya.MDagPathArray()
        self.sknFn.influenceObjects(jointPaths)

        self.shapePath = OpenMaya.MDagPath()
        self.sknFn.getPathAtIndex(0, self.shapePath)
        shapeName = self.shapePath.fullPathName()
        vertexCount = 0

        componentType = OpenMaya.MFn.kMeshVertComponent
        if cmds.nodeType(shapeName) == "nurbsCurve":
            componentType = OpenMaya.MFn.kCurveCVComponent
            crvFn = OpenMaya.MFnNurbsCurve(self.shapePath)
            cvPoints = OpenMaya.MPointArray()

            crvFn.getCVs(cvPoints, OpenMaya.MSpace.kObject)
            vertexCount = cvPoints.length()
        else:
            mshFn = OpenMaya.MFnMesh(self.shapePath)
            vertexCount = mshFn.numVertices()
        fnComponent = OpenMaya.MFnSingleIndexedComponent()
        self.fullComponent = fnComponent.create(componentType)

        if not indices:
            fnComponent.setCompleteData(vertexCount)
        else:
            for ind in indices:
                fnComponent.addElement(ind)
        weights = OpenMaya.MDoubleArray()

        intptrUtil = OpenMaya.MScriptUtil()
        intptrUtil.createFromInt(0)
        intPtr = intptrUtil.asUintPtr()

        self.sknFn.getWeights(self.shapePath, self.fullComponent, weights, intPtr)

        return weights

    def getSkinClusterFromSel(self, sel):
        if sel:
            hist = cmds.listHistory(sel, lv=0, pruneDagObjects=True)
            if hist:
                skinClusters = cmds.ls(hist, type="skinCluster")
                if skinClusters:
                    skinCluster = skinClusters[0]
                    theDeformedMesh = cmds.ls(
                        cmds.listHistory(skinCluster, af=True, f=True), type="mesh"
                    )
                    return skinCluster, theDeformedMesh[0]
        return "", ""

    def getSkinClusterValues(self, skinCluster):
        driverNames = cmds.skinCluster(skinCluster, q=True, inf=True)
        skinningMethod = cmds.getAttr(skinCluster + ".skinningMethod")
        normalizeWeights = cmds.getAttr(skinCluster + ".normalizeWeights")
        return (driverNames, skinningMethod, normalizeWeights)

    def getZeroColumns(self):
        """
        arr = np.array([])
        lent = self.rawSkinValues.length()
        arr.resize( lent )
        with GlobalContext (message = "convertingSkinValues"):
            for i, val in enumerate (self.rawSkinValues) : arr[i]=val
        self.raw2dArray = np.reshape(arr, (-1, self.nbDrivers))
        """
        """
        # faster -----------------------------------------------
        arr = np.array([])
        lent = self.rawSkinValues.length() 
        arr.resize( lent )
        res = OpenMaya.MScriptUtil( self.rawSkinValues)
        util = maya.OpenMaya.MScriptUtil()
        ptr = res.asDoublePtr()
        with GlobalContext (message = "convertingSkinValues"):    
            for i in xrange (lent): arr[i]=util.getDoubleArrayItem(ptr, i)
        self.raw2dArray = np.reshape(arr, (-1, self.nbDrivers))
        """
        # deadFast ----------------------------------------------------
        res = OpenMaya.MScriptUtil(self.rawSkinValues)
        util = OpenMaya.MScriptUtil()
        ptr = res.asDoublePtr()

        lent = self.rawSkinValues.length()
        with GlobalContext(message="convertingSkinValues"):
            cta = (c_double * lent).from_address(int(ptr))
            arr = np.ctypeslib.as_array(cta)
            self.raw2dArray = np.copy(arr)
            self.raw2dArray = np.reshape(self.raw2dArray, (-1, self.nbDrivers))
        # now find the zeroColumns ------------------------------------
        myAny = np.any(self.raw2dArray, axis=0)
        self.usedDeformersIndices = np.where(myAny)[0]
        self.hideColumnIndices = np.where(~myAny)[0]

        ################################################################################################################################################################

    def getAllData(self):
        sel = cmds.ls(sl=True)
        self.theSkinCluster, self.deformedShape = self.getSkinClusterFromSel(sel)
        self.raw2dArray = None

        if not self.theSkinCluster:
            self.usedDeformersIndices = []
            self.hideColumnIndices = []
            self.vertices = []
            self.rowCount = 0
            self.columnCount = 0
            self.meshIsUsed = False
            return
        self.driverNames, self.skinningMethod, self.normalizeWeights = self.getSkinClusterValues(
            self.theSkinCluster
        )
        self.nbDrivers = len(self.driverNames)

        with GlobalContext(message="rawSkinValues"):
            self.vertices = self.getIndicesFromSelection(sel)
            if not self.vertices:
                self.vertices = cmds.getAttr(
                    "{0}.weightList".format(self.theSkinCluster), multiIndices=True
                )
                self.meshIsUsed = True
                self.rawSkinValues = self.exposeSkinData(self.theSkinCluster)
            else:
                self.rawSkinValues = self.exposeSkinData(self.theSkinCluster, indices=self.vertices)
                self.meshIsUsed = False
            # print "rawSkinValues length : {0}" .format (self.rawSkinValues.length())
        self.hideColumnIndices = []
        self.usedDeformersIndices = range(self.nbDrivers)

        self.rowCount = len(self.vertices)
        self.columnCount = self.nbDrivers

        self.getZeroColumns()

    def rebuildRawSkin(self):
        if self.meshIsUsed:
            self.rawSkinValues = self.exposeSkinData(self.theSkinCluster)
        else:
            self.rawSkinValues = self.exposeSkinData(self.theSkinCluster, indices=self.vertices)

    def getValue(self, row, column):
        # return self.raw2dArray [row][column] if self.raw2dArray !=None else self.rawSkinValues [row*self.nbDrivers+column]
        return self.raw2dArray[row][column]
        # return

    def setValue(self, row, column, value):
        vertexIndex = self.vertices[row]
        deformerName = self.driverNames[column]
        theVtx = "{0}.vtx[{1}]".format(self.deformedShape, vertexIndex)
        print self.theSkinCluster, theVtx, deformerName, value
        cmds.skinPercent(
            self.theSkinCluster, theVtx, transformValue=(deformerName, float(value)), normalize=True
        )


###################################################################################
#
#   Table FUNCTIONS
#
###################################################################################

styleSheet = """
QWidget {
background:  #aba8a6;
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


###################################################################################
#
#   the slider setting
#
###################################################################################
class ValueSetting(QtWidgets.QWidget):
    theStyleSheet = "QDoubleSpinBox {color: black; background-color:rgb(200,200,200) ; border: 1px solid black;text-align: center;}"

    def __init__(self, parent=None, singleStep=0.1, precision=2):
        super(ValueSetting, self).__init__(parent=None)
        self.theProgress = ProgressItem("skinVal", szrad=0, value=50)
        self.theProgress.mainWindow = parent
        self.mainWindow = parent
        # self.displayText = QtWidgets.QLabel (self)

        layout = QtWidgets.QHBoxLayout(self)
        # layout.setContentsMargins(40, 0, 0, 0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # layout.addStretch ()
        """
        self.theSpinner = QtWidgets.QLineEdit(parent)
        self.theSpinner.setStyleSheet("QLineEdit { background-color: yellow; color : black; }")
        validator = QtGui.QDoubleValidator()
        self.theSpinner.setValidator(validator)
        self.theSpinner.editingFinished.connect (self.spinnerValueSet)
        """

        self.theSpinner = QtWidgets.QDoubleSpinBox(self)
        self.theSpinner.setRange(-16777214, 16777215)
        self.theSpinner.setSingleStep(singleStep)
        self.theSpinner.setDecimals(precision)
        self.theSpinner.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.theSpinner.setStyleSheet(self.theStyleSheet)

        # self.theSpinner.valueChanged.connect (self.valueEntered)

        newPolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum
        )
        self.theSpinner.setMaximumWidth(40)
        newPolicy.setHorizontalStretch(0)
        newPolicy.setVerticalStretch(0)
        self.theSpinner.setSizePolicy(newPolicy)

        self.theProgress.setMaximumHeight(18)

        self.theLineEdit = None
        for chd in self.theSpinner.children():
            if isinstance(chd, QtWidgets.QLineEdit):
                self.theLineEdit = chd
                break
        self.theLineEdit.returnPressed.connect(self.spinnerValueEntered)
        self.theSpinner.focusInEvent = self.theSpinner_focusInEvent

        layout.addWidget(self.theSpinner)
        layout.addWidget(self.theProgress)

        # layout.addStretch ()
        self.theProgress.valueChanged.connect(self.setVal)
        # self.theProgress.setMaximumWidth (300)

    def theSpinner_focusInEvent(self, event):
        # print "focus press"
        QtWidgets.QDoubleSpinBox.focusInEvent(self.theSpinner, event)
        cmds.evalDeferred(self.theLineEdit.selectAll)
        # cmds.evalDeferred (self.theLineEdit.grabKeyboard)

    def spinnerValueEntered(self):
        theVal = self.theSpinner.value()
        print "value Set {0}".format(theVal)

        self.mainWindow.prepareToSetValue()
        self.mainWindow.doAddValue(theVal)
        self.setVal(self.theProgress.releasedValue)

    def setVal(self, val):
        # theVal = val/100.
        theVal = (val - 50) / 50.0
        # ------- SETTING FUNCTION ---------------------
        if self.theProgress.startDrag:
            self.mainWindow.doAddValue(theVal)
        else:
            self.mainWindow.dataOfSkin.postSkinSet()

        # else : # wheelEvent

        self.theSpinner.setValue(theVal)

    """
    def valueEntered (self, theVal) : 
        if theVal == 0.0 : # end of drag
            print "end of drag", theVal
            self.mainWindow.dataOfSkin.postSkinSet ()

        if self.theProgress.startDrag : 
            self.mainWindow.doAddValue (theVal)
        else : # if direct value or whheel we set prepare it
            print "spinnerValueSet [{0}]".format( theVal )
            self.mainWindow.prepareToSetValue()
            self.mainWindow.doAddValue (theVal)
            self.theProgress.applyVal (self.theProgress.releasedValue)
            #self.mainWindow.dataOfSkin.postSkinSet ()

    def setVal (self, val) :
        #theVal = val/100.
        theVal = (val-50)/50.
        # ------- SETTING FUNCTION ---------------------
        self.theSpinner.setValue (theVal)
    """


class ProgressItem(QtWidgets.QProgressBar):
    theStyleSheet = "QProgressBar {{color: black; background-color:{bgColor} ; border: 1px solid black;text-align: center;\
    border-bottom-right-radius: {szrad}px;\
    border-bottom-left-radius: {szrad}px;\
    border-top-right-radius: {szrad}px;\
    border-top-left-radius: {szrad}px;}}\
    QProgressBar::chunk {{background:{chunkColor};\
    border-bottom-right-radius: {szrad}px;\
    border-bottom-left-radius: {szrad}px;\
    border-top-right-radius: {szrad}px;\
    border-top-left-radius: {szrad}px;\
    }}"
    mainWindow = None

    def __init__(self, theName, value=0, **kwargs):
        super(ProgressItem, self).__init__()
        self.multiplier = 1
        self.range = (0, 1)

        # self.setFormat (theName+" %p%")
        self.setFormat("")
        self.dicStyleSheet = dict(
            {"szrad": 7, "bgColor": "rgb(136,136,136)", "chunkColor": "rgb(200,200,200)"}, **kwargs
        )

        self.setStyleSheet(self.theStyleSheet.format(**self.dicStyleSheet))
        self.setValue(value)

        self.autoReset = True
        self.releasedValue = 50.0

    def refreshValue(self):
        att = self.theAttr if not isinstance(self.theAttr, list) else self.theAttr[0]
        if cmds.objExists(att):
            val = cmds.getAttr(att) * self.multiplier
            # with toggleBlockSignals ([self]) :
            self.setValue(int(val * 100))

    def changeColor(self, **kwargs):
        self.dicStyleSheet = dict(
            {"szrad": 7, "bgColor": "rgb(200,200,230)", "chunkColor": "#FF0350"}, **kwargs
        )
        self.setStyleSheet(self.theStyleSheet.format(**self.dicStyleSheet))

    def setEnabled(self, val):
        super(ProgressItem, self).setEnabled(val)
        print "set Enalbeld {0}".format(val)
        if not val:
            tmpDic = dict(
                self.dicStyleSheet,
                **{"szrad": 7, "bgColor": "rgb(100,100,100)", "chunkColor": "#FF0350"}
            )
            self.setStyleSheet(self.theStyleSheet.format(**tmpDic))
        else:
            self.setStyleSheet(self.theStyleSheet.format(**self.dicStyleSheet))

    def applyVal(self, val):
        # print "applyVal {0}".format (val)
        val *= self.multiplier
        if self.minimum() == -100:
            val = val * 2 - 1
        self.setValue(int(val * 100))

    def wheelEvent(self, e):
        delta = e.delta()
        # print delta
        val = self.value() / 100.0
        if self.minimum() == -100:
            val = val * 0.5 + 0.5

        offset = -0.1 if delta < 0 else 0.1
        val += offset
        if val > 1.0:
            val = 1.0
        elif val < 0.0:
            val = 0.0
        self.applyVal(val)

    startDrag = False

    def mousePressEvent(self, event):
        if event.modifiers() == QtCore.Qt.ControlModifier or event.button() != QtCore.Qt.LeftButton:
            super(ProgressItem, self).mousePressEvent(event)
            self.startDrag = False
        else:
            cmds.undoInfo(stateWithoutFlush=False)
            # ------------- PREPARE FUNCTION -------------------------------------------------------------------------------------
            self.startDrag = self.mainWindow.prepareToSetValue()
            if self.startDrag:
                self.applyTheEvent(event)

    def mouseReleaseEvent(self, event):
        self.startDrag = False
        if event.modifiers() == QtCore.Qt.ControlModifier or event.button() != QtCore.Qt.LeftButton:
            super(ProgressItem, self).mouseReleaseEvent(event)
        else:
            # print "releasing"
            self.setMouseTracking(False)
            cmds.undoInfo(stateWithoutFlush=True)
            super(ProgressItem, self).mouseReleaseEvent(event)
        if self.autoReset:
            self.setValue(self.releasedValue)

    def applyTheEvent(self, e):
        shitIsHold = e.modifiers() == QtCore.Qt.ShiftModifier
        theWdth = self.width()
        # print e.mouseButtons()
        # print "moving {0}".format (e.x())
        val = e.x() / float(theWdth)
        if shitIsHold:
            val = round(val * 4.0) / 4.0
        # print (val)
        if val > 1.0:
            val = 1.0
        elif val < 0.0:
            val = 0.0
        self.applyVal(val)

    def mouseMoveEvent(self, event):
        isLeft = event.button() == QtCore.Qt.LeftButton
        isCtr = event.modifiers() == QtCore.Qt.ControlModifier
        # print "mouseMoveEvent ", isLeft, isCtr
        if self.startDrag:
            self.applyTheEvent(event)
        super(ProgressItem, self).mouseMoveEvent(event)


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
        theLayout = QtWidgets.QVBoxLayout(self)
        theLayout.setContentsMargins(10, 10, 10, 10)
        theLayout.setSpacing(3)

        self._tm = TableModel(self)
        self._tm.update(self.dataOfSkin)

        self._tv = TableView(self)
        self._tv.setModel(self._tm)
        # self._tm._tv = self._tv

        refreshBTN = QtWidgets.QPushButton("refresh")
        refreshBTN.clicked.connect(self.refreshBtn)
        theLayout.addWidget(refreshBTN)

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
        self._tv.headerView.setMaximumWidth(
            self.colWidth * len(self.dataOfSkin.usedDeformersIndices)
        )

    def get_data_frame(self):
        with GlobalContext(message="get_data_frame"):
            self.dataOfSkin.getAllData()
        return self.dataOfSkin


###################################################################################
#
#   Global FUNCTIONS
#
###################################################################################
def isin(element, test_elements, assume_unique=False, invert=False):
    element = np.asarray(element)
    return np.in1d(element, test_elements, assume_unique=assume_unique, invert=invert).reshape(
        element.shape
    )


class GlobalContext(object):
    def __init__(self, raise_error=True, message="processing", openUndo=True, suspendRefresh=False):
        self.raise_error = raise_error
        self.openUndo = openUndo
        self.suspendRefresh = suspendRefresh
        self.message = message

    def __enter__(self):
        self.startTime = time.time()
        cmds.waitCursor(state=True)
        if self.openUndo:
            cmds.undoInfo(openChunk=True)
        if self.suspendRefresh:
            cmds.refresh(suspend=True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if cmds.waitCursor(q=True, state=True):
            cmds.waitCursor(state=False)
        if self.openUndo:
            cmds.undoInfo(closeChunk=True)
        if self.suspendRefresh:
            cmds.refresh(suspend=False)
            cmds.refresh()
        completionTime = time.time() - self.startTime
        timeRes = str(datetime.timedelta(seconds=int(completionTime))).split(":")
        result = "{0} hours {1} mins {2} secs".format(*timeRes)
        print "{0} executed in {1} [{2:.2f} secs]".format(self.message, result, completionTime)

        if exc_type is not None:
            if self.raise_error:
                import traceback

                traceback.print_tb(exc_tb)
                raise exc_type, exc_val
            else:
                sys.stderr.write("%s" % exc_val)

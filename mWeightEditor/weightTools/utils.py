from __future__ import print_function
from __future__ import absolute_import
import sys
from maya import cmds
import time
import datetime
from maya import OpenMaya
import six
from six.moves import range
from six.moves import zip

# -------------------------------------------------------------------------------------------
# ------------------------ global functions -------------------------------------------------
# -------------------------------------------------------------------------------------------
class SettingWithRedraw(object):
    def __init__(self, theWindow, raise_error=True):
        self.theWindow = theWindow

    def __enter__(self):
        self.theWindow.storeSelection()
        self.theWindow._tm.beginResetModel()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.theWindow._tm.endResetModel()
        self.theWindow.retrieveSelection()
        # add a refresh of the locator ?


class SettingVariable(object):
    def __init__(self, variableHolder, variableName, valueOn=True, valueOut=False):
        self.variableHolder = variableHolder
        self.variableName = variableName
        self.valueOn = valueOn
        self.valueOut = valueOut

    def __enter__(self):
        if isinstance(self.variableHolder, dict):
            self.variableHolder[self.variableName] = self.valueOn
        else:
            self.variableHolder.__dict__[self.variableName] = self.valueOn

    def __exit__(self, exc_type, exc_val, exc_tb):
        if isinstance(self.variableHolder, dict):
            self.variableHolder[self.variableName] = self.valueOut
        else:
            self.variableHolder.__dict__[self.variableName] = self.valueOut


class ToggleHeaderVisibility(object):
    def __init__(self, HH, raise_error=True):
        self.HH = HH

    def __enter__(self):
        self.HH.hide()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.HH.show()


class GlobalContext(object):
    def __init__(
        self,
        message="processing",
        raise_error=True,
        openUndo=True,
        suspendRefresh=False,
        doPrint=True,
    ):
        self.raise_error = raise_error
        self.openUndo = openUndo
        self.suspendRefresh = suspendRefresh
        self.message = message
        self.doPrint = doPrint

    def __enter__(self):
        self.startTime = time.time()
        cmds.waitCursor(state=True)
        if self.openUndo:
            cmds.undoInfo(openChunk=True, chunkName=self.message)
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
        if self.doPrint:
            result = "{0} hours {1} mins {2} secs".format(*timeRes)
            print("{0} executed in {1}[{2:.2f} secs]".format(self.message, result, completionTime))
        if exc_type is not None:
            if self.raise_error:
                import traceback

                traceback.print_tb(exc_tb)
                raise exc_type(exc_val)
            else:
                sys.stderr.write("%s" % exc_val)


class toggleBlockSignals(object):
    def __init__(self, listWidgets, raise_error=True):
        self.listWidgets = listWidgets

    def __enter__(self):
        for widg in self.listWidgets:
            widg.blockSignals(True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for widg in self.listWidgets:
            widg.blockSignals(False)


# -------------------------------------------------------------------------------------------
# ------------------------ softSelections ---------------------------------------------------
# -------------------------------------------------------------------------------------------
def getSoftSelectionValues():
    richSel = OpenMaya.MRichSelection()
    try:
        OpenMaya.MGlobal.getRichSelection(richSel)
    except RuntimeError:
        return []
    richSelList = OpenMaya.MSelectionList()
    richSel.getSelection(richSelList)

    path = OpenMaya.MDagPath()
    component = OpenMaya.MObject()
    richSelList.getDagPath(0, path, component)

    try:
        componentFn = OpenMaya.MFnSingleIndexedComponent(component)
    except Exception:
        return []
    count = componentFn.elementCount()
    elementIndices = [componentFn.element(i) for i in range(count)]
    elementWeights = [componentFn.weight(i).influence() for i in range(count)]
    return elementIndices, elementWeights


def getSoftSelectionValuesNEW(returnSimpleIndices=True, forceReturnWeight=False):
    softOn = cmds.softSelect(q=True, softSelectEnabled=True)
    richSelList = OpenMaya.MSelectionList()

    if softOn:
        richSel = OpenMaya.MRichSelection()
        try:
            OpenMaya.MGlobal.getRichSelection(richSel)
        except RuntimeError:
            return []
        richSel.getSelection(richSelList)
    else:
        OpenMaya.MGlobal.getActiveSelectionList(richSelList)
    uVal = OpenMaya.MScriptUtil()
    uVal.createFromInt(0)
    ptru = uVal.asIntPtr()

    vVal = OpenMaya.MScriptUtil()
    vVal.createFromInt(0)
    ptrv = vVal.asIntPtr()

    wVal = OpenMaya.MScriptUtil()
    wVal.createFromInt(0)
    ptrw = wVal.asIntPtr()
    toReturn = {}
    if not richSelList.isEmpty():
        iterSel = OpenMaya.MItSelectionList(richSelList)

        while not iterSel.isDone():
            component = OpenMaya.MObject()
            dagPath = OpenMaya.MDagPath()
            try:
                iterSel.getDagPath(dagPath, component)
            except Exception:
                iterSel.next()
                continue
            depNode_name = dagPath.fullPathName()

            elementIndices = []
            elementWeights = []
            if not component.isNull():
                componentFn = OpenMaya.MFnComponent(component)
                count = componentFn.elementCount()
                if componentFn.componentType() in [
                    OpenMaya.MFn.kCurveCVComponent,
                    OpenMaya.MFn.kMeshVertComponent,
                    OpenMaya.MFn.kMeshPolygonComponent,
                    OpenMaya.MFn.kMeshEdgeComponent,
                ]:
                    singleFn = OpenMaya.MFnSingleIndexedComponent(component)

                    if componentFn.componentType() == OpenMaya.MFn.kMeshPolygonComponent:
                        polyIter = OpenMaya.MItMeshPolygon(dagPath, component)
                        setOfVerts = set()
                        while not polyIter.isDone():
                            connectedVertices = OpenMaya.MIntArray()
                            polyIter.getVertices(connectedVertices)
                            for j in range(connectedVertices.length()):
                                setOfVerts.add(connectedVertices[j])
                            polyIter.next()
                        lstVerts = list(setOfVerts)
                        lstVerts.sort()
                        for vtx in lstVerts:
                            elementIndices.append(vtx)
                            elementWeights.append(1)
                        # convert
                    elif (
                        componentFn.componentType() == OpenMaya.MFn.kMeshEdgeComponent
                    ):  # not softSel
                        edgeIter = OpenMaya.MItMeshEdge(dagPath, component)
                        setOfVerts = set()
                        while not edgeIter.isDone():
                            for j in [0, 1]:
                                setOfVerts.add(edgeIter.index(j))
                            edgeIter.next()
                        lstVerts = list(setOfVerts)
                        lstVerts.sort()
                        for vtx in lstVerts:
                            elementIndices.append(vtx)
                            elementWeights.append(1)
                    else:  # regular vertices or softSelection
                        for i in range(count):
                            weight = componentFn.weight(i).influence() if softOn else 1
                            elementIndices.append(singleFn.element(i))
                            elementWeights.append(weight)
                elif componentFn.componentType() == OpenMaya.MFn.kSurfaceCVComponent:
                    numCVsInV_ = cmds.getAttr(depNode_name + ".spansV") + cmds.getAttr(
                        depNode_name + ".degreeV"
                    )

                    doubleFn = OpenMaya.MFnDoubleIndexedComponent(component)
                    for i in range(count):
                        weight = componentFn.weight(i).influence() if softOn else 1
                        doubleFn.getElement(i, ptru, ptrv)
                        u = uVal.getInt(ptru)
                        v = vVal.getInt(ptrv)
                        if returnSimpleIndices:
                            elementIndices.append(numCVsInV_ * u + v)
                        else:
                            elementIndices.append((u, v))
                        elementWeights.append(weight)
                elif componentFn.componentType() == OpenMaya.MFn.kLatticeComponent:
                    div_s = cmds.getAttr(depNode_name + ".sDivisions")
                    div_t = cmds.getAttr(depNode_name + ".tDivisions")
                    div_u = cmds.getAttr(depNode_name + ".uDivisions")

                    tripleFn = OpenMaya.MFnTripleIndexedComponent(component)
                    for i in range(count):
                        tripleFn.getElement(i, ptru, ptrv, ptrw)
                        s = uVal.getInt(ptru)
                        t = vVal.getInt(ptrv)
                        u = wVal.getInt(ptrw)
                        simpleIndex = getThreeIndices(div_s, div_t, div_u, s, t, u)
                        weight = componentFn.weight(i).influence() if softOn else 1

                        if returnSimpleIndices:
                            elementIndices.append(simpleIndex)
                        else:
                            elementIndices.append((s, t, u))
                        elementWeights.append(weight)
            if forceReturnWeight or softOn:
                toReturn[depNode_name] = (elementIndices, elementWeights)
            else:
                toReturn[depNode_name] = elementIndices
            iterSel.next()
    return toReturn


def getThreeIndices(div_s, div_t, div_u, *args):
    if len(args) == 1:
        (simpleIndex,) = args
        s = simpleIndex % div_s
        t = (simpleIndex - s) / div_s % div_t
        u = (simpleIndex - s - t * div_s) / (div_s * div_t)
        return s, t, u
    elif len(args) == 3:
        s, t, u = args
        simpleIndex = u * div_s * div_t + t * div_s + s
        return simpleIndex


def getComponentIndexList(componentList=[]):
    # https://github.com/bungnoid/glTools/blob/master/utils/component.py
    """
    Return an list of integer component index values
    @param componentList: A list of component names. if empty will default to selection.
    @type componentList: list
    """
    # Initialize return dictionary
    componentIndexList = {}

    # Check string input
    if type(componentList) == str or type(componentList) == six.text_type:
        componentList = [componentList]
    # Get selection if componentList is empty
    if not componentList:
        componentList = cmds.ls(sl=True, fl=True) or []
    if not componentList:
        return []
    # Get MSelectionList
    selList = OpenMaya.MSelectionList()
    for i in componentList:
        selList.add(str(i))
    # Iterate through selection list
    selPath = OpenMaya.MDagPath()
    componentObj = OpenMaya.MObject()
    componentSelList = OpenMaya.MSelectionList()
    for i in range(selList.length()):
        # Check for valid component selection
        selList.getDagPath(i, selPath, componentObj)
        if componentObj.isNull():
            # Clear component MSelectionList
            componentSelList.clear()
            # Get current object name
            objName = selPath.partialPathName()

            # Transform
            if selPath.apiType() == OpenMaya.MFn.kTransform:
                numShapesUtil = OpenMaya.MScriptUtil()
                numShapesUtil.createFromInt(0)
                numShapesPtr = numShapesUtil.asUintPtr()
                selPath.numberOfShapesDirectlyBelow(numShapesPtr)
                numShapes = OpenMaya.MScriptUtil(numShapesPtr).asUint()
                selPath.extendToShapeDirectlyBelow(numShapes - 1)
            # Mesh
            if selPath.apiType() == OpenMaya.MFn.kMesh:
                meshFn = OpenMaya.MFnMesh(selPath.node())
                vtxCount = meshFn.numVertices()
                componentSelList.add(objName + ".vtx[0:" + str(vtxCount - 1) + "]")
            # Curve
            elif selPath.apiType() == OpenMaya.MFn.kNurbsCurve:
                curveFn = OpenMaya.MFnNurbsCurve(selPath.node())
                componentSelList.add(objName + ".cv[0:" + str(curveFn.numCVs() - 1) + "]")
            # Surface
            elif selPath.apiType() == OpenMaya.MFn.kNurbsSurface:
                surfaceFn = OpenMaya.MFnNurbsSurface(selPath.node())
                toAdd = "{0}.cv[0:{1}][0:{2}]".format(
                    objName,
                    surfaceFn.numCVsInU() - 1,
                    surfaceFn.numCVsInV() - 1,
                )
                componentSelList.add(toAdd)

            # Lattice
            elif selPath.apiType() == OpenMaya.MFn.kLattice:
                sDiv = cmds.getAttr(objName + ".sDivisions")
                tDiv = cmds.getAttr(objName + ".tDivisions")
                uDiv = cmds.getAttr(objName + ".uDivisions")

                toAdd = "{0}.pt[0:{1}][0:{2}][0:{3}]".format(
                    objName, sDiv - 1, tDiv - 1, uDiv - 1
                )
                componentSelList.add(toAdd)

            # Get object component MObject
            componentSelList.getDagPath(0, selPath, componentObj)
        # =======================
        # - Check Geometry Type -
        # =======================
        # MESH / NURBS CURVE
        if (selPath.apiType() == OpenMaya.MFn.kMesh) or (
            selPath.apiType() == OpenMaya.MFn.kNurbsCurve
        ):
            indexList = OpenMaya.MIntArray()
            componentFn = OpenMaya.MFnSingleIndexedComponent(componentObj)
            componentFn.getElements(indexList)
            componentIndexList[selPath.partialPathName()] = list(indexList)
        # NURBS SURFACE
        if selPath.apiType() == OpenMaya.MFn.kNurbsSurface:
            indexListU = OpenMaya.MIntArray()
            indexListV = OpenMaya.MIntArray()
            componentFn = OpenMaya.MFnDoubleIndexedComponent(componentObj)
            componentFn.getElements(indexListU, indexListV)
            componentIndexList[selPath.partialPathName()] = list(
                zip(list(indexListU), list(indexListV))
            )
        # LATTICE
        if selPath.apiType() == OpenMaya.MFn.kLattice:
            indexListS = OpenMaya.MIntArray()
            indexListT = OpenMaya.MIntArray()
            indexListU = OpenMaya.MIntArray()
            componentFn = OpenMaya.MFnTripleIndexedComponent(componentObj)
            componentFn.getElements(indexListS, indexListT, indexListU)
            componentIndexList[selPath.partialPathName()] = list(
                zip(list(indexListS), list(indexListT), list(indexListU))
            )
    # Return Result
    return componentIndexList


# -------------------------------------------------------------------------------------------
# get UV Map
def getMapForSelectedVerticesFromSelection(normalize=True, opp=False, axis="uv"):
    # Get MSelectionList
    selList = OpenMaya.MSelectionList()
    OpenMaya.MGlobal.getActiveSelectionList(selList)

    iterSel = OpenMaya.MItSelectionList(selList)

    util = OpenMaya.MScriptUtil()
    util.createFromList([0.0, 0.0], 2)
    uvPoint = util.asFloat2Ptr()
    indicesValues = []
    while not iterSel.isDone():
        component = OpenMaya.MObject()
        dagPath = OpenMaya.MDagPath()
        iterSel.getDagPath(dagPath, component)
        if not component.isNull():
            componentFn = OpenMaya.MFnComponent(component)
            if componentFn.componentType() == OpenMaya.MFn.kMeshVertComponent:  # vertex
                vertIter = OpenMaya.MItMeshVertex(dagPath, component)
                while not vertIter.isDone():
                    theVert = vertIter.index()
                    vertIter.getUV(uvPoint)
                    uPt = OpenMaya.MScriptUtil.getFloat2ArrayItem(uvPoint, 0, 0)
                    vPt = OpenMaya.MScriptUtil.getFloat2ArrayItem(uvPoint, 0, 1)
                    indicesValues.append((theVert, uPt, vPt))
                    vertIter.next()
        iterSel.next()
    if normalize:
        maxV = max(indicesValues, key=lambda x: x[2])[2]
        minV = min(indicesValues, key=lambda x: x[2])[2]
        diffV = maxV - minV

        maxU = max(indicesValues, key=lambda x: x[1])[1]
        minU = min(indicesValues, key=lambda x: x[1])[1]
        diffU = maxU - minU

        indicesValues = [
            (vert, (u - minU) / diffU, (v - minU) / diffV)
            for (vert, u, v) in indicesValues
        ]
    if opp:
        indicesValues = [(vert, -1.0 * u, -1.0 * v) for (vert, u, v) in indicesValues]
    if axis != "uv":
        indReturn = "uv".index(axis) + 1
        indicesValues = [(el[0], el[indReturn]) for el in indicesValues]
    return indicesValues


def getMapForSelectedVertices(vertIter, normalize=True, opp=False, axis="uv"):
    # Get MSelectionList
    util = OpenMaya.MScriptUtil()
    util.createFromList([0.0, 0.0], 2)
    uvPoint = util.asFloat2Ptr()
    indicesValues = []

    while not vertIter.isDone():
        theVert = vertIter.index()
        vertIter.getUV(uvPoint)
        uPt = OpenMaya.MScriptUtil.getFloat2ArrayItem(uvPoint, 0, 0)
        vPt = OpenMaya.MScriptUtil.getFloat2ArrayItem(uvPoint, 0, 1)
        indicesValues.append((theVert, uPt, vPt))
        vertIter.next()
    if normalize:
        maxV = max(indicesValues, key=lambda x: x[2])[2]
        minV = min(indicesValues, key=lambda x: x[2])[2]
        diffV = maxV - minV

        maxU = max(indicesValues, key=lambda x: x[1])[1]
        minU = min(indicesValues, key=lambda x: x[1])[1]
        diffU = maxU - minU

        indicesValues = [
            (vert, (u - minU) / diffU, (v - minU) / diffV)
            for (vert, u, v) in indicesValues
        ]
    if opp:
        indicesValues = [(vert, 1.0 - u, 1.0 - v) for (vert, u, v) in indicesValues]
    if axis != "uv":
        indReturn = "uv".index(axis) + 1
        indicesValues = [(el[0], el[indReturn]) for el in indicesValues]
    return indicesValues


# -------------------------------------------------------------------------------------------
# ------------------------ callBacks --------------------------------------------------------
# -------------------------------------------------------------------------------------------
def deleteTheJobs(toSearch="BrushFunctions.callAfterPaint"):
    res = cmds.scriptJob(listJobs=True)
    for job in res:
        if toSearch in job:
            jobIndex = int(job.split(":")[0])
            cmds.scriptJob(kill=jobIndex)


def addNameChangedCallback(callback):
    def omcallback(mobject, oldname, _):
        newname = OpenMaya.MFnDependencyNode(mobject).name()
        callback(oldname, newname)  #

    listenTo = OpenMaya.MObject()
    return OpenMaya.MNodeMessage.addNameChangedCallback(listenTo, omcallback)


def addNameDeletedCallback(callback):
    def omcallback(mobject, _):
        nodeName = OpenMaya.MFnDependencyNode(mobject).name()
        callback(nodeName)  #

    listenTo = OpenMaya.MObject()
    return OpenMaya.MNodeMessage.addNodeAboutToDeleteCallback(listenTo, omcallback)


def removeNameChangedCallback(callbackId):
    OpenMaya.MNodeMessage.removeCallback(callbackId)

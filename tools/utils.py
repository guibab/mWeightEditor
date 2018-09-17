from maya import cmds
import time, datetime
from maya import OpenMaya

###################################################################################
#
#   Global FUNCTIONS
#
###################################################################################


class GlobalContext(object):
    def __init__(
        self,
        raise_error=True,
        message="processing",
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
        if self.doPrint:
            result = "{0} hours {1} mins {2} secs".format(*timeRes)
            print "{0} executed in {1} [{2:.2f} secs]".format(self.message, result, completionTime)
        if exc_type is not None:
            if self.raise_error:
                import traceback

                traceback.print_tb(exc_tb)
                raise exc_type, exc_val
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


def deleteTheJobs(toSearch="BrushFunctions.callAfterPaint"):
    res = cmds.scriptJob(listJobs=True)
    for job in res:
        if toSearch in job:
            jobIndex = int(job.split(":")[0])
            cmds.scriptJob(kill=jobIndex)


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
    except:
        return []
    count = componentFn.elementCount()
    # elementIndicesWeights = [ (componentFn.element(i), componentFn.weight(i).influence() ) for i in range (count)]
    elementIndices = [componentFn.element(i) for i in range(count)]
    elementWeights = [componentFn.weight(i).influence() for i in range(count)]
    return elementIndices, elementWeights

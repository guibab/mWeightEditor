from maya import cmds
import time, datetime

###################################################################################
#
#   Global FUNCTIONS
#
###################################################################################


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

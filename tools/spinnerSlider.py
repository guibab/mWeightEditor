from Qt import QtGui, QtCore, QtWidgets
from maya import cmds


###################################################################################
#
#   the slider setting
#
###################################################################################
class ValueSetting(QtWidgets.QWidget):
    theStyleSheet = "QDoubleSpinBox {color: black; background-color:rgb(200,200,200) ; border: 1px solid black;text-align: center;}"

    def __init__(self, parent=None, singleStep=0.1, precision=1):
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
        # print "value Set {0}".format (theVal)

        self.mainWindow.prepareToSetValue()
        self.mainWindow.doAddValue(theVal / 100.0)
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

        self.theSpinner.setValue(theVal * 100.0)

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

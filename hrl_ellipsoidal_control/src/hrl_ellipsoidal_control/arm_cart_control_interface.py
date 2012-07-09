#! /usr/bin/python

import sys
from PyQt4 import QtCore, QtGui, uic
import functools

import roslib
roslib.load_manifest("rospy")
roslib.load_manifest("std_msgs")
roslib.load_manifest("std_srvs")
import rospy
from std_msgs.msg import String
from std_srvs.srv import Empty, EmptyResponse

from arm_cart_control_gui import Ui_Frame as QTArmControlGUIFrame

MOVE_BUTTONS = ['translate_up', 
                'translate_down',
                'translate_left', 
                'translate_right', 
                'translate_in', 
                'translate_out',
                'rotate_x_pos', 
                'rotate_x_neg', 
                'rotate_y_pos', 
                'rotate_y_neg', 
                'rotate_z_pos', 
                'rotate_z_neg'] 

TEXT_BUTTONS = ['reset_rotation']

BUTTON_STYLESHEET = """image: url(:/resources/%s_%s.png);
                       background-image: url(:/resources/empty.png);"""
MONITOR_RATE = 20.

class ArmCartControlGUIFrame(QtGui.QFrame):
    def __init__(self):
        super(ArmCartControlGUIFrame, self).__init__()
        self.enable_buttons = False
        self.disable_buttons = False
        self.set_ctrl_name = None
        self.set_status = None
        self.hide_button = None
        self.show_button = None

        self.button_clk_pub = rospy.Publisher("/arm_ctrl_gui/button_clk", String)
        self.buttons_enable_srv = rospy.Service("/arm_ctrl_gui/buttons_enable", Empty, 
                                                self._buttons_enable_cb)
        self.buttons_disable_srv = rospy.Service("/arm_ctrl_gui/buttons_disable", Empty,
                                                 self._buttons_disable_cb)
        self.set_ctrl_name_sub = rospy.Subscriber("/arm_ctrl_gui/set_controller_name", String,
                                                  self._set_ctrl_name_cb)
        self.set_status_sub = rospy.Subscriber("/arm_ctrl_gui/set_status", String,
                                               self._set_status_cb)
        self.hide_button_sub = rospy.Subscriber("/arm_ctrl_gui/hide_button", String,
                                                self._hide_button_cb)
        self.show_button_sub = rospy.Subscriber("/arm_ctrl_gui/show_button", String,
                                                self._show_button_cb)

        self.init_ui()

    def init_ui(self):
        self.ui = QTArmControlGUIFrame()
        self.ui.setupUi(self)
        self.buttons_enabled()
        for button in MOVE_BUTTONS + TEXT_BUTTONS:
            _publish_button_clk = functools.partial(self._publish_button_clk, button)
            exec("self.ui.%s.clicked.connect(_publish_button_clk)" % button)

        self.monitor_timer = QtCore.QTimer(self)
        QtCore.QObject.connect(self.monitor_timer, QtCore.SIGNAL("timeout()"), self.monitor_cb)
        self.monitor_timer.start(MONITOR_RATE)

    def _buttons_enable_cb(self, req):
        self.enable_buttons = True
        return EmptyResponse()

    def _buttons_disable_cb(self, req):
        self.disable_buttons = True
        return EmptyResponse()

    def _publish_button_clk(self, button):
        self.button_clk_pub.publish(button)

    def _set_ctrl_name_cb(self, msg):
        self.set_ctrl_name = msg.data

    def _set_status_cb(self, msg):
        self.set_status = msg.data

    def _hide_button_cb(self, msg):
        self.hide_button = msg.data

    def _show_button_cb(self, msg):
        self.show_button = msg.data

    def monitor_cb(self):
        if self.enable_buttons:
            self.buttons_enabled(True)
            self.enable_buttons = False
        if self.disable_buttons:
            self.buttons_enabled(False)
            self.disable_buttons = False
        if self.set_ctrl_name is not None:
            self.ui.controller_name.setText(self.set_ctrl_name)
            self.set_ctrl_name = None
        if self.set_status is not None:
            self.ui.status_text.setText(self.set_status)
            self.set_status = None
        if self.hide_button is not None:
            exec("self.ui.%s.hide()" % self.hide_button)
            self.hide_button = None
        if self.show_button is not None:
            exec("self.ui.%s.show()" % self.show_button)
            self.show_button = None

    def buttons_enabled(self, enabled=True):
        for button in MOVE_BUTTONS:
            exec("cur_button = self.ui.%s" % button)
            if enabled:
                cur_button.setEnabled(True)
                cur_button.setStyleSheet(BUTTON_STYLESHEET % (button, 'on'))
            else:
                cur_button.setDisabled(True)
                cur_button.setStyleSheet(BUTTON_STYLESHEET % (button, 'off'))
        for button in TEXT_BUTTONS:
            exec("cur_button = self.ui.%s" % button)
            if enabled:
                cur_button.setEnabled(True)
            else:
                cur_button.setDisabled(True)

def main():
    rospy.init_node("arm_cart_control_interface")
    app = QtGui.QApplication(sys.argv)
    frame = ArmCartControlGUIFrame()
    frame.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

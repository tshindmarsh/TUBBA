#!/usr/bin/env python3
"""
TUBBA - Tracking and Understanding Behavior-Based Analysis
Main entry point for the application
"""

import sys
from PyQt5.QtWidgets import QApplication
from gui.TUBBA_launch import TUBBALauncher

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TUBBALauncher()
    window.show()
    sys.exit(app.exec_())

# -*- coding: utf-8 -*-
import sys
import os

# Add the root directory to sys.path so that 'app' can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import gui

if __name__ == "__main__":
    gui.main()

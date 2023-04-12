#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 14:59:20 2023

@author: akshay
"""

import subprocess,main
import webbrowser as web
def launch(port="localhost:4549"):
    web.open_new('http://127.0.0.1:4549/')
    subprocess.run(["gunicorn","main:server","-b",port])

launch()



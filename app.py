#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 16:29:02 2021

@author: akshay
"""

import dash
import dash_bootstrap_components as dbc


external_stylesheets=[dbc.themes.DARKLY,
                      'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css']
app = dash.Dash(__name__, suppress_callback_exceptions=True,external_stylesheets=external_stylesheets)
app.config['suppress_callback_exceptions'] = True
#server = app.server
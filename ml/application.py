import random

import aif360.sklearn.metrics



from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

from Tabs import *
from Baseline_tab import baseline_get_tab_dcc, baseline_get_app_callbacks
from Different_distribution_tab import dd_get_tab_dcc, dd_get_app_callbacks
from Bias_tab import bias_get_tab_dcc, bias_get_app_callbacks


def get_app_callbacks(app):
    @app.callback(Output(TABS_DIV_ID, 'children'), Input(TABS_HEADER_ID, 'value'))
    def render_content(tab):
        tab_mapping = {
            TAB_BASELINE: [baseline_get_tab_dcc],
            TAB_DD: [dd_get_tab_dcc],
            TAB_BIAS: [bias_get_tab_dcc]
        }
        # Return requested tab
        tab_layout = tab_mapping[tab][0]()
        return tab_layout

    # Get all app callbacks for each tab
    baseline_get_app_callbacks(app)  # Pass on shared variables/objects to all your callbacks in different files
    dd_get_app_callbacks(app)
    bias_get_app_callbacks(app)

if __name__ == '__main__':
    START_TAB = TAB_BASELINE  # Choose here which tab to show when loading in app

    # app = Dash(external_stylesheets=[dbc.themes.LUX])
    app = Dash(external_stylesheets=[dbc.themes.LUX])
    app.layout = get_tab_layout(TABS_HEADER_ID, START_TAB, all_tabs, TABS_DIV_ID, top=0, layout_id="app_layout")

    # Get all app callbacks for each tab
    get_app_callbacks(app)
    app.run_server(debug=True, use_reloader=False)

from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc

from Tabs import *
from Baseline_tab import baseline_get_tab_dcc, baseline_get_app_callbacks
from Different_distribution_tab import dd_get_tab_dcc, dd_get_app_callbacks
from Bias_tab import bias_get_tab_dcc, bias_get_app_callbacks
from Upload_tab import upload_get_tab_dcc, upload_get_app_callbacks
# TODO:
import Upload_tab


def get_app_callbacks(app):
    @app.callback(Output(TABS_DIV_ID, 'children'), Input(TABS_HEADER_ID, 'value'))
    def render_content(tab):
        tab_mapping = {
            # TODO:
            # TAB_UPLOAD: [upload_get_tab_dcc],
            TAB_UPLOAD: [Upload_tab.upload_get_tab_dcc],
            #
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
    # TODO:
    # upload_get_app_callbacks(app)
    Upload_tab.upload_get_app_callbacks(app)

if __name__ == '__main__':
    START_TAB = TAB_UPLOAD  # Choose here which tab to show when loading in app

    # app = Dash(external_stylesheets=[dbc.themes.LUX])
    app = Dash(suppress_callback_exceptions=True)
    app.layout = get_tab_layout(TABS_HEADER_ID, START_TAB, all_tabs, TABS_DIV_ID, top=0, layout_id="app_layout")

    # Get all app callbacks for each tab
    get_app_callbacks(app)
    app.run_server(debug=True, use_reloader=False)

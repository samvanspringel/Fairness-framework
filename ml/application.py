import random

import aif360.sklearn.metrics

# Modellen
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import plotly.graph_objects as go

import hiring_ml as hire
from Tabs import *
from Baseline_tab import baseline_get_tab_dcc, baseline_get_app_callbacks
from Different_distribution_tab import dd_get_tab_dcc, dd_get_app_callbacks
from Bias_tab import bias_get_tab_dcc, bias_get_app_callbacks




# Base scenario
def load_scenario(scenario):
    environment = hire.setup_environment(scenario)
    training_data = hire.generate_training_data(environment, 1000)
    test_data = hire.rename_goodness(hire.generate_test_data(environment, 1000))

    trained_models = hire.train_models(training_data, models)
    predictions = hire.make_predictions(test_data, trained_models)

    sensitive_features = ["gender"]
    output_label = "hired"
    fairness_notions = hire.calculate_fairness(predictions, sensitive_features, output_label)

    dataframes_count_hired = hire.count_hired(predictions)
    cm = hire.generate_cm(predictions)

    scenarios_elements[scenario] = {'Dataset': {'cm-women-hired': cm[0],
                                                'cm-men-hired': cm[1],
                                                'df-hired': dataframes_count_hired[0],
                                                'fairness': fairness_notions[0]
                                                },

                                    'Decision tree': {'cm-women-hired': cm[2],
                                                      'cm-men-hired': cm[3],
                                                      'df-hired': dataframes_count_hired[1],
                                                      'fairness': fairness_notions[1]
                                                      },
                                    'k-Nearest neighbours': {'cm-women-hired': cm[4],
                                                             'cm-men-hired': cm[5],
                                                             'df-hired': dataframes_count_hired[2],
                                                             'fairness': fairness_notions[2]},
                                    'Preview': hire.make_preview(test_data),
                                    }


# Gebruikte modellen
models = [KNeighborsClassifier(n_neighbors=3), tree.DecisionTreeClassifier()]
scenarios_elements = {}
# Base
load_scenario('Base')
# Different distribution
load_scenario('Different distribution')
# Bias scenario
load_scenario('Bias')


def get_app_callbacks(app, scenarios_elements):
    @app.callback(Output(TABS_DIV_ID, 'children'), Input(TABS_HEADER_ID, 'value'))
    def render_content(tab):
        tab_mapping = {
            TAB_DATA: data_get_tab_dcc,
            TAB_BASELINE: baseline_get_tab_dcc,
            TAB_DD: dd_get_tab_dcc,
            TAB_BIAS: bias_get_tab_dcc
        }
        # Return requested tab
        tab_layout = tab_mapping[tab](scenarios_elements)
        return tab_layout

    # Get all app callbacks for each tab
    baseline_get_app_callbacks(app, scenarios_elements)  # Pass on shared variables/objects to all your callbacks in different files
    dd_get_app_callbacks(app, scenarios_elements)
    bias_get_app_callbacks(app, scenarios_elements)

if __name__ == '__main__':
    START_TAB = TAB_BASELINE  # Choose here which tab to show when loading in app

    app = Dash()
    app.layout = get_tab_layout(TABS_HEADER_ID, START_TAB, all_tabs, TABS_DIV_ID, top=0, layout_id="app_layout")

    # Get all app callbacks for each tab
    get_app_callbacks(app, scenarios_elements)
    app.run_server(debug=True, use_reloader=False)

from dash import dcc, html

# Tabs object
TABS_HEADER_ID = "tabs_header"
TABS_DIV_ID = "tabs_div"
# Tab IDs
TAB_HOME = "tab_home"
TAB_BASELINE = "tab_baseline"
TAB_DD = "tab_dd"
TAB_BIAS = "tab_bias"

all_tabs = {
    TAB_HOME: "Home",
    TAB_BASELINE: "Baseline",
    TAB_DD: "Different distribution",
    TAB_BIAS: "Bias"
}


def get_tab_layout(tab_id, start_tab, tab_id_names, tab_div_id, top="0%", layout_id=None):
    background = "#ffffff"  # TODO based on dark_mode
    children = [
        html.Div([
            dcc.Tabs(id=tab_id, value=start_tab,
                     children=[dcc.Tab(label=name, value=tid) for tid, name in tab_id_names.items()],
                     style={"background": background}
                     ),
        ],
            style={'position': 'sticky', "z-index": "999", "width": "100%", "background": background, "top": f"{top}"}
        ),
        html.Div(id=tab_div_id),
    ]
    return html.Div(children, id=layout_id) if layout_id else html.Div(children)


# Helper function for horizontal div
def horizontal_div(elements, width="15%", space_width="5%", style=None):
    """Put elements next to each other, with given width of window size"""
    if not isinstance(elements, list):
        elements = [elements]
    div = html.Div(children=[])
    if isinstance(width, str):
        width = [width] * len(elements)
    if isinstance(space_width, str):
        space_width = [space_width] * len(elements)
    for element, w, sw in zip(elements, width, space_width):
        div.children.append(html.Div(children=[element],
                                     style={'width': w if element is not None else sw,
                                            "verticalAlign": "top", 'display': 'inline-block'}))
    if style:
        div.style = style
    return div

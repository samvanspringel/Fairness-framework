from dash import Dash, dcc, html, Input, Output
import plotly.express as px
from sklearn.metrics import confusion_matrix

# Create a dash app with the current file name as name
app = Dash(__name__)

# Try using variables to keep track of the id of objects in your dash application, so you make fewer mistakes
ID_GRAPH_BAR = "Graph_Bar"
ID_GRAPH_BAR_GROUPED = "Graph_Bar_Grouped"
ID_GRAPH_CM = "Graph_Confusion_Matrix"
ID_DROPDOWN_FEATURE = "Dropdown_Feature"
ID_DROPDOWN_SENSITIVE_FEATURE = "Dropdown_Sensitive_Feature"

# Existing dataset in plotly as an example for the plots
df = px.data.tips()
# Adding classification to df: tipped_enough, whether they tipped at least 20% of the total bill
df_classification = "tipped_enough"
df[df_classification] = (df["tip"] >= (0.2 * df["total_bill"])).astype(int)  # 0 - not enough, 1 - enough
actions = [0, 1]
classification_labels = ["too little", "enough"]
df_sensitive_features = ["sex", "smoker"]
df_attributes = [c for c in df.columns if c not in [df_classification] + df_sensitive_features]

# sampling the classification to get a "prediction" to compare with
df_model_prediction = df[df_classification].sample(frac=1)

# Confusion matrix
cm = confusion_matrix(df[df_classification], df_model_prediction,  # normalize=True,
                      labels=actions)
tn, fp, fn, tp = cm.ravel()

fig_cm = px.imshow(cm, labels=dict(x="Predicted", y="True"), x=classification_labels, y=classification_labels,
                   text_auto=True)
fig_cm.update_xaxes(side="top")


# Set the layout of the app (webpage you open)
app.layout = html.Div([
    html.H1("This is a large header (html.H1)"),
    html.H4('Smaller headers: html.H2, html.H3, html.H4, html.H5, html.H6'),
    html.P("Paragraph of text:"
           "The focus on Machine Learning (ML) algorithms has been on improving accuracy when it comes to "
           "classification. When applying ML to a job hiring setting, we require an algorithm capable of treating "
           "job candidates fairly, by not discriminating based on sensitive features such as race or gender. "
           "Therefore, we require a methodology to detect any bias present as well as suggestions to mitigate this "
           "discrimination and improve the model."),
    html.Br(),  # Add some space between elements
    html.H4("These are dropdowns, to select which feature to plot"),
    html.H5("Feature:"),
    dcc.Dropdown(id=ID_DROPDOWN_FEATURE, options=df_attributes, value=df_attributes[0], clearable=False),
    html.H5("Sensitive feature:"),
    dcc.Dropdown(id=ID_DROPDOWN_SENSITIVE_FEATURE, options=df_sensitive_features, value=df_sensitive_features[0], clearable=False),
    html.H4("Check other components at "),
    html.A(["https://dash.plotly.com/dash-core-components"],href="https://dash.plotly.com/dash-core-components"),
    html.H4("Below are some plotly plots, added in dash graphs:"),
    dcc.Graph(id=ID_GRAPH_BAR),
    dcc.Graph(id=ID_GRAPH_BAR_GROUPED),
    dcc.Graph(id=ID_GRAPH_CM, figure=fig_cm),
])


# This is a callback to make your dash app reactive.
# It looks for if (at least one of) the given inputs [Input(<ID>, <ATTRIBUTE>), ...] changes the given attribute
# (e.g., the dropdown selected a new option) and returns values to update the attributes of the given outputs
# [Output(<ID>, <ATTRIBUTE>), ...]
@app.callback(
    [Output(ID_GRAPH_BAR, "figure"), Output(ID_GRAPH_BAR_GROUPED, "figure")],
    [Input(ID_DROPDOWN_FEATURE, "value"), Input(ID_DROPDOWN_SENSITIVE_FEATURE, "value")])
def update_graph_feature(feature, sensitive_feature):
    if feature in ["total_bill", "tip"]:
        fig_bar = px.histogram(df, x=feature, y=df_classification)
        fig_bar_grouped = px.histogram(df, x=feature, y=df_classification, color=sensitive_feature)
    else:
        # Group and sum df_classification to represent 1 bar per combination
        new_df = df.groupby([feature, sensitive_feature])[df_classification].sum().reset_index()

        fig_bar = px.bar(new_df, x=feature, y=df_classification)
        fig_bar_grouped = px.bar(new_df, x=feature, y=df_classification, color=sensitive_feature, barmode="group")

    return [fig_bar, fig_bar_grouped]


if __name__ == '__main__':

    app.run_server(debug=False, use_reloader=False)
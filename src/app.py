import pandas as pd
import dash
from dash import dcc, html
from dash_html_components import Iframe
from dash.dependencies import Input, Output, State
from dash import callback_context
import plotly.express as px
import folium
import os
import dash_bootstrap_components as dbc
from folium.plugins import MousePosition
from folium.plugins import MarkerCluster
import joblib
import base64
import numpy as np

# Reading data
spacex_df = pd.read_csv("spacex_launch_dash.csv")
max_payload = spacex_df['Payload Mass (kg)'].max()
min_payload = spacex_df['Payload Mass (kg)'].min()
launch_sites = spacex_df['Launch Site'].unique()

spacex_geo = pd.read_csv("spacex_launch_geo.csv")
select_geo = spacex_geo.groupby('Launch Site')[['Lat', 'Long']].mean()
# load ml model
# load_model = pickle.load(open('launch_predict_SpX_DT.sav', 'rb'))
# Load the model and the scaler
loaded_model = joblib.load('launch_predict_SpX_DT.joblib')
loaded_scaler = joblib.load('model_scaler.joblib')
X = pd.read_csv('dataset_part_3.csv')

# Sample data for dropdowns

payload_options = [{'label': i, 'value': i}
                   for i in np.linspace(100, 15000, 10)]
reused_count_options = [{'label': i, 'value': i}
                        for i in X.ReusedCount.unique()]
orbit_options = [{'label': i, 'value': i}
                 for i in spacex_geo['Orbit'].unique()]
launchsite_options = [{'label': i, 'value': i} for i in launch_sites]


def map_lauch_loc(df):
    my_map = folium.Map(location=[40, -85],
                        zoom_start=3,
                        # max_bounds=True,
                        min_zoom=2)
    # tiles='cartodbpositron')
    marker_cluster = MarkerCluster()
    my_map.add_child(marker_cluster)
    # Add Mouse Position to get the coordinate (Lat, Long) for a mouse over on the map
    formatter = "function(num) {return L.Util.formatNum(num, 5);};"
    mouse_position = MousePosition(
        position='topright',
        separator=' Long: ',
        empty_string='NaN',
        lng_first=False,
        num_digits=20,
        prefix='Lat:',
        lat_formatter=formatter,
        lng_formatter=formatter,
    )
    my_map.add_child(mouse_position)
    for row in df.itertuples():
        marker = folium.Marker(location=[row.Lat, row.Long], popup=row[0])
        # marker.add_to(my_map)
        marker.add_to(marker_cluster)

    return my_map

  


min_slid = 0
max_slid = 10000
# Create Dash app
app = dash.Dash(__name__)

# Create the dropdown menu options
dropdown_options = [{'label': i, 'value': i} for i in launch_sites]
dropdown_options.append({'label': 'All sites', 'value': 'All sites'})
map_html = ''
# Define app layout
app.layout = html.Div(children=[
    html.H1('SpaceX Launch Records Dashboard',
            style={'textAlign': 'center', 'color': '#503D36', 'font-size': 40}),
    html.Div([
        html.Label("Select launch site:"),
        dcc.Dropdown(
            id='site-dropdown',
            options=dropdown_options,
            value='All sites',
        )
    ]),
    html.Br(),
    # html.Iframe(id='map',srcDoc=map_html,width='50%',height='600'),
    # html.Div([dcc.Graph(id='success-pie-chart')], style={'display': 'flex'}),
    dbc.Row([
        html.Iframe(id='map', srcDoc=map_html, width='50%', height='600'),
        dcc.Graph(id='success-pie-chart')
    ], style={'display': 'flex'}),

    html.Br(),
    html.Div([html.Div([
        html.H1('SpaceX Launch prediction', style={
                'textAlign': 'center', 'color': '#503D36', 'font-size': 40}),

        # First Column of Dropdowns

        html.Div([
            html.Label("Select Payload:"),
            dcc.Slider(
                id='slider',
                min=min_slid,
                max=max_slid,
                marks={i: str(i) for i in range(
                    int(min_slid), int(max_slid) + 1, 2000)},
                value=min_slid
            ),


            html.Label("Select Reused Count:"),
            dcc.Dropdown(
                id='reused-count-dropdown',
                options=reused_count_options,
                value=reused_count_options[0]['value']
            ),
        ], style={'width': '50%', 'display': 'inline-block'}),

        # Second Column of Dropdowns
        html.Div([
            html.Label("Select Orbit:"),
            dcc.Dropdown(
                id='orbit-dropdown',
                options=orbit_options,
                value=orbit_options[0]['value']
            ),

            html.Label("Select Launch Site:"),
            dcc.Dropdown(
                id='launchsite-dropdown',
                options=launchsite_options,
                value=launchsite_options[0]['value']
            ),
        ], style={'width': '50%', 'display': 'inline-block'}),

        html.Button('Predict the Success', id='generate-btn', n_clicks=0),
    ], style={'width': '50%', 'display': 'inline-block'}),

        # Second Half
        # html.Label[title=f'Selected Options: {selected_option1}, {selected_option2}')]
        # html.Label("Prediction Result:", id='prediction-label'),
        # html.Div([
        #     html.Img(id='output_fig')
        # ], style={'margin-top': '20px'})]),

        html.Div([
            html.Label("Prediction Result:"),
            html.Label(id='prediction-label'),
            html.Img(id='output_fig')
        ], style={'margin-right': '40px'})]),

    html.Br(),
    html.P("Payload range (Kg):"),
    dcc.RangeSlider(
        id='payload-slider',
        min=min_payload,
        max=max_payload,
        marks={i: str(i) for i in range(
            int(min_payload), int(max_payload) + 1, 1000)},
        value=[min_payload, max_payload]
    ),
    html.Br(),
    html.Div([dcc.Graph(id='success-payload-scatter-chart')]),
])


# Callback for updating pie chart and map


@app.callback(
    [Output(component_id='map', component_property='srcDoc'),
     Output(component_id='success-pie-chart', component_property='figure')],
    Input(component_id='site-dropdown', component_property='value'))
def update_pie_chart(selected_site):
    if selected_site == 'All sites':
        # Create a pie chart showing success count for all sites

        for_chart = spacex_df.groupby(['Launch Site', 'class'])[
            'class'].count().loc[(slice(None), 1)].to_frame().reset_index().rename(columns={'class': 'class_success'})

        fig_pie = px.pie(for_chart,
                         values='class_success',
                         names='Launch Site',
                         title='succes count of each site'
                         )

        location_map = map_lauch_loc(select_geo)
        location_map.save('SpaceX_lauch_site.html')
        script_dir = os.path.dirname(os.path.realpath(__file__))
        html_file_name = 'SpaceX_lauch_site.html'
        html_file_path = os.path.join(script_dir, html_file_name)
        if os.path.exists(html_file_path):
            map_html = open(html_file_path, 'r').read()
        else:
            map_html = "<p>HTML file not found</p>"
    else:
        # Create a pie chart showing success count for the selected site
        chart_data = spacex_df[spacex_df['Launch Site'] ==
                               selected_site]['class'].value_counts().reset_index()
        chart_data.columns = ['outcome', 'count']
        fig_pie = px.pie(chart_data,
                         values='count',
                         names='outcome',
                         title=f'Success Count at Site - {selected_site}'
                         )
        # Create a map showing the location of the selected launch site
        m = folium.Map(location=[select_geo.loc[selected_site, 'Lat'],
                                 select_geo.loc[selected_site, 'Long']], zoom_start=9)
        folium.Marker([select_geo.loc[selected_site, 'Lat'], select_geo.loc[selected_site, 'Long']],
                      popup=f"{selected_site} Launch Site").add_to(m)
        dff = select_geo.loc[selected_site].to_frame().T
        location_map = map_lauch_loc(dff)
        location_map.save('SpaceX_lauch_site.html')

        script_dir = os.path.dirname(os.path.realpath(__file__))
        html_file_name = 'SpaceX_lauch_site.html'
        html_file_path = os.path.join(script_dir, html_file_name)
        if os.path.exists(html_file_path):
            map_html = open(html_file_path, 'r').read()
        else:
            map_html = "<p>HTML file not found</p>"

    return [map_html, fig_pie]


# Function to encode an image file to base64
def encode_image(image_path):
    with open(image_path, 'rb') as file:
        encoded_image = base64.b64encode(file.read()).decode('utf-8')
    return f'data:image/png;base64,{encoded_image}'


@app.callback(
    [Output(component_id='prediction-label', component_property='children'),
     Output(component_id='output_fig', component_property='src')],
    [Input(component_id='generate-btn', component_property='n_clicks')],
    [State(component_id='slider', component_property='value'),
     State(component_id='reused-count-dropdown', component_property='value'),
     State(component_id='orbit-dropdown', component_property='value'),
     State(component_id='launchsite-dropdown', component_property='value')]
)
def update_output_image(n_clicks, selected_payload, selected_reused_count, selected_orbit, selected_launch_site):
    pr_df = X.iloc[25].copy()

    print(selected_payload)
    pr_df['PayloadMass'] = float(selected_payload)

    pr_df['ReusedCount'] = selected_reused_count
    if selected_orbit in pr_df.index:
        pr_df[selected_orbit] = 1
    if selected_launch_site in pr_df.index:
        pr_df[selected_launch_site] = 1

    scaled_dat = loaded_scaler.transform(pr_df.to_numpy().reshape(1, -1))
    if loaded_model.predict(scaled_dat):
        # If 'True', load the image from a local file
        image_path = 'landing_1.gif'  # Replace with the actual path to your image
        prediction_result = f"Successful Launch! selected values {selected_payload}, {selected_reused_count}, {selected_orbit},{selected_launch_site} "
    else:
        # If not 'True', return an empty image (or None)
        image_path = 'crash.gif'  # Replace with the actual path to your image
        prediction_result = f"Launch Failure! selected values {selected_payload}, {selected_reused_count}, {selected_orbit},{selected_launch_site} "

    encoded_image = encode_image(image_path)

    return [prediction_result, encoded_image]


# Callback for updating payload scatter chart


@app.callback(
    Output(component_id='success-payload-scatter-chart',
           component_property='figure'),
    [Input(component_id='site-dropdown', component_property='value'),
     Input(component_id='payload-slider', component_property='value')]
)
def update_payload_chart(selected_site, value):
    if selected_site == 'All sites':
        select_df = spacex_df
    else:
        select_df = spacex_df[spacex_df['Launch Site'] == selected_site]

    filtered_df = select_df.query(
        '`Payload Mass (kg)` >= @value[0] and `Payload Mass (kg)` <= @value[1]')

    fig_scatter = px.scatter(filtered_df,
                             x='Payload Mass (kg)',
                             y='class',
                             color='Booster Version Category',
                             title=f'Correlation between Payload and Success for Site = {selected_site}',

                             )
    fig_scatter.update_traces(marker=dict(size=20))

    return fig_scatter


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

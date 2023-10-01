import pandas as pd
import dash
from dash import dcc, html
from dash_html_components import Iframe
from dash.dependencies import Input, Output
import plotly.express as px
import folium
import os
import dash_bootstrap_components as dbc
from folium.plugins import MousePosition
from folium.plugins import MarkerCluster

# Reading data
spacex_df = pd.read_csv("spacex_launch_dash.csv")
max_payload = spacex_df['Payload Mass (kg)'].max()
min_payload = spacex_df['Payload Mass (kg)'].min()
launch_sites = spacex_df['Launch Site'].unique()

spacex_geo = pd.read_csv("spacex_launch_geo.csv")
select_geo = spacex_geo.groupby('Launch Site')[['Lat', 'Long']].mean()


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

    #    , avg_long = select_geo[['Lat', 'Long']].mean()


# Create Dash app
app = dash.Dash(__name__)
server = app.server

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
    Input(component_id='site-dropdown', component_property='value')
)
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

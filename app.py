import base64
import datetime
import io

import dash
from dash.dependencies import Input, Output, State, MATCH, ALL
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import plotly.express as px
import lasio
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from dash.exceptions import PreventUpdate
import re
import webbrowser





app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config['suppress_callback_exceptions'] = True
server = app.server

SIDEBAR_STYLE = {
	"position": "fixed",
	"top": 0,
	"left": 0,
	"bottom": 0,
	"width": "16rem",
	"padding": "2rem 1rem",
	"background-color": "#f9f9fa",
	"overflow": "scroll",
}

CONTENT_STYLE = {
	"margin-left": "18rem",
	"margin-right": "2rem",
	"padding": "2rem 1rem",
}

sidebar = html.Div([
		html.H3("Visualizador Multifinger", className="display-5", style={'textAlign':'center'}),
		html.Hr(),
		dcc.Upload(
		id='upload-data',
		children=html.Div([
			'Drag and Drop your .las file'
		]),
		style={
			'width': '100%',
			'height': '60px',
			'lineHeight': '60px',
			'borderWidth': '2px',
			'borderStyle': 'dashed',
			'borderRadius': '5px',
			'textAlign': 'center',
		},
		
	),
	html.Hr(),
	html.Div(id='select_fingers'),
	html.Div(id='select_curves'),
	html.Hr(),
	html.Button('Graficar', id='graficar'),
	],
	style=SIDEBAR_STYLE,
	
)

content = html.Div(id="page-content",
				style=CONTENT_STYLE,
				children=[
					dcc.Store(id='stored-data'),
					dcc.Store(id='radios_units'),
					dbc.Row([
						dbc.Col(
								[html.H3('POLAR PLOT', style={'textAlign':'center'}),
								dcc.Graph(id="plot2d", figure={
																'layout': go.Layout(                                
																	xaxis =  {                                     
																		'visible': False
																			 },
																	yaxis = {                              
																	   'visible': False,
																	   
																			}                                            
																							)
																}),
														   
								html.Div(
										dcc.Input(id='polar_center',
												type='number',
												placeholder='Input an MD to plot',
												),
										style=dict(display='flex', justifyContent='center'),
										)
								], width=4),
						dbc.Col([
								html.H3('3D SURFACE', style={'textAlign':'center'}),
								dcc.Graph(id="plot3d", figure={
																'layout': go.Layout(                                
																	xaxis =  {                                     
																		'visible': False
																			 },
																	yaxis = {                              
																	   'visible': False,
																	   
																			}                                            
																							)
																}),
								dcc.RangeSlider(id='slider',
												tooltip = { 'always_visible': True },
												),
								], width=6	),
						dbc.Col([
								html.H5('Z aspect ratio'),
								dcc.Input(id='z-aspectratio',
										type='number',
										value=1
										),
								html.Hr(),
								html.H5('X-Y aspect ratio'),
								dcc.Input(id='xy-aspectratio',
										type='number',
										value=1
										),
								html.Hr(),
								], width=2),
							],no_gutters=True, align="center")
						]
						)

app.layout = html.Div([sidebar, content])	   

@app.callback(
			Output('select_fingers', 'children'),
			Output('stored-data', 'data'),
			Output('radios_units', 'data'),
			Input('upload-data', 'contents'),
			State('upload-data', 'filename'),
			)
			  
def update_output(contents, filename):
	if contents is None:
		raise PreventUpdate
	else:
		content_type, content_string = contents.split(',')
		decoded = base64.b64decode(content_string)
		if '.las' in filename or '.LAS' in filename:
			las = lasio.read(io.StringIO(decoded.decode('utf-8')))
			curvesdict = {k:las.curvesdict[k].unit for k in las.curvesdict}
			curvesdict['step'] = abs(las.well.STEP.value)
			df = las.df().reset_index()
			options = [{'label':n, 'value':n} for n in range(100)]
			data = df.to_dict('records')
			children = html.Div([
							html.H5(filename),
							html.Hr(),
							html.H5('Nominal Inner Diameter (mm)'),
							dcc.Input(id='nominal_id',
										type='number',
										placeholder='Input an MD to plot',
										value=104.8,
										),
							html.Hr(),
							dcc.Dropdown(id='depth_index', 
										options = [{'label':c, 'value':c} for c in df.columns],
										placeholder='pick depht'),
							html.Hr(),
							dcc.Dropdown(id='tool_rotation', 
										options = [{'label':c, 'value':c} for c in df.columns],
										placeholder='pick tool rotation'),
							html.Hr(),
							dcc.Dropdown(id='tool_offset', 
										options = [{'label':c, 'value':c} for c in df.columns],
										placeholder='pick tool offset'),
							dcc.Dropdown(id='tool_theta', 
										options = [{'label':c, 'value':c} for c in df.columns],
										placeholder='pick tool angle'),
							html.Hr(),
							dcc.Dropdown(id='fingers_n', 
										options = options, 
										placeholder='pick number of fingers'),
							])
		return children, data, curvesdict


@app.callback(
			Output("select_curves", "children"),
			Input("fingers_n", "value"),
			State('stored-data','data'),
			)

def curves_selection(n_fingers, data):
	if n_fingers is not None:
		df = pd.DataFrame(data)
		options = [{'label':c, 'value':c} for c in df.columns]
		return [
				html.Hr(),
				html.Div(id='curvas', children=[
												dcc.Dropdown(id={
																	'type': 'filter-dropdown',
																	'index': i
																}, 
																options=options, 
																placeholder='finger_{}'.format(i+1), 
																# value='FING{:02d}'.format(i+1),
															) 
												for i in range(n_fingers)],
						),
				]
@app.callback(
			Output({'type': 'filter-dropdown', 'index': ALL}, 'value'),
			Input({'type': 'filter-dropdown', 'index': ALL}, 'value'),
			State({'type': 'filter-dropdown', 'index': ALL}, 'id'),
			State("fingers_n", "value"),
			)

def find_regex(allvalues, allindex, n_fingers):
	try:
		index, value = [(i,v) for i,v in enumerate(allvalues) if v is not None][0]
		
		notnumber = re.sub(r"\d+", '#$#', value)
		number = re.sub(r'\D', '', value)
		if index<9 and number==str(index):
			final_values = [notnumber.replace('#$#', str(i)) for i in range(n_fingers)]
		elif index<9 and number=='0'+str(index+1): 
			final_values = [notnumber.replace('#$#', '{:02d}'.format(i+1)) for i in range(n_fingers)]
		elif index<9 and number=='0'+str(index):
			final_values = [notnumber.replace('#$#', '{:02d}'.format(i)) for i in range(n_fingers)]
		return final_values
	except:
		# raise PreventUpdate
		return [None for i in range(n_fingers)]


@app.callback(
			Output('plot3d', 'figure'),
			Output('plot2d', 'figure'),
			Output('slider', 'min'),
			Output('slider', 'max'),
			Output('slider', 'value'),
			Output('polar_center', 'value'),
			Output('polar_center', 'step'),
			Input('graficar', 'n_clicks'),
			Input('slider', 'value'),
			Input('polar_center', 'value'),
			Input('z-aspectratio', 'value'),
			Input('xy-aspectratio', 'value'),
			State({'type': 'filter-dropdown', 'index': ALL}, 'value'),
			State('stored-data','data'),
			State('depth_index', 'value'),
			State('tool_rotation', 'value'),
			State('tool_offset', 'value'),
			State('tool_theta', 'value'),
			State('nominal_id', 'value'),
			State('radios_units', 'data'),
			)
		
def plot_graf(n_clicks, range_values, polar_center, zratio, xyratio, fingers, data, depth, rot, offset, angle, nomid, curvesdict):

	unit = curvesdict[fingers[0]]
	step = curvesdict['step']
	if unit=='IN':
		factor = 25.4
	else:
		factor = 1
	ctx = dash.callback_context
	trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
	
	df = pd.DataFrame(data).sort_values(depth).set_index(depth).dropna()
	
	if trigger_id=='graficar':
		radios = df[fingers]
		if rot is not None:
			rot3d = df[rot]
		
	else:
		i_min = np.searchsorted(df.index, range_values[0], side="left")
		i_max = np.searchsorted(df.index, range_values[1], side="left")
		radios = df.iloc[i_min:i_max][fingers]
		if rot is not None:
			rot3d = df.iloc[i_min:i_max][rot]
		
	radios = radios*factor
	
	min, max = radios.index.min(), radios.index.max()
	
	nmediciones, npatines = radios.shape
	
	radios_casing = np.full(radios.shape, nomid/2)
	
	diff = radios - radios_casing 
	
	Z = np.vstack([radios.index]*npatines)
	p = np.linspace(0, 2*np.pi, npatines)
	P = np.column_stack([p]*nmediciones) 
	if rot is not None:
		P = P + np.radians(rot3d.values)
	X, Y = radios.values.transpose()*np.cos(P), radios.values.transpose()*np.sin(P)
	

	fig3d = go.Figure(data=[go.Surface(x=X, y=Y, z=Z,
										surfacecolor=diff.transpose(),
										colorscale='Jet',
										cmin=-5,
										cmax=5,
										# customdata=,
										hovertemplate='z: %{z:.2f}<extra></extra>'+
										'<br><b>z*2</b>: %{z:.2f}<br>',
										# text=['ovalizacion: {}'.format(i) for i in Z[:,0]],
										)])
	fig3d.update_scenes(xaxis_visible=False, 
						yaxis_visible=False, 
						zaxis_visible=False, 
						xaxis_showgrid=False, 
						yaxis_showgrid=False, 
						zaxis_showgrid=False,
						aspectmode='manual',
						aspectratio=dict(x=xyratio, y=xyratio, z=zratio),
						)

	fig3d.add_trace(go.Scatter3d(x=[X[0,1]], y=[Y[0,1]], z=[Z[0,1]],
                                   mode='markers',
								   marker = dict(size=10,
												color='blue',
												opacity=.8,)))
	xtop, ytop = nomid/2, 0											
	fig3d.add_trace(go.Scatter3d(x=[xtop], y=[ytop], z=[Z[0,1]],
                                   mode='markers',
								   marker = dict(size=10,
												color='grey',
												opacity=.8,)))
	if polar_center is None or trigger_id!='polar_center':
		radios_polar_plot = radios.iloc[0].values
		polar_depth = radios.index[0]
		if rot is not None:
			rot2d = df[rot].loc[polar_depth]
	else:
		i = np.searchsorted(df.index, polar_center, side="right")
		radios_polar_plot = df[fingers].iloc[i].values*factor
		polar_depth = df.index[i]
		if rot is not None:
			rot2d = df[rot].loc[polar_depth]


	radios_polar_casing = np.full(radios_polar_plot.shape, nomid/2)
	diff_polar = radios_polar_plot - radios_polar_casing
	if rot is not None:
		p = p + np.radians(rot2d)
	
	polar_data = pd.DataFrame({
								'theta':[np.degrees(i) for i in p],
								'radios':radios_polar_plot,
								# 'text':['finger_{}'.format(i) for i in range(1, len(fingers)+1)],
								})
	fig2d = px.scatter_polar(polar_data,
							r="radios",
							theta="theta",
							# text='text',
							color=diff_polar,
							color_continuous_scale='jet',
							range_color=[-5,5],
							)
	fig2d.update_traces(marker=dict(size=10),)
	
	fig2d.add_trace(go.Scatterpolar(
									r = [0, polar_data.radios.iloc[0]],
									theta = [0, polar_data.theta.iloc[0]],
									name = "finger_1",
									mode = "lines",
									))
									
	fig2d.add_trace(go.Scatterpolar(
									r = [nomid/2]*len(fingers),
									theta = [np.degrees(i) for i in p],
									name = "casing ID",
									mode = "lines",
									line_color = 'black',
									line_width = 4,
									opacity = .2,
									))
	
	if offset is not None and angle is not None:
		fig2d.add_trace(go.Scatterpolar(
											r = [df[offset].loc[polar_depth]],
											theta = [np.radians(df[angle].loc[polar_depth])],
											text = 'tool_center',
											marker=dict(size=15, color = "magenta", symbol='x'),
											name = "tool_center",
										))
	
	
	fig2d.update(layout_coloraxis_showscale=False)
	fig2d.update_polars(
						radialaxis_range=[0, (nomid//2)+10],
						radialaxis_showticklabels=False,
						bgcolor='white',
						angularaxis_gridcolor='grey',
						radialaxis_gridcolor='white',
						)
	
	return fig3d, fig2d, df.index.min(), df.index.max(), [min, max], polar_depth, step
	


	
	
if __name__ == '__main__':
	url = 'http://127.0.0.1:8050/'
	webbrowser.open(url, new=1, autoraise=True)
	app.run_server(debug=False)# ,dev_tools_ui=False,dev_tools_props_check=False)
	

import pandas as pd
import dash
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import dash_bootstrap_components as dbc
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import consultas_postgres as consulta

# Obtener todos los datos
df_base = consulta.obtener_datos_base().rename(columns={
    "total_inversion_ifc_aprobada_junta_millones_usd": "Inversión Aprobada (millones USD)"
})

# Convertir fechas
if 'fecha_divulgada' in df_base.columns:
    df_base['fecha_divulgada'] = pd.to_datetime(df_base['fecha_divulgada'])
    df_base['Año'] = df_base['fecha_divulgada'].dt.year

# Inicializar app
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Dashboard_IFC"
color_secuencia = px.colors.sequential.Viridis

# Tabla interactiva
table = dash_table.DataTable(
    id='tabla_ifc',
    columns=[{"name": i, "id": i} for i in df_base.columns],
    data=df_base.to_dict('records'),
    page_size=10,
    filter_action="native",
    sort_action="native",
    style_table={'overflowX': 'auto'}
)

def construir_modelo_regresion(df):
    columnas = [
        'inversion_ifc_gestion_riesgo_millones_usd',
        'inversion_ifc_garantia_millones_usd',
        'inversion_ifc_prestamo_millones_usd',
        'inversion_ifc_capital_millones_usd',
        'industria',
        'categoria_ambiental',
        'Inversión Aprobada (millones USD)'
    ]
    df_modelo = df[columnas].dropna()

    X_cat = pd.get_dummies(df_modelo[['industria', 'categoria_ambiental']], drop_first=True)
    X_num = df_modelo[[
        'inversion_ifc_gestion_riesgo_millones_usd',
        'inversion_ifc_garantia_millones_usd',
        'inversion_ifc_prestamo_millones_usd',
        'inversion_ifc_capital_millones_usd'
    ]]
    X = pd.concat([X_num, X_cat], axis=1)
    y = df_modelo['Inversión Aprobada (millones USD)']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    coef_df = pd.DataFrame({
        'Variable': X.columns,
        'Coeficiente': model.coef_
    }).sort_values(by='Coeficiente', ascending=False)

    coef_df['Signo'] = coef_df['Coeficiente'].apply(lambda x: 'Positivo' if x >= 0 else 'Negativo')
    fig_modelo = px.bar(coef_df, x='Coeficiente', y='Variable', orientation='h',
                        color='Signo', color_discrete_map={'Positivo': 'green', 'Negativo': 'crimson'},
                        title="Importancia y dirección de los coeficientes del modelo",
                        height=600)
    fig_modelo.update_layout(yaxis={'categoryorder': 'total ascending'})

    fig_real_vs_pred = go.Figure()
    fig_real_vs_pred.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predicciones', marker=dict(color='blue')))
    fig_real_vs_pred.add_trace(go.Scatter(x=y_test, y=y_test, mode='lines', name='Ideal', line=dict(color='black', dash='dash')))
    fig_real_vs_pred.update_layout(title="Predicciones vs Valores Reales", height=600)

    errores = y_test - y_pred
    fig_errores = make_subplots(rows=1, cols=2, subplot_titles=("Distribución de los errores", "Errores vs Predicción"))
    fig_errores.add_trace(go.Histogram(x=errores, nbinsx=30, marker_color='indianred'), row=1, col=1)
    fig_errores.add_trace(go.Scatter(x=y_pred, y=errores, mode='markers', marker=dict(color='navy', opacity=0.6)), row=1, col=2)
    fig_errores.add_trace(go.Scatter(x=y_pred, y=[0]*len(y_pred), mode='lines', line=dict(color='black', dash='dot')), row=1, col=2)
    fig_errores.update_layout(height=600)

    indicadores = html.Div([
        html.H5("Indicadores del Modelo", className="mt-3"),
        html.P(f"R²: {r2:.4f}"),
        html.P(f"RMSE: {rmse:.4f} millones USD")
    ])

    return dcc.Graph(figure=fig_modelo), dcc.Graph(figure=fig_real_vs_pred), dcc.Graph(figure=fig_errores), indicadores

grafico_modelo, grafico_modelo2, grafico_errores, indicadores_modelo = construir_modelo_regresion(df_base)

# Layout
app.layout = dbc.Container([
    html.H1(" Tendencias y Evolución de los Proyectos de Servicios de Inversión del IFC", className="text-center mt-4 mb-4"),
    dcc.Tabs([
        dcc.Tab(label='1.Introducción', children=[
            html.Div([
                html.P("Este dashboard presenta un análisis de los proyectos de inversión del IFC, "
                       "incluyendo la distribución de inversiones por industria y categoría ambiental, "
                       "así como un modelo predictivo para estimar la inversión total aprobada.")
            ], style={'padding': '20px'})
        ]),
        
        dcc.Tab(label='2. Contexto', children=[
            html.Div([
                html.P("El IFC (International Finance Corporation) es una institución del Grupo Banco Mundial "
                       "que se dedica a promover el desarrollo económico sostenible mediante la inversión en el sector privado.")
            ], style={'padding': '20px'})
        ]),
        
        dcc.Tab(label='3. Planteamiento del Problema', children=[
            html.Div([
                html.P("El objetivo de este análisis es entender las tendencias y patrones en los proyectos de inversión del IFC, "
                       "así como desarrollar un modelo predictivo que permita estimar la inversión total aprobada en función de diversas variables.")
                
        ])
        ]),
        
        dcc.Tab(label='4. Objetivos y Justificación', children=[
            html.Div([
                html.P("El objetivo principal es proporcionar una herramienta que permita a los tomadores de decisiones del IFC "
                       "evaluar y predecir la inversión total aprobada en función de diferentes variables, "
                       "lo que puede ayudar en la planificación y asignación de recursos.")
            ], style={'padding': '20px'})
        ]),
        
        dcc.Tab(label='5. Marcto Teórico', children=[
            html.Div([
                html.P("El análisis de datos y la modelización predictiva son herramientas clave en la toma de decisiones informadas. "
                       "El uso de modelos de regresión lineal permite entender la relación entre variables y hacer predicciones basadas en datos históricos.")
                
            ], style={'padding': '20px'})
        ]),
        
        dcc.Tab(label='6. Metodología', children=[
            html.Div([
                html.P("La metodología utilizada incluye la limpieza y preparación de datos, "
                       "el análisis exploratorio de datos y la construcción de un modelo de regresión lineal. ")
            ], style={'padding': '20px'})
        ]),
        
        
        dcc.Tab(label='7. Resultados y Análisis Final', children=[
            html.H2("Resultados y Análisis Final", className="text-center mt-4 mb-4"),
            dcc.Tabs([

                dcc.Tab(label='Resumen de Inversiones', children=[
                    dbc.Row([
                        dbc.Col([
                            html.Label("Filtrar por Industria:"),
                            dcc.Dropdown(
                                id='filtro_industria',
                                options=[{'label': i, 'value': i} for i in sorted(df_base['industria'].dropna().unique())],
                                value=None,
                                multi=True,
                                placeholder="Selecciona industria(s)"
                            )
                        ], md=6),
                        dbc.Col([
                            html.Label("Filtrar por Estado del Proyecto:"),
                            dcc.Dropdown(
                                id='filtro_estado',
                                options=[{'label': i, 'value': i} for i in sorted(df_base['estado'].dropna().unique())],
                                value=None,
                                multi=True,
                                placeholder="Selecciona estado(s)"
                            )
                        ], md=6)
                    ]),

                    dbc.Row([
                        dbc.Col(dcc.Graph(id='grafico_inversion'), md=6),
                        dbc.Col(dcc.Graph(id='grafico_categoria'), md=6)
                    ]),
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='grafico_industria'), md=6),
                        dbc.Col(dcc.Graph(id='grafico_histograma'), md=6)
                    ]),
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='grafico_anio'), md=6),
                        dbc.Col(dcc.Graph(id='grafico_estado'), md=6)
                    ])
                ]),
                dcc.Tab(label='Análisis por País', children=[
                    html.Label("Ordenar por:", className="mt-3"),
                    dcc.RadioItems(
                        id='orden_paises',
                        options=[
                            {'label': 'Top 10 (Mayor a Menor)', 'value': 'desc'},
                            {'label': 'Top 10 (Menor a Mayor)', 'value': 'asc'}
                        ],
                        value='desc',
                        inline=True
                    ),
                    dcc.Graph(id='fig_pais')
                ]),
                
                dcc.Tab(label='Visualizacion del modelo', children=[html.Div([
                    html.Div([grafico_modelo], style={'margin-bottom': '30px'}),
                    html.Div([grafico_modelo2], style={'margin-bottom': '30px'}),
                    ], style={'padding': '20px'})
                ]),
                
                dcc.Tab(label='Indicadores del Modelo', children=[
                    html.Div([
                    html.Div([indicadores_modelo]),
                    html.Div([grafico_errores], style={'margin-bottom': '30px'}),
                    ], style={'padding': '20px'})
                        
                
                ]),
                

                dcc.Tab(label='Tabla de Datos', children=[table])
            ])
        ]),
        
        dcc.Tab(label='8. Conclusiones', children=[
            html.Div([
                html.P("El análisis revela patrones interesantes en la inversión del IFC, "
                       "y el modelo predictivo proporciona una herramienta valiosa para estimar la inversión total aprobada.")
            ], style={'padding': '20px'})
        ])    
    ]),
], fluid=True)

# Callback 
@app.callback(
    Output('grafico_inversion', 'figure'),
    Output('grafico_categoria', 'figure'),
    Output('grafico_industria', 'figure'),
    Output('grafico_histograma', 'figure'),
    Output('grafico_anio', 'figure'),
    Output('grafico_estado', 'figure'),
    Input('filtro_industria', 'value'),
    Input('filtro_estado', 'value')
)
def actualizar_graficos_resumen(filtro_industria, filtro_estado):
    df = df_base.copy()

    if filtro_industria:
        df = df[df['industria'].isin(filtro_industria)]
    if filtro_estado:
        df = df[df['estado'].isin(filtro_estado)]

    fig1 = px.box(df, x="industria", y="Inversión Aprobada (millones USD)",
                  title="Distribución de la Inversión IFC por Industria",
                  color_discrete_sequence=color_secuencia)

    fig2 = px.bar(df.groupby("categoria_ambiental")["Inversión Aprobada (millones USD)"].sum().reset_index().rename(columns={
        "categoria_ambiental": "Categoría Ambiental",
        "Inversión Aprobada (millones USD)": "Total Inversión (millones USD)"
    }), x="Categoría Ambiental", y="Total Inversión (millones USD)",
           title="Inversión por Categoría Ambiental", text_auto=True,
           color="Categoría Ambiental", color_discrete_sequence=color_secuencia)

    fig3 = px.bar(df.groupby("industria").size().reset_index(name="Número de Proyectos"),
                  x="Número de Proyectos", y="industria", orientation='h',
                  title="Cantidad de Proyectos por Industria",
                  color="industria", color_discrete_sequence=color_secuencia)
    fig3.update_layout(showlegend=False)

    fig4 = px.histogram(df, x="Inversión Aprobada (millones USD)", nbins=30,
                        title="Distribución de la Inversión Total IFC Aprobada",
                        marginal="rug", color_discrete_sequence=color_secuencia)

    fig5 = px.line(df.groupby("Año").size().reset_index(name="Número de Proyectos"),
                   x="Año", y="Número de Proyectos",
                   title="Proyectos Aprobados por Año",
                   color_discrete_sequence=color_secuencia)

    fig6 = px.bar(df.groupby("estado")["Inversión Aprobada (millones USD)"].sum().reset_index().rename(columns={
        "estado": "Estado del Proyecto",
        "Inversión Aprobada (millones USD)": "Total Inversión (millones USD)"
    }), x="Estado del Proyecto", y="Total Inversión (millones USD)",
           title="Estado del Proyecto vs. Monto Invertido",
           text_auto=True, color="Estado del Proyecto",
           color_discrete_sequence=color_secuencia)

    return fig1, fig2, fig3, fig4, fig5, fig6

# Callback para gráfico de países
@app.callback(
    Output('fig_pais', 'figure'),
    Input('orden_paises', 'value')
)
def actualizar_fig_pais(orden):
    top_paises = consulta.top_paises_inversion(orden).rename(columns={
        "pais": "País", "total": "Total Inversión (millones USD)"
    })
    titulo = "Top 10 Países con Mayor Inversión Aprobada" if orden == 'desc' else "Top 10 Países con Menor Inversión Aprobada"

    fig = px.bar(top_paises, x="País", y="Total Inversión (millones USD)",
                 title=titulo, labels={'Total Inversión (millones USD)': "Inversión Total (millones USD)", 'País': "País"},
                 color="Total Inversión (millones USD)",
                 color_continuous_scale=color_secuencia)
    return fig

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)



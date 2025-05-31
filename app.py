
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
from sklearn.ensemble import RandomForestRegressor

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

def construir_modelo_random_forest(df):
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

    # Variables categóricas a dummies
    X_cat = pd.get_dummies(df_modelo[['industria', 'categoria_ambiental']], drop_first=True)
    X_num = df_modelo[[
        'inversion_ifc_gestion_riesgo_millones_usd',
        'inversion_ifc_garantia_millones_usd',
        'inversion_ifc_prestamo_millones_usd',
        'inversion_ifc_capital_millones_usd'
    ]]
    X = pd.concat([X_num, X_cat], axis=1)
    y = df_modelo['Inversión Aprobada (millones USD)']

    # División en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modelo Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Métricas
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Importancia de variables
    coef_df = pd.DataFrame({
        'Variable': X.columns,
        'Importancia': model.feature_importances_
    }).sort_values(by='Importancia', ascending=False)

    fig_modelo = px.bar(coef_df, x='Importancia', y='Variable', orientation='h',
                        color='Importancia', color_continuous_scale='Viridis',
                        title="Importancia de las variables (Random Forest)",
                        height=600)
    fig_modelo.update_layout(yaxis={'categoryorder': 'total ascending'})

    # Gráfico Real vs Predicho
    fig_real_vs_pred = go.Figure()
    fig_real_vs_pred.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predicciones', marker=dict(color='blue')))
    fig_real_vs_pred.add_trace(go.Scatter(x=y_test, y=y_test, mode='lines', name='Ideal', line=dict(color='black', dash='dash')))
    fig_real_vs_pred.update_layout(title="Predicciones vs Valores Reales", height=600)

    # Análisis de errores
    errores = y_test - y_pred
    fig_errores = make_subplots(rows=1, cols=2, subplot_titles=("Distribución de los errores", "Errores vs Predicción"))
    fig_errores.add_trace(go.Histogram(x=errores, nbinsx=30, marker_color='indianred'), row=1, col=1)
    fig_errores.add_trace(go.Scatter(x=y_pred, y=errores, mode='markers', marker=dict(color='navy', opacity=0.6)), row=1, col=2)
    fig_errores.add_trace(go.Scatter(x=y_pred, y=[0]*len(y_pred), mode='lines', line=dict(color='black', dash='dot')), row=1, col=2)
    fig_errores.update_layout(height=600)

    # Indicadores del modelo
    indicadores = html.Div([
        html.H5("Indicadores del Modelo Random Forest", className="mt-3"),
        html.P(f"R²: {r2:.4f}"),
        html.P(f"RMSE: {rmse:.4f} millones USD")
    ])

    return dcc.Graph(figure=fig_modelo), dcc.Graph(figure=fig_real_vs_pred), dcc.Graph(figure=fig_errores), indicadores

# Uso
grafico_modelo, grafico_modelo2, grafico_errores, indicadores_modelo = construir_modelo_random_forest(df_base)

titulo_dashboard = "Tablero Interactivo de Proyectos de Inversión IFC"

descripcion = """
Este tablero interactivo tiene como objetivo analizar las tendencias globales de los proyectos de servicios de inversión impulsados por la Corporación Financiera Internacional (IFC, por sus siglas en inglés). La base de datos utilizada incluye información detallada de inversiones realizadas en más de 100 países y regiones, abarcando todos los continentes. A través de esta herramienta se facilita la visualización y comprensión de los patrones de inversión del IFC, con énfasis en sectores económicos, montos de financiamiento, productos financieros utilizados (como préstamos o capital accionario) y estados de avance de los proyectos.
El tablero permite además filtrar la información por categoría ambiental, país o región, tipo de industria, entre otros factores clave. Esto brinda una visión integral de cómo la IFC contribuye al desarrollo económico global mediante el fortalecimiento del sector privado. Este análisis es especialmente relevante para evaluar las estrategias de financiamiento sostenible y su impacto en países en desarrollo o economías emergentes.
"""

contexto = """
La Corporación Financiera Internacional (IFC) es una institución miembro del Grupo Banco Mundial cuya misión es promover el desarrollo del sector privado en economías en desarrollo a través de inversiones estratégicas. En este contexto, el tablero presenta una visión consolidada de proyectos financiados por el IFC en distintas regiones del mundo, incluyendo América Latina, África, Asia, Europa del Este y Oceanía.
Los datos muestran cómo el IFC ha canalizado recursos hacia sectores clave como instituciones financieras, manufactura, infraestructura, salud, educación, agroindustria y tecnologías de la información. Estos sectores han sido identificados como motores esenciales para el crecimiento sostenible y la reducción de la pobreza. La amplitud geográfica y sectorial de los datos evidencia un enfoque integral del IFC hacia el desarrollo inclusivo. Este tablero proporciona una oportunidad para comprender mejor las dinámicas globales de inversión del IFC, así como su alineación con los Objetivos de Desarrollo Sostenible (ODS) de las Naciones Unidas.
"""

planteamiento_problema = """
A pesar del volumen de información disponible sobre los proyectos del IFC, esta suele encontrarse dispersa y poco accesible para análisis comparativos o visuales. La ausencia de una plataforma unificada que permita explorar de manera interactiva la distribución, características y evolución de estos proyectos representa una barrera para investigadores, formuladores de políticas y actores del sector privado interesados en conocer las prioridades de inversión del IFC.
El problema específico que aborda este tablero es la dificultad para comprender la magnitud y el alcance de las inversiones del IFC a nivel global, tanto en términos de sectores económicos como de impacto geográfico, tipo de financiamiento y consideraciones ambientales. Sin una herramienta de análisis adecuada, es complejo identificar patrones de inversión, evaluar el cumplimiento de metas de desarrollo o detectar oportunidades de mejora. Este tablero busca resolver esta necesidad, organizando y visualizando los datos de forma clara, dinámica y orientada a la toma de decisiones informadas.
"""

objetivos_justificacion = """
El objetivo general de este proyecto es diseñar y desarrollar un tablero interactivo que permita visualizar y analizar las inversiones de la Corporación Financiera Internacional (IFC) en distintos países y sectores del mundo.
Entre los objetivos específicos se incluyen:
Facilitar la exploración por industria, país, estado del proyecto y categoría ambiental.
Permitir comparaciones regionales o sectoriales en cuanto al volumen de inversión.
Ofrecer una herramienta didáctica y analítica que apoye la comprensión de las estrategias de financiamiento internacional.
La justificación de este tablero radica en la importancia de contar con una visualización clara y accesible de los datos de inversión del IFC, lo cual contribuye a la transparencia, la rendición de cuentas y el análisis estratégico. Además, al integrar múltiples variables en un solo entorno visual, se mejora la capacidad para identificar tendencias, detectar vacíos de inversión y analizar el impacto potencial en términos de desarrollo sostenible a nivel global.
"""

marco_teorico = """
Este estudio se fundamenta en teorías relacionadas con la inversión extranjera directa (IED), el financiamiento para el desarrollo y la sostenibilidad. La IFC, como brazo del Grupo Banco Mundial orientado al sector privado, actúa como catalizador del crecimiento económico mediante inversiones que buscan no solo retorno financiero, sino también impacto social y ambiental positivo.
Desde el enfoque de la teoría del desarrollo endógeno, estas inversiones permiten compensar fallas estructurales en países en desarrollo, facilitando el acceso a financiamiento, tecnologías y mejores prácticas de gestión. Además, se consideran los principios de inversión responsable y los estándares ambientales, sociales y de gobernanza (ESG), que guían la actuación del IFC. En este marco, la categoría ambiental de cada proyecto adquiere especial relevancia. Asimismo, la diversidad de productos financieros utilizados (préstamos, capital, garantías) puede analizarse a la luz de la ingeniería financiera para el desarrollo. Este conjunto teórico justifica la importancia de explorar y evaluar los proyectos de inversión del IFC desde una perspectiva multidimensional.
"""

metodologia = """
Para la construcción del tablero interactivo, se utilizó una base de datos obtenida del portal de transparencia del IFC, que incluye información detallada de proyectos de inversión a nivel mundial. Se realizó un proceso de depuración de datos para eliminar registros incompletos y normalizar las categorías de variables como países, industrias, productos financieros y estados del proyecto.
El tratamiento de datos se llevó a cabo en Python, utilizando bibliotecas como Pandas para manipulación tabular y Dash para el desarrollo del entorno interactivo. Se diseñaron distintos paneles de análisis que permiten al usuario explorar las inversiones mediante filtros como industria, tipo de producto, estado del proyecto y categoría ambiental. Los gráficos incluyen visualizaciones de barras, líneas y tablas dinámicas.
La metodología también contempló principios de usabilidad y claridad visual, con el fin de facilitar la interpretación de resultados. Este enfoque permite una exploración flexible de la base de datos, ofreciendo tanto análisis generales como consultas específicas de interés para múltiples públicos.
"""
conclusiones = """
Los resultados del modelo de importancia de variables indican que los factores más determinantes para explicar las inversiones del IFC están relacionados directamente con los montos financieros involucrados, en particular: inversion_ifc_prestamo_millones_usd, inversion_ifc_garantia_millones_usd y inversion_ifc_capital_millones_usd. Estas variables tienen una importancia conjunta superior al 90%, lo que evidencia que los instrumentos financieros utilizados (préstamo, garantía y capital) son el principal motor explicativo en los proyectos analizados. Por el contrario, variables como el tipo de industria o la categoría ambiental tienen un peso marginal en la predicción, lo cual puede interpretarse como una oportunidad para profundizar en el análisis cualitativo de impacto.

El gráfico de predicciones vs. valores reales muestra un buen ajuste general del modelo, con una nube de puntos ajustada a la diagonal ideal. Esto valida la capacidad predictiva del modelo para capturar las dinámicas de inversión del IFC, con algunos valores atípicos que podrían deberse a proyectos con montos excepcionalmente altos o características particulares que no se replican con frecuencia.
"""

# Layout
app.layout = dbc.Container([
    html.H1(titulo_dashboard, className="text-center mt-4 mb-4"),
    dcc.Tabs([
        dcc.Tab(label='1.Introducción', children=[
            html.Div([
                html.P(descripcion)
            ], style={'padding': '20px'})
        ]),
        
        dcc.Tab(label='2. Contexto', children=[
            html.Div([
                html.P(contexto)
            ], style={'padding': '20px'})
        ]),
        
        dcc.Tab(label='3. Planteamiento del Problema', children=[
            html.Div([
                html.P(planteamiento_problema)  
        ])
        ]),
        
        dcc.Tab(label='4. Objetivos y Justificación', children=[
            html.Div([
                html.P(objetivos_justificacion)
            ], style={'padding': '20px'})
        ]),
        
        dcc.Tab(label='5. Marcto Teórico', children=[
            html.Div([
                html.P(marco_teorico)    
            ], style={'padding': '20px'})
        ]),
        
        dcc.Tab(label='6. Metodología', children=[
            html.Div([
                html.P(metodologia)
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
                html.P(conclusiones)
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



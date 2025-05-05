
import pandas as pd
import dash
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import dash_bootstrap_components as dbc
import consultas_postgres as consulta

# Obtener todos los datos
df_base = consulta.obtener_datos_base().rename(columns={
    "total_inversion_ifc_aprobada_junta_millones_usd": "Inversión Aprobada (millones USD)"
})

# Inicializar app
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server
app.title = "Dashboard_IFC"

# Colores consistentes
color_secuencia = px.colors.sequential.Viridis

# Layout y tabs
app.layout = dbc.Container([
    html.H1("Tendencias y Evolución de los Proyectos de Servicios de Inversión del IFC", 
            className="text-center mt-4 mb-4"),

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

        dcc.Tab(label='Tabla de Datos', children=[
            dash_table.DataTable(
                id='tabla_ifc',
                columns=[{"name": i, "id": i} for i in df_base.columns],
                data=df_base.to_dict('records'),
                page_size=10,
                filter_action="native",
                sort_action="native",
                style_table={'overflowX': 'auto'}
            )
        ])
    ])
], fluid=True)

# Callback filtros resumen
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

    df_anio = df.copy()
    df_anio['Año'] = df_anio['fecha_divulgada'].str[:4]
    fig5 = px.line(df_anio.groupby("Año").size().reset_index(name="Número de Proyectos"),
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

# Callback para países
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
                 title=titulo,
                 labels={'Total Inversión (millones USD)': "Inversión Total (millones USD)", 'País': "País"},
                 color="Total Inversión (millones USD)",
                 color_continuous_scale=color_secuencia)
    return fig

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)


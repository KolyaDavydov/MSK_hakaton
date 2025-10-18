import dash
from dash import dcc, html, Dash, register_page, dash_table, callback
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import clickhouse_connect
import sys
sys.path.append("../pages")

from config import CLICK_CONN

register_page(__name__, path="/", name='Аналитика системы водоснабжения')

# Создание клиента к БД
client = clickhouse_connect.get_client(**CLICK_CONN)

result = client.query('SELECT DISTINCT id FROM msk_database.analytic')
unique_ids = [row[0] for row in result.result_rows]

# Кастомная цветовая палитра
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#18BC9C',
    'warning': '#F39C12',
    'danger': '#E74C3C',
    'dark': '#2C3E50',
    'light': '#ECF0F1',
    'hot_in': '#FF6B6B',
    'hot_out': '#FFA726',
    'cold': '#4ECDC4',
    'leak': '#95A5A6',
    'temp_in': '#E74C3C',
    'temp_out': '#3498DB'
}

CARD_STYLE = {
    "border": "none",
    "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.1)",
    "borderRadius": "12px",
    "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
}


# Стилизованные карточки с иконками
id_selection = dbc.Card(
    [
        html.Div(
            [
                html.I(className="fas fa-building fa-lg", style={'color': COLORS['primary'], 'margin-right': '10px'}),
                html.Span('Выбор дома', style={'font-weight': 'bold', 'color': COLORS['dark']})
            ],
            className="d-flex align-items-center mb-2"
        ),
        dcc.Dropdown(
            id="id-dropdown",
            options=[{'label': f'МКД № {id}', 'value': id} for id in unique_ids],
            value=unique_ids[0],
            style={
                'border': 'none',
                'box-shadow': '0 2px 4px rgba(0,0,0,0.1)',
                'position': 'relative',  # Добавляем позиционирование
                'zIndex': 1000  # Высокий z-index для выпадающего списка
            }
        ),
    ],
    body=True,
    style={
        'border': 'none',
        'box-shadow': '0 4px 6px rgba(0,0,0,0.1)',
        'border-radius': '10px',
        'background': 'linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%)',
        'position': 'relative',  # Добавляем позиционирование для карточки
        'zIndex': 999  # Немного меньший z-index для карточки
    }
)


period_selection = dbc.Card(
    [
        html.Div(
            [
                html.I(className="fas fa-calendar-alt fa-lg", style={'color': COLORS['secondary'], 'margin-right': '10px'}),
                html.Span('Период отображения', style={'font-weight': 'bold', 'color': COLORS['dark']})
            ],
            className="d-flex align-items-center mb-2"
        ),
        dcc.RadioItems(
            id='period-radio',
            options=[
                {'label': html.Span([' 1 месяц'], style={'font-weight': '500'}), 'value': 1},
                {'label': html.Span([' 3 месяца'], style={'font-weight': '500'}), 'value': 3},
                {'label': html.Span([' 12 месяцев'], style={'font-weight': '500'}), 'value': 12}
            ],
            value=3,
            inline=True,
            labelStyle={'margin-right': '25px', 'cursor': 'pointer'},
            inputStyle={'margin-right': '5px', 'cursor': 'pointer'},
            style={'margin': '0px 0'}
        ),
    ],
    body=True,
    style={
        'border': 'none',
        'box-shadow': '0 4px 6px rgba(0,0,0,0.1)',
        'border-radius': '10px',
        'background': 'linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%)'
    }
)

# Статистические карточки
stats_cards = dbc.Row([
    dbc.Col(dbc.Card([
        dbc.CardBody([
            html.Div([
                html.I(className="fas fa-fire fa-2x", style={'color': COLORS['hot_in']}),
                html.Div([
                    html.H4("0.0", id="avg-hot-consumption", className="card-value"),
                    html.P("Средний расход ГВС, м³/ч", className="card-label")
                ], style={'margin-left': '15px'})
            ], className="d-flex align-items-center")
        ])
    ], style={'border': 'none', 'box-shadow': '0 4px 6px rgba(0,0,0,0.1)', 'border-radius': '10px'})),
    
    dbc.Col(dbc.Card([
        dbc.CardBody([
            html.Div([
                html.I(className="fas fa-snowflake fa-2x", style={'color': COLORS['cold']}),
                html.Div([
                    html.H4("0.0", id="avg-cold-consumption", className="card-value"),
                    html.P("Средний расход ХВС, м³/ч", className="card-label")
                ], style={'margin-left': '15px'})
            ], className="d-flex align-items-center")
        ])
    ], style={'border': 'none', 'box-shadow': '0 4px 6px rgba(0,0,0,0.1)', 'border-radius': '10px'})),
    
    dbc.Col(dbc.Card([
        dbc.CardBody([
            html.Div([
                html.I(className="fas fa-tint fa-2x", style={'color': COLORS['leak']}),
                html.Div([
                    html.H4("0.0", id="avg-leak", className="card-value"),
                    html.P("Средняя утечка, м³/ч", className="card-label")
                ], style={'margin-left': '15px'})
            ], className="d-flex align-items-center")
        ])
    ], style={'border': 'none', 'box-shadow': '0 4px 6px rgba(0,0,0,0.1)', 'border-radius': '10px'})),
    
    dbc.Col(dbc.Card([
        dbc.CardBody([
            html.Div([
                html.I(className="fas fa-thermometer-half fa-2x", style={'color': COLORS['temp_in']}),
                html.Div([
                    html.H4("0.0", id="avg-temp-input", className="card-value"),
                    html.P("Средняя температура входа ГВС, °C", className="card-label")
                ], style={'margin-left': '15px'})
            ], className="d-flex align-items-center")
        ])
    ], style={'border': 'none', 'box-shadow': '0 4px 6px rgba(0,0,0,0.1)', 'border-radius': '10px'})),

    dbc.Col(dbc.Card([
        dbc.CardBody([
            html.Div([
                html.I(className="fas fa-thermometer-half fa-2x", style={'color': COLORS['temp_in']}),
                html.Div([
                    html.H4("0.0", id="avg-temp-output", className="card-value"),
                    html.P("Средняя температура выхода ГВС, °C", className="card-label")
                ], style={'margin-left': '15px'})
            ], className="d-flex align-items-center")
        ])
    ], style={'border': 'none', 'box-shadow': '0 4px 6px rgba(0,0,0,0.1)', 'border-radius': '10px'})),
], className="mb-4")


# Заголовок страницы
header = dbc.Card(
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.H2("📊 Аналитика системы водоснабжения", 
                       style={'color': 'white', 'margin': '0', 'fontWeight': '600'}),
                html.P("Мониторинг потребления и температурных режимов", 
                      style={'color': 'rgba(255,255,255,0.8)', 'margin': '0', 'fontSize': '14px'})
            ]),
        ])
    ]),
    style=CARD_STYLE,
    className="mb-4"
)

layout = dbc.Container(
    [
        # Заголовок
        header,
        
        # Фильтры
        dbc.Row([
            dbc.Col(id_selection, width=6),
            dbc.Col(period_selection, width=6),
        ], className="mb-4"),
        
        # Статистические карточки
        stats_cards,
        
        # График
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(id='graph', style={"height": 750}),
                    ])
                ], style={
                    'border': 'none',
                    'box-shadow': '0 4px 6px rgba(0,0,0,0.1)',
                    'border-radius': '10px'
                })
            ])
        ], className="mb-4"),
        
        # Таблицы
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("🔥 Горячее водоснабжение", 
                               style={'margin': '0', 'color': COLORS['dark'], 
                                      'display': 'flex', 'align-items': 'center'}),
                        html.I(className="fas fa-fire", style={'color': COLORS['hot_in'], 'margin-left': '10px'})
                    ], style={'background': 'none', 'border-bottom': f'2px solid {COLORS["hot_in"]}'}),
                    dbc.CardBody([
                        dash_table.DataTable(
                            id='hot-table',
                            columns=[
                                {"name": "Дата", "id": "datetime", "type": "datetime"},
                                {"name": "Время", "id": "hour"},
                                {"name": "Подача, м³", "id": 'input_hot', "type": "numeric", "format": {"specifier": ".2f"}},
                                {"name": "Обратка, м³", "id": 'output_hot', "type": "numeric", "format": {"specifier": ".2f"}},
                                {"name": "Потребление, м³", "id": 'rashod_hot', "type": "numeric", "format": {"specifier": ".2f"}},
                                {"name": "Т1 гвс, °C", "id": "temp_input"},
                                {"name": "Т2 гвс, °C", "id": "temp_output"},
                            ],
                            data=[],
                            style_header={
                                'fontWeight': 'bold',
                                'textAlign': 'center',
                                'backgroundColor': COLORS['light'],
                                'color': COLORS['dark'],
                                'border': 'none'
                            },
                            style_cell={
                                'textAlign': 'center',
                                'padding': '10px',
                                'border': 'none'
                            },
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': 'rgba(248, 249, 250, 0.8)',
                                }
                            ],
                            style_table={
                                'height': '600px',
                                'overflowX': 'auto',
                                'borderRadius': '8px'
                            },
                            export_format='xlsx',
                            export_headers='display',
                        )
                    ])
                ], style={
                    'border': 'none',
                    'box-shadow': '0 4px 6px rgba(0,0,0,0.1)',
                    'border-radius': '10px'
                })
            ], width=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("❄️ Холодное водоснабжение", 
                               style={'margin': '0', 'color': COLORS['dark'],
                                      'display': 'flex', 'align-items': 'center'}),
                        html.I(className="fas fa-snowflake", style={'color': COLORS['cold'], 'margin-left': '10px'})
                    ], style={'background': 'none', 'border-bottom': f'2px solid {COLORS["cold"]}'}),
                    dbc.CardBody([
                        dash_table.DataTable(
                            id='cold-table',
                            columns=[
                                {"name": "Дата", "id": "datetime"},
                                {"name": "Время", "id": "hour"},
                                {"name": "Накопленное, м³", "id": "cumulative_rashod_cold", "type": "numeric", "format": {"specifier": ".3f"}},
                                {"name": "Потребление, м³", "id": "rashod_cold", "type": "numeric", "format": {"specifier": ".2f"}}
                            ],
                            data=[],  
                            style_header={
                                'fontWeight': 'bold',
                                'textAlign': 'center',
                                'backgroundColor': COLORS['light'],
                                'color': COLORS['dark'],
                                'border': 'none'
                            },
                            style_cell={
                                'textAlign': 'center',
                                'padding': '10px',
                                'border': 'none'
                            },
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': 'rgba(248, 249, 250, 0.8)',
                                }
                            ],
                            style_table={
                                'height': '600px',
                                'overflowY': 'auto',
                                'borderRadius': '8px'
                            },
                            export_format='xlsx',
                            export_headers='display',
                        )
                    ])
                ], style={
                    'border': 'none',
                    'box-shadow': '0 4px 6px rgba(0,0,0,0.1)',
                    'border-radius': '10px'
                })
            ], width=6),
        ]),
        
        # Интервал обновления
        dcc.Interval(
            id='interval-component',
            interval=30*60*1000,
            n_intervals=0
        )
    ],
    fluid=True,
    style={
        'background': 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
        'min-height': '100vh',
        'padding': '20px'
    }
)

@callback(
    Output('graph', 'figure'),
    Output('hot-table', 'data'),
    Output('cold-table', 'data'),
    Output('avg-hot-consumption', 'children'),
    Output('avg-cold-consumption', 'children'),
    Output('avg-leak', 'children'),
    Output('avg-temp-input', 'children'),
    Output('avg-temp-output', 'children'),
    Input('id-dropdown', 'value'),
    Input('period-radio', 'value'),
    Input('interval-component', 'n_intervals')
)
def update_graph(id, period, n):
    client = clickhouse_connect.get_client(**CLICK_CONN)
    df = client.query_df(f'SELECT * FROM msk_database.analytic WHERE id={id}')
    df['dif'] = df['rashod_cold'] - df['rashod_hot'] - 0.03

    df = df.sort_values(['datetime'])
    df = df.tail(period * 30 * 24)

    # Расчет статистики для карточек
    avg_hot = df['rashod_hot'].mean()
    avg_cold = df['rashod_cold'].mean()
    avg_leak = df['dif'].mean()
    # avg_temp = (df['temp_input'].mean() + df['temp_output'].mean()) / 2
    avg_temp_input = df['temp_input'].mean()
    avg_temp_output = df['temp_output'].mean()

    # Создаем subplots с улучшенным дизайном
    fig = make_subplots(
        rows=3, 
        cols=1,
        subplot_titles=(
            f'<b>Расход воды в МКД № {id}</b>',
            f'<b>Баланс горячего водоснабжения</b>', 
            f'<b>Температурный режим ГВС</b>'
        ),
        vertical_spacing=0.08,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]]
    )

    # Первый график - расходы с заливкой для утечки
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['rashod_hot'], 
                  name='Расход горячей', 
                  line=dict(width=3, color=COLORS['hot_in']),
                  fill='tozeroy',
                  fillcolor=f'rgba{tuple(int(COLORS["hot_in"][i:i+2], 16) for i in (1, 3, 5)) + (0.2,)}',
                  legendgroup="group1"),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['rashod_cold'], 
                  name='Расход холодной', 
                  line=dict(width=3, color=COLORS['cold']),
                  fill='tozeroy',
                  fillcolor=f'rgba{tuple(int(COLORS["cold"][i:i+2], 16) for i in (1, 3, 5)) + (0.2,)}',
                  legendgroup="group1"),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['dif'], 
                  name='Утечка', 
                  line=dict(width=2, color=COLORS['leak'], dash='dot'),
                  legendgroup="group1"),
        row=1, col=1
    )
    
    # Второй график - подача и выход горячей воды
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['input_hot'], 
                  name='Подача горячей', 
                  line=dict(width=3, color=COLORS['hot_in']),
                  fill='tozeroy',
                  fillcolor=f'rgba{tuple(int(COLORS["hot_in"][i:i+2], 16) for i in (1, 3, 5)) + (0.3,)}',
                  legendgroup="group2"),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['output_hot'], 
                  name='Обратка горячей', 
                  line=dict(width=3, color=COLORS['hot_out']),
                  fill='tozeroy',
                  fillcolor=f'rgba{tuple(int(COLORS["hot_out"][i:i+2], 16) for i in (1, 3, 5)) + (0.3,)}',
                  legendgroup="group2"),
        row=2, col=1
    )
    
    # Третий график - температуры
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['temp_input'], 
                  name='Температура подачи', 
                  line=dict(width=3, color=COLORS['temp_in']),
                  fill='tozeroy',
                  fillcolor=f'rgba{tuple(int(COLORS["temp_in"][i:i+2], 16) for i in (1, 3, 5)) + (0.2,)}',
                  legendgroup="group3"),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['temp_output'], 
                  name='Температура обратки', 
                  line=dict(width=3, color=COLORS['temp_out']),
                  fill='tozeroy',
                  fillcolor=f'rgba{tuple(int(COLORS["temp_out"][i:i+2], 16) for i in (1, 3, 5)) + (0.2,)}',
                  legendgroup="group3"),
        row=3, col=1
    )
    
    # Обновляем layout с современным дизайном
    fig.update_layout(
        height=800,
        template="plotly_white",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif", size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1,
            font=dict(size=11)
        ),
        margin=dict(l=50, r=50, t=80, b=50),
        hovermode='x unified'
    )
    
    # Обновляем оси с улучшенным дизайном
    for i in range(1, 4):
        fig.update_xaxes(
            title_text="Дата" if i == 3 else "",
            row=i, col=1,
            gridcolor='rgba(0,0,0,0.1)',
            showline=True,
            linewidth=1,
            linecolor='rgba(0,0,0,0.2)'
        )
        
    fig.update_yaxes(
        title_text="м³/ч", 
        row=1, col=1,
        gridcolor='rgba(0,0,0,0.1)'
    )
    fig.update_yaxes(
        title_text="м³/ч", 
        row=2, col=1,
        gridcolor='rgba(0,0,0,0.1)'
    )
    fig.update_yaxes(
        title_text="°C", 
        row=3, col=1,
        gridcolor='rgba(0,0,0,0.1)'
    )
    
    # Обновляем заголовки подграфиков
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=14, color=COLORS['dark'])
    
    # Подготовка данных для таблиц
    df = df[['datetime', 'input_hot', 'output_hot', 'rashod_hot', 'temp_input', 'temp_output', 'cumulative_rashod_cold', 'rashod_cold']]
    df['hour_1'] = df['datetime'].dt.hour
    df['hour_2'] = df['hour_1'] + 1
    df['hour'] = (df['hour_1']).astype(str) + '-' + (df['hour_2']).astype(str)
    df['datetime'] = df['datetime'].dt.date

    df_cold = df[['datetime','hour', 'cumulative_rashod_cold', 'rashod_cold']]
    df_hot = df[['datetime', 'hour', 'input_hot', 'output_hot', 'rashod_hot', 'temp_input', 'temp_output']]
    df_cold = df_cold.to_dict('records')
    df_hot = df_hot.to_dict('records')
  
    return (
        fig, 
        df_hot, 
        df_cold,
        f"{avg_hot:.2f}",
        f"{avg_cold:.2f}", 
        f"{avg_leak:.2f}",
        f"{avg_temp_input:.1f}",
        f"{avg_temp_output:.1f}"
    )
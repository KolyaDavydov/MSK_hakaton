import dash
from dash import dcc, html, Dash, register_page, dash_table, callback
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from tqdm import tqdm
import clickhouse_connect
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
from datetime import timedelta
import sys
sys.path.append("../pages")

from config import CLICK_CONN

register_page(__name__, path="/predictive/", name='Прогнозирование')

# Стили для компонентов
CARD_STYLE = {
    "border": "none",
    "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.1)",
    "borderRadius": "12px",
    "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
}

CONTENT_CARD_STYLE = {
    "border": "none",
    "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.05)",
    "borderRadius": "10px",
    "background": "white"
}

METRIC_CARD_STYLE = {
    "border": "none",
    "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.08)",
    "borderRadius": "8px",
    "background": "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
    "color": "white"
}

def prepare_data_for_predict():
    # Создание клиента
    client = clickhouse_connect.get_client(**CLICK_CONN)

    df = client.query_df(
        f"""
        SELECT *
        FROM msk_database.analytic
        ORDER BY id, datetime
        """
        )

    df['dif'] = df['rashod_cold'] - df['rashod_hot']
    df['percent_dif'] =  df['dif'] / df['rashod_cold'] * 100 #Будет нашим таргетом
    del df['cumulative_rashod_cold']

    for i in tqdm(range(1, 30)):
        df[f'percent_lag{i}'] = df.groupby('id')['percent_dif'].shift(i)
        df[f'dif_lag{i}'] = df.groupby('id')['dif'].shift(i)
        df[f'temp_output_lag{i}'] = df.groupby('id')['temp_output'].shift(i)
    
    return df

def load_model():
    print('Загрузка моделей...')
    model_4 = CatBoostRegressor()
    model_4.load_model('models/model_4h.cbm')

    model_24 = CatBoostRegressor()
    model_24.load_model('models/model_24h.cbm')

    model_72 = CatBoostRegressor()
    model_72.load_model('models/model_72h.cbm')

    return model_4, model_24, model_72

def get_predict(df, model_4, model_24, model_72):
    df_clear = df.dropna()

    pred_4 = model_4.predict(df_clear.drop(columns=['datetime']))
    pred_24 = model_24.predict(df_clear.drop(columns=['datetime']))
    pred_72 = model_72.predict(df_clear.drop(columns=['datetime']))

    df_clear = df_clear[['id', 'datetime', 'dif', 'percent_dif', 'rashod_hot', 'rashod_cold']]
    df_clear['pred_4h'] = pred_4
    df_clear['pred_24h'] = pred_24
    df_clear['pred_72h'] = pred_72

    # Находим максимальную дату для каждого id
    max_dates = df_clear.groupby('id')['datetime'].max().reset_index()
    max_dates.rename(columns={'datetime': 'max_datetime'}, inplace=True)
    
    # Создаем список для новых строк
    new_rows = []
    
    # Для каждого уникального id
    for id_val in df_clear['id'].unique():
        # Находим максимальную дату для этого id
        max_date = max_dates[max_dates['id'] == id_val]['max_datetime'].iloc[0]
        
        # Берем последнюю строку для этого id как шаблон
        last_row = df_clear[df_clear['id'] == id_val].iloc[-1].copy()
        
        # Создаем 72 новых строк
        for i in range(1, 73):
            new_row = last_row.copy()
            new_row['datetime'] = max_date + timedelta(hours=i)
            # Обнуляем или оставляем пустыми некоторые поля (по желанию)
            new_row['dif'] = np.nan
            new_row['percent_dif'] = np.nan
            new_row['pred_4h'] = np.nan
            new_row['pred_24h'] = np.nan
            new_row['pred_72h'] = np.nan
            new_row['rashod_hot'] = np.nan
            new_row['rashod_cold'] = np.nan
            
            new_rows.append(new_row)
    
    # Создаем DataFrame из новых строк
    new_df = pd.DataFrame(new_rows)
    
    # Объединяем с исходным DataFrame
    result_df = pd.concat([df_clear, new_df], ignore_index=True)
    
    # Сортируем по id и datetime
    result_df = result_df.sort_values(['id', 'datetime']).reset_index(drop=True)

    result_df['pred_4h'] = result_df.groupby('id')['pred_4h'].shift(4)
    result_df['pred_24h'] = result_df.groupby('id')['pred_24h'].shift(24)
    result_df['pred_72h'] = result_df.groupby('id')['pred_72h'].shift(72)

    return result_df

# Создание клиента к БД
client = clickhouse_connect.get_client(**CLICK_CONN)

result = client.query('SELECT DISTINCT id FROM msk_database.analytic')
# Получим уникальные значения id в виде списка Python
unique_ids = [row[0] for row in result.result_rows]

# Заголовок страницы
header = dbc.Card(
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.H2("🎯 Прогнозирование инцидентов", 
                       style={'color': 'white', 'margin': '0', 'fontWeight': '600'}),
                html.P("Система прогнозирования аномалий в системах водоснабжения", 
                      style={'color': 'rgba(255,255,255,0.8)', 'margin': '0', 'fontSize': '14px'})
            ]),
            dbc.Col([
                html.Div([
                    html.I(className="fas fa-sync-alt", style={'marginRight': '8px'}),
                    "Обновляется каждые 30 минут"
                ], style={'color': 'white', 'textAlign': 'right', 'fontSize': '14px'})
            ], width="auto")
        ])
    ]),
    style=CARD_STYLE,
    className="mb-4"
)

# Карточка выбора параметров
selection_card = dbc.Card(
    dbc.CardBody([
        html.H5("📊 Параметры анализа", className="card-title", 
               style={'color': '#2c3e50', 'marginBottom': '20px'}),
        
        dbc.Row([
            dbc.Col([
                html.Label("Выбор дома", className="form-label", 
                          style={'fontWeight': '600', 'color': '#34495e', 'marginBottom': '8px'}),
                dcc.Dropdown(
                    id="id-dropdown-predict",
                    options=[{"label": f"МКД № {id}", "value": id} for id in unique_ids],
                    value=unique_ids[0],
                    className="dropdown-custom"
                )
            ]),
        ]),
        dbc.Row([
            dbc.Col([
                html.Label("Период отображения", className="form-label", 
                          style={'fontWeight': '600', 'color': '#34495e', 'marginBottom': '8px', 'marginTop': '15px'}),
                dcc.RadioItems(
                    id='period-radio-predict',
                    options=[
                        {'label': html.Span([' 1 месяц'], style={'fontWeight': '500'}), 'value': 1},
                        {'label': html.Span([' 3 месяца'], style={'fontWeight': '500'}), 'value': 3},
                        {'label': html.Span([' 1 год'], style={'fontWeight': '500'}), 'value': 12}
                    ],
                    value=3,
                    inline=False,
                    # labelStyle={'display': 'block'},
                    className="radioitems-custom"
                )
            ]),
        ])
    ]),
    style=CONTENT_CARD_STYLE,
    className="mb-4"
)

# Карточки для приоритетных прогнозов (отдельные друг под другом)
attention_4h = dbc.Card(
    dbc.CardBody([
        html.H6("⏰ Прогноз на 4 часа", 
               style={'textAlign': 'center', 'marginBottom': '15px', 'color': '#3498db', 'fontWeight': '600'}),
        dash_table.DataTable(
            id='pred-4h-table',
            columns=[
                {"name": "МКД №", "id": "id", "type": "numeric"},
                {"name": "Дата", "id": "datetime", "type": "datetime"},
                {"name": "Время", "id": "hour"},
                {"name": "Вероятность", "id": 'pred_4h', "type": "numeric", "format": {"specifier": ".1f"}},
            ],
            data=[],
            style_header={
                'fontWeight': '600',
                'textAlign': 'center',
                'backgroundColor': '#3498db',
                'color': 'white',
                'fontSize': '11px',
                'padding': '6px',
                'border': 'none'
            },
            style_data_conditional=[
                {
                    'if': {
                        'filter_query': '{pred_4h} <= 20.0',
                    },
                    'backgroundColor': '#d4edda',
                    'color': '#155724',
                },
                {
                    'if': {
                        'filter_query': '{pred_4h} > 20.0 && {pred_4h} <= 40.0',
                    },
                    'backgroundColor': '#fff3cd',
                    'color': '#856404',
                },
                {
                    'if': {
                        'filter_query': '{pred_4h} > 40.0',
                    },
                    'backgroundColor': '#f8d7da',
                    'color': '#721c24',
                },
            ],
            style_cell={
                'fontSize': '10px',
                'padding': '4px 6px',
                'textAlign': 'center',
                'backgroundColor': '#f8f9fa',
                'color': '#212529',
                'border': 'none'
            },
            style_table={'height': '180px', 'overflowY': 'auto'},
        )
    ]),
    style=CONTENT_CARD_STYLE,
    className="mb-3"
)

attention_24h = dbc.Card(
    dbc.CardBody([
        html.H6("📅 Прогноз на 24 часа", 
               style={'textAlign': 'center', 'marginBottom': '15px', 'color': '#e67e22', 'fontWeight': '600'}),
        dash_table.DataTable(
            id='pred-24h-table',
            columns=[
                {"name": "МКД №", "id": "id", "type": "numeric"},
                {"name": "Дата", "id": "datetime", "type": "datetime"},
                {"name": "Время", "id": "hour"},
                {"name": "Вероятность", "id": 'pred_24h', "type": "numeric", "format": {"specifier": ".1f"}},
            ],
            data=[],
            style_header={
                'fontWeight': '600',
                'textAlign': 'center',
                'backgroundColor': '#e67e22',
                'color': 'white',
                'fontSize': '11px',
                'padding': '6px',
                'border': 'none'
            },
            style_data_conditional=[
                {
                    'if': {
                        'filter_query': '{pred_24h} <= 20.0',
                    },
                    'backgroundColor': '#d4edda',
                    'color': '#155724',
                },
                {
                    'if': {
                        'filter_query': '{pred_24h} > 20.0 && {pred_24h} <= 40.0',
                    },
                    'backgroundColor': '#fff3cd',
                    'color': '#856404',
                },
                {
                    'if': {
                        'filter_query': '{pred_24h} > 40.0',
                    },
                    'backgroundColor': '#f8d7da',
                    'color': '#721c24',
                },
            ],
            style_cell={
                'fontSize': '10px',
                'padding': '4px 6px',
                'textAlign': 'center',
                'backgroundColor': '#f8f9fa',
                'color': '#212529',
                'border': 'none'
            },
            style_table={'height': '180px', 'overflowY': 'auto'},
        )
    ]),
    style=CONTENT_CARD_STYLE,
    className="mb-3"
)

attention_72h = dbc.Card(
    dbc.CardBody([
        html.H6("📊 Прогноз на 3 дня", 
               style={'textAlign': 'center', 'marginBottom': '15px', 'color': '#e74c3c', 'fontWeight': '600'}),
        dash_table.DataTable(
            id='pred-72h-table',
            columns=[
                {"name": "МКД №", "id": "id", "type": "numeric"},
                {"name": "Дата", "id": "datetime", "type": "datetime"},
                {"name": "Время", "id": "hour"},
                {"name": "Вероятность", "id": 'pred_72h', "type": "numeric", "format": {"specifier": ".1f"}},
            ],
            data=[],
            style_header={
                'fontWeight': '600',
                'textAlign': 'center',
                'backgroundColor': '#e74c3c',
                'color': 'white',
                'fontSize': '11px',
                'padding': '6px',
                'border': 'none'
            },
            style_data_conditional=[
                {
                    'if': {
                        'filter_query': '{pred_72h} <= 20.0',
                    },
                    'backgroundColor': '#d4edda',
                    'color': '#155724',
                },
                {
                    'if': {
                        'filter_query': '{pred_72h} > 20.0 && {pred_72h} <= 40.0',
                    },
                    'backgroundColor': '#fff3cd',
                    'color': '#856404',
                },
                {
                    'if': {
                        'filter_query': '{pred_72h} > 40.0',
                    },
                    'backgroundColor': '#f8d7da',
                    'color': '#721c24',
                },
            ],
            style_cell={
                'fontSize': '10px',
                'padding': '4px 6px',
                'textAlign': 'center',
                'backgroundColor': '#f8f9fa',
                'color': '#212529',
                'border': 'none'
            },
            style_table={'height': '180px', 'overflowY': 'auto'},
        )
    ]),
    style=CONTENT_CARD_STYLE,
    className="mb-3"
)

# Общие метрики точности моделей
metrics_all = dbc.Card(
    dbc.CardBody([
        html.H5("📈 Точность моделей", 
               className="card-title", style={'color': '#2c3e50', 'marginBottom': '15px'}),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-chart-line", style={'fontSize': '20px', 'marginBottom': '8px'}),
                            html.H6("4 часа", className="card-subtitle", 
                                   style={'fontSize': '0.8rem', 'color': 'rgba(255,255,255,0.9)'}),
                            html.H3(id="mae-4h", children="...", 
                                   className="card-text", style={'margin': '0', 'fontWeight': '600'})
                        ], style={'textAlign': 'center'})
                    ])
                ], style=METRIC_CARD_STYLE)
            ], width=12, className="mb-3"),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-chart-bar", style={'fontSize': '20px', 'marginBottom': '8px'}),
                            html.H6("24 часа", className="card-subtitle", 
                                   style={'fontSize': '0.8rem', 'color': 'rgba(255,255,255,0.9)'}),
                            html.H3(id="mae-24h", children="...", 
                                   className="card-text", style={'margin': '0', 'fontWeight': '600'})
                        ], style={'textAlign': 'center'})
                    ])
                ], style=METRIC_CARD_STYLE)
            ], width=12, className="mb-3"),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-chart-area", style={'fontSize': '20px', 'marginBottom': '8px'}),
                            html.H6("72 часа", className="card-subtitle", 
                                   style={'fontSize': '0.8rem', 'color': 'rgba(255,255,255,0.9)'}),
                            html.H3(id="mae-72h", children="...", 
                                   className="card-text", style={'margin': '0', 'fontWeight': '600'})
                        ], style={'textAlign': 'center'})
                    ])
                ], style=METRIC_CARD_STYLE)
            ], width=12)
        ])
    ]),
    style=CONTENT_CARD_STYLE,
    className="mb-4"
)

# График
graph_card = dbc.Card(
    dbc.CardBody([
        html.H5("📈 Аналитика прогнозов", 
               className="card-title", style={'color': '#2c3e50', 'marginBottom': '20px'}),
        dcc.Graph(id='graph-predict', style={"height": 650}),
        dcc.Interval(
            id='interval-component-predict',
            interval=30*60*1000,
            n_intervals=0
        )
    ]),
    style=CONTENT_CARD_STYLE
)

layout = dbc.Container(
    [
        header,
        dbc.Row([
            # Левая колонка - выбор параметров и точность для текущего дома
            dbc.Col([
                selection_card,
                metrics_all, #metrics_current,
            ], width=2),
            # Центральная колонка - график и общие метрики
            dbc.Col([
                graph_card,
                # metrics_all
            ], width=7),
            # Правая колонка - приоритетные прогнозы
            dbc.Col([
                attention_4h,
                attention_24h,
                attention_72h
            ], width=3)
        ])
    ],
    fluid=True,
    style={'backgroundColor': '#f8f9fa', 'minHeight': '100vh', 'padding': '20px'}
)

# CSS стили
custom_css = """
.dropdown-custom {
    border-radius: 8px;
}

.radioitems-custom .form-check {
    margin-right: 15px;
}

.radioitems-custom .form-check-input:checked {
    background-color: #667eea;
    border-color: #667eea;
}

.card {
    transition: transform 0.2s ease-in-out;
}

.card:hover {
    transform: translateY(-2px);
}
"""

# Добавляем CSS стили
app = dash.get_app()
app.index_string = app.index_string.replace('</head>', f'<style>{custom_css}</style></head>')

@callback(
    Output('mae-4h', 'children'),
    Output('mae-24h', 'children'),
    Output('mae-72h', 'children'),
    Output('pred-4h-table', 'data'),
    Output('pred-24h-table', 'data'),
    Output('pred-72h-table', 'data'),
    Input('interval-component-predict', 'n_intervals'))
def save_predict_to_db(n):
    df = prepare_data_for_predict()
    model_4, model_24, model_72 = load_model()
    df = get_predict(df, model_4, model_24, model_72)
    
    client = clickhouse_connect.get_client(**CLICK_CONN)
    client.command('DROP TABLE IF EXISTS msk_database.prediction')

    create_table_query = '''
    CREATE TABLE IF NOT EXISTS msk_database.prediction (
        id Int32,
        datetime DateTime,
        dif Float32,
        percent_dif Float32,
        rashod_hot Float32,
        rashod_cold Float32,
        pred_4h Float32,
        pred_24h Float32,
        pred_72h Float32,
    ) ENGINE = MergeTree()
    ORDER BY (datetime)
    '''
    client.command(create_table_query)
    client.insert_df('msk_database.prediction', df)

    mae = df.dropna()
    mae_4h = mean_absolute_error(mae['percent_dif'], mae['pred_4h'])
    mae_4h = f"{100 - mae_4h:.1f}%"

    mae_24h = mean_absolute_error(mae['percent_dif'], mae['pred_24h'])
    mae_24h = f"{100 - mae_24h:.1f}%"

    mae_72h = mean_absolute_error(mae['percent_dif'], mae['pred_72h'])
    mae_72h = f"{100 - mae_72h:.1f}%"

    df_4 = client.query_df(
        f"""
            SELECT id, datetime, pred_4h from msk_database.prediction
            WHERE NOT isNaN(pred_4h)
            ORDER BY datetime DESC, pred_4h DESC
            LIMIT 1 BY id
        """
    )
    df_4['hour_1'] = df_4['datetime'].dt.hour
    df_4['hour_2'] = df_4['hour_1'] + 1
    df_4['hour'] = (df_4['hour_1']).astype(str) + '-' + (df_4['hour_2']).astype(str)
    df_4['datetime'] = df_4['datetime'].dt.date
    del df_4['hour_1']
    del df_4['hour_2']

    df_24 = client.query_df(
        f"""
            SELECT id, datetime, pred_24h from msk_database.prediction
            WHERE NOT isNaN(pred_24h)
            ORDER BY datetime DESC, pred_24h DESC
            LIMIT 1 BY id
        """
    )
    df_24['hour_1'] = df_24['datetime'].dt.hour
    df_24['hour_2'] = df_24['hour_1'] + 1
    df_24['hour'] = (df_24['hour_1']).astype(str) + '-' + (df_24['hour_2']).astype(str)
    df_24['datetime'] = df_24['datetime'].dt.date
    del df_24['hour_1']
    del df_24['hour_2']

    df_72 = client.query_df(
        f"""
            SELECT id, datetime, pred_72h from msk_database.prediction
            WHERE NOT isNaN(pred_72h)
            ORDER BY datetime DESC, pred_72h DESC
            LIMIT 1 BY id
        """
    )
    df_72['hour_1'] = df_72['datetime'].dt.hour
    df_72['hour_2'] = df_72['hour_1'] + 1
    df_72['hour'] = (df_72['hour_1']).astype(str) + '-' + (df_72['hour_2']).astype(str)
    df_72['datetime'] = df_72['datetime'].dt.date
    del df_72['hour_1']
    del df_72['hour_2']

    return mae_4h, mae_24h, mae_72h, df_4.to_dict('records'), df_24.to_dict('records'), df_72.to_dict('records')


@callback(
    Output('graph-predict', 'figure'),
    Input('id-dropdown-predict', 'value'),
    Input('period-radio-predict', 'value'),
)
def update_graph(id, period):
    client = clickhouse_connect.get_client(**CLICK_CONN)
    try:
        df = client.query_df(f'SELECT * FROM msk_database.prediction WHERE id={id}')
    except:
        df = pd.DataFrame()
    
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Данные еще не загружены, подождите...",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig, "N/A", "N/A", "N/A"

    df = df.sort_values(['datetime'])
    df = df.tail(period * 30 * 24)

    # Создаем subplots с 3 рядами
    fig = make_subplots(
        rows=2, 
        cols=1,
        subplot_titles=(
            f'Вероятность инцидентов для МКД № {id}',
            f'Динамика расходов воды в МКД № {id}', 
        ),
        vertical_spacing=0.08
    )
    
    # ДОБАВЛЯЕМ ОБЛАСТЬ ВЫШЕ УРОВНЯ ОПАСНОСТИ
    fig.add_trace(
        go.Scatter(
            x=df['datetime'],
            y=[100] * len(df),
            fill=None,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['datetime'],
            y=[40] * len(df),
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.05)',
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ),
        row=1, col=1
    )

    # Прогнозы
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=np.floor(df['pred_4h'] / 5) * 5, 
                  name='прогноз на 4 часа', line=dict(width=2, color='#3498db'),
                  legendgroup="pred_4h", showlegend=True),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=np.floor(df['pred_24h'] / 5) * 5, 
                  name='прогноз на сутки', line=dict(width=2, color='#e67e22'),
                  legendgroup="pred_24h", showlegend=True),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=df['datetime'], y=np.floor(df['pred_72h'] / 5) * 5, 
                  name='прогноз на 3 дня', line=dict(width=2, color='#e74c3c'),
                  legendgroup="pred_72h", showlegend=True),
        row=1, col=1
    )

    # ЛИНИЯ УРОВНЯ ОПАСНОСТИ
    fig.add_trace(
        go.Scatter(
            x=df['datetime'],
            y=[40] * len(df),
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='Уровень опасности',
            showlegend=True
        ),
        row=1, col=1
    )

    # Второй график - подача и выход горячей воды
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['rashod_hot'], 
                  name='Расход горячей, м³/ч', line=dict(width=2, color='#9b59b6'),
                  legendgroup="rashod_hot", showlegend=True),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['rashod_cold'], 
                  name='Расход холодной, м³/ч', line=dict(width=2, color='#2ecc71'),
                  legendgroup="rashod_cold", showlegend=True),
        row=2, col=1
    )

    fig.update_yaxes(range=[0, 100], row=1, col=1)

    # Обновляем layout
    fig.update_layout(
        height=650,
        showlegend=True,
        template="plotly_white",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#2c3e50"),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1
        )
    )

    # Обновляем оси
    fig.update_xaxes(title_text="Дата", row=2, col=1, gridcolor='rgba(0,0,0,0.1)')
    fig.update_yaxes(title_text="Вероятность инцидента, %", row=1, col=1, gridcolor='rgba(0,0,0,0.1)')
    fig.update_yaxes(title_text="Расход воды, м³/ч", row=2, col=1, gridcolor='rgba(0,0,0,0.1)')

    # Добавляем сетку
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    mae = df.dropna()
    mae_4h = mean_absolute_error(mae['percent_dif'], mae['pred_4h'])
    mae_4h = f"{100 - mae_4h:.1f}%"

    mae_24h = mean_absolute_error(mae['percent_dif'], mae['pred_24h'])
    mae_24h = f"{100 - mae_24h:.1f}%"

    mae_72h = mean_absolute_error(mae['percent_dif'], mae['pred_72h'])
    mae_72h = f"{100 - mae_72h:.1f}%"
  
    return fig

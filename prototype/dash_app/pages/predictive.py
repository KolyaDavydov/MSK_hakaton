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


    # # Сдвигаем таргет для трех моделей и добавляем лаговые признаки
    # df['target_4h'] = df.groupby('id')['percent_dif'].shift(-4)
    # df['target_24h'] = df.groupby('id')['percent_dif'].shift(-24)
    # df['target_72h'] = df.groupby('id')['percent_dif'].shift(-72)
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

df = prepare_data_for_predict()

model_4, model_24, model_72 = load_model()

df = get_predict(df, model_4, model_24, model_72)


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




result = client.query('SELECT DISTINCT id FROM msk_database.analytic')
# Получим уникальные значения id в виде списка Python
unique_ids = [row[0] for row in result.result_rows]

id_selection_predict = dbc.Card(
    [
        dcc.Dropdown(
            id="id-dropdown-predict",
            options=unique_ids,
            value=unique_ids[0],
            style={"margin-top": -10}
        ),
        html.Div('Выбор дома', style={'font-weight': 'bold', "height": 10})
    ],
    body=True,
    color="secondary",
    outline=True
)

period_selection_predict = dbc.Card(
    [
        dcc.RadioItems(
            id='period-radio-predict',
            options=[
                {'label': ' 1 месяц', 'value': 1},
                {'label': ' 3 месяца', 'value': 3},
                {'label': ' 12 месяцев', 'value': 12}
            ],
            value=1,
            inline=True,
            labelStyle={'margin-right': '30px'},  # горизонтальные отступы
            style={'margin': '0px 0'}  # вертикальные отступы
        ),
        html.Div('Период отображение', style={'font-weight': 'bold', "height": 10})
    ],
    body=True,
    color="secondary",
    outline=True
)

# Создаем карточки для метрик
metric_predict_all = dbc.Card(
    [
        dbc.Row([  # Добавляем dbc.Row для размещения колонок в одной строке
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("4ч", className="card-title", style={"font-size": "0.9rem", "text-align": "right"}),
                        html.H4(id="mae-4h", children="расчет...", className="card-text"),
                    ])
                ], color="light", outline=True)
            ], width=4),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("24ч", className="card-title", style={"font-size": "0.9rem", "text-align": "right"}),
                        html.H4(id="mae-24h", children="расчет...", className="card-text"),
                    ])
                ], color="light", outline=True)
            ], width=4),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("72ч", className="card-title", style={"font-size": "0.9rem", "text-align": "right"}),
                        html.H4(id="mae-72h", children="расчет...", className="card-text"),
                    ])
                ], color="light", outline=True)
            ], width=4)
        ], className="g-2"),  # Отступы между колонками
        html.Div('Общая средня ошибка', style={'font-weight': 'bold', "height": 10, 'text-align': 'center', 'color': '#2c3e50', 'margin-top':10})
    ],
    body=True,
    color="secondary",
    outline=True
)

# Создаем карточки для метрик
metric_predict_current = dbc.Card(
    [
        dbc.Row([  # Добавляем dbc.Row для размещения колонок в одной строке
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("4ч", className="card-title", style={"font-size": "0.9rem", "text-align": "right"}),
                        html.H4(id="mae-4h-curr", children="расчет...", className="card-text"),
                    ])
                ], color="light", outline=True)
            ], width=4),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("24ч", className="card-title", style={"font-size": "0.9rem", "text-align": "right"}),
                        html.H4(id="mae-24h-curr", children="расчет...", className="card-text"),
                    ])
                ], color="light", outline=True)
            ], width=4),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("72ч", className="card-title", style={"font-size": "0.9rem", "text-align": "right"}),
                        html.H4(id="mae-72h-curr", children="расчет...", className="card-text"),
                    ])
                ], color="light", outline=True)
            ], width=4)
        ], className="g-2"),  # Отступы между колонками
        html.Div('Текущая средня ошибка', style={'font-weight': 'bold', "height": 10, 'text-align': 'center', 'color': '#2c3e50', 'margin-top':10})
    ],
    body=True,
    color="secondary",
    outline=True
)


layout = dbc.Container(
    [
        dbc.Row([
            dbc.Col([
                dbc.Row(
                    id_selection_predict,
                    style={
                        # "height": 40,
                        "margin-left": 20,
                        "margin-right": 20}
                ),
                dbc.Row(
                    period_selection_predict,
                    style={"margin-left": 20, "margin-top": 5, "height": 80, "margin-right": 20,}),

            ]),
            dbc.Col([
                dbc.Row(
                    metric_predict_current,
                    style={"height": 80, "margin-right": 20,}),
            ]),
            dbc.Col([
                dbc.Row(
                    metric_predict_all,
                    style={"height": 80, "margin-right": 20,}),
            ]),
        ]),
        dbc.Row([

            dcc.Graph(id='graph-predict', style={"height": 750}),
                # Добавьте этот компонент для автоматического обновления
            dcc.Interval(
                id='interval-component-predict',
                interval=30*60*1000,  # 30 минут в миллисекундах
                n_intervals=0
            )
        ]),

    ],
    fluid=True,
)

@callback(
    # Output('graph-predict', 'figure'),
    Output('mae-4h', 'children'),
    Output('mae-24h', 'children'),
    Output('mae-72h', 'children'),

    Input('interval-component-predict', 'n_intervals')  # Добавьте этот Input
)
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
    mae_4h = f"{mae_4h:.1f} %"

    mae_24h = mean_absolute_error(mae['percent_dif'], mae['pred_24h'])
    mae_24h = f"{mae_24h:.1f} %"

    mae_72h = mean_absolute_error(mae['percent_dif'], mae['pred_72h'])
    mae_72h = f"{mae_72h:.1f} %"

    return mae_4h, mae_24h, mae_72h


@callback(
    Output('graph-predict', 'figure'),
    Output('mae-4h-curr', 'children'),
    Output('mae-24h-curr', 'children'),
    Output('mae-72h-curr', 'children'),

    Input('id-dropdown-predict', 'value'),
    Input('period-radio-predict', 'value'),
)
def update_graph(id, period):
    client = clickhouse_connect.get_client(**CLICK_CONN)
    df = client.query_df(f'SELECT * FROM msk_database.prediction WHERE id={id}')

    df = df.sort_values(['datetime'])
    df = df.tail(period * 30 * 24)

    # Создаем subplots с 3 рядами
    fig = make_subplots(
        rows=2, 
        cols=1,
        subplot_titles=(
            f'Прогноз моделей для (МКД) {id}',
            f'Динамика расходов воды', 
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
            showlegend=False  # Скрываем из легенды
        ),
        row=1, col=1
    )

    # Прогнозы
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=np.floor(df['pred_4h'] / 5) * 5, 
                  name='прогноз на 4 часа', line=dict(width=2),
                  legendgroup="pred_4h", showlegend=True),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=np.floor(df['pred_24h'] / 5) * 5, 
                  name='прогноз на сутки', line=dict(width=2),
                  legendgroup="pred_24h", showlegend=True),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=df['datetime'], y=np.floor(df['pred_72h'] / 5) * 5, 
                  name='прогноз на 3 дня', line=dict(width=2),
                  legendgroup="pred_72h", showlegend=True),
        row=1, col=1
    )

    # ЛИНИЯ УРОВНЯ ОПАСНОСТИ (скрываем из легенды)
    fig.add_trace(
        go.Scatter(
            x=df['datetime'],
            y=[40] * len(df),
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            showlegend=False  # Скрываем из легенды
        ),
        row=1, col=1
    )

    # Второй график - подача и выход горячей воды
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['rashod_hot'], 
                  name='Расход горячей, м<sup>3</sup>/ч', line=dict(width=2),
                  legendgroup="rashod_hot", showlegend=True),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['rashod_cold'], 
                  name='Расход холодной, м<sup>3</sup>/ч', line=dict(width=2),
                  legendgroup="rashod_cold", showlegend=True),
        row=2, col=1
    )
    

    fig.update_yaxes(range=[0, 100], row=1, col=1)

    # Обновляем layout
    fig.update_layout(
        height=800,
        title_text="Прогнозная аналитика",
        showlegend=True,
        template="plotly_white",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02
        )
    )
    



    # Обновляем оси
    fig.update_xaxes(title_text="Дата", row=2, col=1)
    fig.update_yaxes(title_text=f"Веротяность инцидента, %", row=1, col=1)
    fig.update_yaxes(title_text="Расход воды, м<sup>3</sup>/ч", row=2, col=1)

    
    # Добавляем сетку
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    mae = df.dropna()
    mae_4h = mean_absolute_error(mae['percent_dif'], mae['pred_4h'])
    mae_4h = f"{mae_4h:.1f} %"

    mae_24h = mean_absolute_error(mae['percent_dif'], mae['pred_24h'])
    mae_24h = f"{mae_24h:.1f} %"

    mae_72h = mean_absolute_error(mae['percent_dif'], mae['pred_72h'])
    mae_72h = f"{mae_72h:.1f} %"
  
    return fig, mae_4h, mae_24h, mae_72h
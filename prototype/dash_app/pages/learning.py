import dash
from dash import dcc, html, Dash, register_page, dash_table, callback
from dash.dependencies import Input, Output, State
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
import time
import sys
import io
sys.path.append("../pages")

from config import CLICK_CONN



register_page(__name__, path="/learning/", name='Прогнозирование')

logs_4h = ['']
logs_24h = ['']
logs_72h = ['']

def prepare_data_for_predict(model='4h'):
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


    # Сдвигаем таргет для трех моделей и добавляем лаговые признаки
    if model=='4h':
        df['target'] = df.groupby('id')['percent_dif'].shift(-4)
    elif model=='24h':
        df['target'] = df.groupby('id')['percent_dif'].shift(-24)
    else:
        df['target'] = df.groupby('id')['percent_dif'].shift(-72)
    for i in tqdm(range(1, 30)):

        df[f'percent_lag{i}'] = df.groupby('id')['percent_dif'].shift(i)
        df[f'dif_lag{i}'] = df.groupby('id')['dif'].shift(i)
        df[f'temp_output_lag{i}'] = df.groupby('id')['temp_output'].shift(i)
    
    return df


def learn_model_get_metric(df, log, model_type='24h', iterations=200, save=False):
    class IterationLogger_4:
        def after_iteration(self, info):

            logs_4h.append(f"Прогресс обучения: {info.iteration / iterations * 100:.2f} %")
            return True
    class IterationLogger_24:
        def after_iteration(self, info):

            logs_24h.append(f"Прогресс обучения: {info.iteration / iterations * 100:.2f} %")
            return True
    class IterationLogger_72:
        def after_iteration(self, info):

            logs_72h.append(f"Прогресс обучения: {info.iteration / iterations * 100:.2f} %")
            return True

    temp = df.dropna()

    X = temp.drop(columns=['datetime', 'target'])
    y = temp['target']


    model = CatBoostRegressor(iterations=iterations, learning_rate=0.01, depth=6, verbose=5)
    if model_type == '4h':
        model.fit(X, y, cat_features=['id'], callbacks=[IterationLogger_4()])
    elif model_type == '24h':
        model.fit(X, y, cat_features=['id'], callbacks=[IterationLogger_24()])
    else:
        model.fit(X, y, cat_features=['id'], callbacks=[IterationLogger_72()])

    # Сохраняем модель если нужно
    if save:
        if model_type == '4h':
            model.save_model('models/model_4h.cbm')
        elif model_type == '24h':
            model.save_model('models/model_24h.cbm')
        else:
            model.save_model('models/model_72h.cbm')


        # log.append(f"Модель сохранена!!! Поздравляю, Вы теперь немножко ML-инженер)")

    mae = mean_absolute_error(y, model.predict(X))
    return mae

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


# Создаем карточки для метрик
learn_4h = dbc.Card(
    [
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Textarea(id='log-output-4h', style={'width': '60%', 'height': 40}),
                        # dcc.Interval(id='interval-24', interval=1000, n_intervals=0),  # Обновление каждую секунду
                        dcc.Loading(
                            id="loading-learn",
                            type="default",
                            children=[
                                html.H4(id="learn_4h-curr", children="Ожидание действия", className="card-text"),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Сложность обучения:"),
                                        dbc.Input(
                                            id=f"iterations-4h",
                                            type="number",
                                            min=5,
                                            max=1000,
                                            step=1,
                                            value=200,
                                            style={"margin-bottom": "10px"}
                                        )
                                    ], width=4),
                                ]),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Button("Обучить модель", id="learn-button-4h", color="primary", className="mt-3")
                                    ], width=6),
                                    dbc.Col([
                                        dbc.Button("Обучить и сохранить", id="learn-save-button-4h", color="secondary", className="mt-3")
                                    ], width=6),
                                ])
                                
                            ]
                        )
                    ])
                ], color="light", outline=True),
            ], width=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Точность используемой модели", className="card-title", style={"font-size": "0.9rem", "text-align": "right"}),
                        html.H4(id="mae-4h-learn", children="расчет...", className="card-text"),
                    ])
                ], color="light", outline=True),
                dbc.Card([
                    dbc.CardBody([
                        html.H3("Модель прогнозирования на 4 часа", className="card-text"),
                    ])
                ], color="light", outline=True),
            ], width=4),
        ])
    ],
    body=True,
    color="secondary",
    outline=True
)

learn_24h = dbc.Card(
    [
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Textarea(id='log-output-24h', style={'width': '60%', 'height': 40}),
                        dcc.Interval(id='interval-24', interval=1000, n_intervals=0),  # Обновление каждую секунду
                        dcc.Loading(
                            id="loading-learn",
                            type="default",
                            children=[
                                html.H4(id="learn_24h-curr", children="Ожидание действия", className="card-text"),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Сложность обучения:"),
                                        dbc.Input(
                                            id=f"iterations-24h",
                                            type="number",
                                            min=5,
                                            max=1000,
                                            step=1,
                                            value=200,
                                            style={"margin-bottom": "10px"}
                                        )
                                    ], width=4),
                                ]),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Button("Обучить модель", id="learn-button-24h", color="primary", className="mt-3")
                                    ], width=6),
                                    dbc.Col([
                                        dbc.Button("Обучить и сохранить", id="learn-save-button-24h", color="secondary", className="mt-3")
                                    ], width=6),
                                ])
                                
                                
                            ]
                        )
                    ])
                ], color="light", outline=True),
            ], width=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Точность используемой модели", className="card-title", style={"font-size": "0.9rem", "text-align": "right"}),
                        html.H4(id="mae-24h-learn", children="расчет...", className="card-text"),
                    ])
                ], color="light", outline=True),
                dbc.Card([
                    dbc.CardBody([
                        html.H3("Модель прогнозирования на 24 часа", className="card-text"),

                    ])
                ], color="light", outline=True),
            ], width=4),
        ])
    ],
    body=True,
    color="secondary",
    outline=True
)


learn_72h = dbc.Card(
    [
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Textarea(id='log-output-72h', style={'width': '60%', 'height': 40}),
                        # dcc.Interval(id='interval-24', interval=1000, n_intervals=0),  # Обновление каждую секунду
                        dcc.Loading(
                            id="loading-learn",
                            type="default",
                            children=[
                                html.H4(id="learn_72h-curr", children="Ожидание действия", className="card-text"),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Сложность обучения:"),
                                        dbc.Input(
                                            id=f"iterations-72h",
                                            type="number",
                                            min=5,
                                            max=1000,
                                            step=1,
                                            value=200,
                                            style={"margin-bottom": "10px"}
                                        )
                                    ], width=4),
                                ]),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Button("Обучить модель", id="learn-button-72h", color="primary", className="mt-3")
                                    ], width=6),
                                    dbc.Col([
                                        dbc.Button("Обучить и сохранить", id="learn-save-button-72h", color="secondary", className="mt-3")
                                    ], width=6),
                                ])
                                
                                
                            ]
                        )
                    ])
                ], color="light", outline=True),
            ], width=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Точность используемой модели", className="card-title", style={"font-size": "0.9rem", "text-align": "right"}),
                        html.H4(id="mae-72h-learn", children="расчет...", className="card-text"),
                    ])
                ], color="light", outline=True),
                dbc.Card([
                    dbc.CardBody([
                        html.H3("Модель прогнозирования на 72 часа", className="card-text"),

                    ])
                ], color="light", outline=True),
            ], width=4),
        ])
    ],
    body=True,
    color="secondary",
    outline=True
)

layout = dbc.Container(
    [
        dbc.Row([
            dcc.Interval(
                        id='interval-component-learn',
                        interval=30*60*1000,  # 30 минут в миллисекундах
                        n_intervals=0
                    ),
            learn_4h
        ]),
        dbc.Row([
            learn_24h
        ]),
        dbc.Row([
            learn_72h
        ]),
    ],
    fluid=True,
)

def train_model_4(save=False, iterations=200):
    logs_4h.append('Подготовка данных для обучения...')
    df = prepare_data_for_predict(model='4h')
    mae = learn_model_get_metric(df, logs_4h, model_type='4h', iterations=iterations, save=save)
    return mae


@callback(
    Output('log-output-4h', 'value'),
    Input('interval-24', 'n_intervals')
)
def update_logs(n):
    return '\n'.join(logs_4h[-1:])  # показываем последние 20 записей логов


# Callback для обработки нажатия кнопки
@callback(
    Output("learn_4h-curr", "children"),
    Output("learn-button-4h", "disabled"),  # блокируем кнопку во время выполненияб
    Output("learn-save-button-4h", "disabled"),
    Input("learn-button-4h", "n_clicks"),
    Input("learn-save-button-4h", "n_clicks"),
    State("iterations-4h", "value"),
)
def on_learn_click(n_clicks_learn, n_clicks_save, iterations):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, False, False
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    # Блокируем обе кнопки
    logs_4h = ['Старт обучения модели 4h']
    

    if button_id == "learn-save-button-4h":
        logs_4h.append(f'Модель будет сохранена')
        result = train_model_4(save=True, iterations=iterations)
    else:
        logs_4h.append('Обучение без сохранения модели')
        result = train_model_4(save=False, iterations=iterations)
    
    # logs_4h.append('Модель сохранена!!! Поздравляю, Вы теперь немножко ML-инженер)')

    return f"Точность модели: {100 - result:.1f} %", False, False




def train_model_24(save=False, iterations=200):
    logs_24h.append('Подготовка данных для обучения...')
    df = prepare_data_for_predict(model='24h')
    mae = learn_model_get_metric(df, logs_24h, model_type='24h', iterations=iterations, save=save)
    return mae


@callback(
    Output('log-output-24h', 'value'),
    Input('interval-24', 'n_intervals')
)
def update_logs(n):
    return '\n'.join(logs_24h[-1:])  # показываем последние 20 записей логов


# Callback для обработки нажатия кнопки
@callback(
    Output("learn_24h-curr", "children"),
    Output("learn-button-24h", "disabled"),  # блокируем кнопку во время выполненияб
    Output("learn-save-button-24h", "disabled"),
    Input("learn-button-24h", "n_clicks"),
    Input("learn-save-button-24h", "n_clicks"),
    State("iterations-24h", "value"),
)
def on_learn_click(n_clicks_learn, n_clicks_save, iterations):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, False, False
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    # Блокируем обе кнопки
    logs_24h = ['Старт обучения модели 24h']
    

    if button_id == "learn-save-button-24h":
        logs_24h.append(f'Модель будет сохранена')
        result = train_model_24(save=True, iterations=iterations)
    else:
        logs_24h.append('Обучение без сохранения модели')
        result = train_model_24(save=False, iterations=iterations)
    
    # logs_24h.append('Обучение модели 24h завершено')

    return f"Точность модели: {100 - result:.1f} %", False, False


def train_model_72(save=False, iterations=200):
    logs_72h.append('Подготовка данных для обучения...')
    df = prepare_data_for_predict(model='72h')
    mae = learn_model_get_metric(df, logs_72h, model_type='72h', iterations=iterations, save=save)
    return mae


@callback(
    Output('log-output-72h', 'value'),
    Input('interval-24', 'n_intervals')
)
def update_logs(n):
    return '\n'.join(logs_72h[-1:])  # показываем последние 20 записей логов


# Callback для обработки нажатия кнопки
@callback(
    Output("learn_72h-curr", "children"),
    Output("learn-button-72h", "disabled"),  # блокируем кнопку во время выполненияб
    Output("learn-save-button-72h", "disabled"),
    Input("learn-button-72h", "n_clicks"),
    Input("learn-save-button-72h", "n_clicks"),
    State("iterations-72h", "value"),
)
def on_learn_click(n_clicks_learn, n_clicks_save, iterations):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, False, False
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    # Блокируем обе кнопки
    logs_72h = ['Старт обучения модели 72h']
    

    if button_id == "learn-save-button-72h":
        logs_72h.append(f'Модель будет сохранена')
        result = train_model_72(save=True, iterations=iterations)
    else:
        logs_72h.append('Обучение без сохранения модели')
        result = train_model_72(save=False, iterations=iterations)
    
    # logs_72h.append('Модель сохранена!!! Поздравляю, Вы теперь немножко ML-инженер)')

    return f"Точность модели: {100 - result:.1f} %", False, False




@callback(
    Output('mae-4h-learn', 'children'),
    Output('mae-24h-learn', 'children'),
    Output('mae-72h-learn', 'children'),
    Input('interval-component-learn', 'n_intervals'))
def get_current_mae(n):


    client = clickhouse_connect.get_client(**CLICK_CONN)

    df = client.query_df(
        f"""
            SELECT percent_dif, pred_4h, pred_24h, pred_72h from msk_database.prediction
        """
    )

    mae = df.dropna()
    mae_4h = mean_absolute_error(mae['percent_dif'], mae['pred_4h'])
    mae_4h = f"{100 - mae_4h:.1f} %"

    mae_24h = mean_absolute_error(mae['percent_dif'], mae['pred_24h'])
    mae_24h = f"{100 - mae_24h:.1f} %"

    mae_72h = mean_absolute_error(mae['percent_dif'], mae['pred_72h'])
    mae_72h = f"{100 - mae_72h:.1f} %"

    return mae_4h, mae_24h, mae_72h

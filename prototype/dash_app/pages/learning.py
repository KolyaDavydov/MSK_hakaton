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

register_page(__name__, path="/learning/", name='Обучение моделей')

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

PROGRESS_CARD_STYLE = {
    "border": "none",
    "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.05)",
    "borderRadius": "8px",
    "background": "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)",
    "color": "white"
}

# Используем хранилище для логов вместо глобальных переменных
logs_4h = []
logs_24h = [] 
logs_72h = []

def prepare_data_for_predict(model='4h'):
    client = clickhouse_connect.get_client(**CLICK_CONN)

    df = client.query_df(
        f"""
        SELECT *
        FROM msk_database.analytic
        ORDER BY id, datetime
        """
    )

    df['dif'] = df['rashod_cold'] - df['rashod_hot']
    df['percent_dif'] = df['dif'] / df['rashod_cold'] * 100
    del df['cumulative_rashod_cold']

    if model == '4h':
        df['target'] = df.groupby('id')['percent_dif'].shift(-4)
    elif model == '24h':
        df['target'] = df.groupby('id')['percent_dif'].shift(-24)
    else:
        df['target'] = df.groupby('id')['percent_dif'].shift(-72)
        
    for i in tqdm(range(1, 30)):
        df[f'percent_lag{i}'] = df.groupby('id')['percent_dif'].shift(i)
        df[f'dif_lag{i}'] = df.groupby('id')['dif'].shift(i)
        df[f'temp_output_lag{i}'] = df.groupby('id')['temp_output'].shift(i)
    
    return df

def learn_model_get_metric(df, log_list, model_type='24h', iterations=200, save=False):
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

    if save:
        if model_type == '4h':
            model.save_model('models/model_4h.cbm')
        elif model_type == '24h':
            model.save_model('models/model_24h.cbm')
        else:
            model.save_model('models/model_72h.cbm')

    mae = mean_absolute_error(y, model.predict(X))
    return mae

# Заголовок страницы
header = dbc.Card(
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.H2("🧠 Обучение моделей прогнозирования", 
                       style={'color': 'white', 'margin': '0', 'fontWeight': '600'}),
                html.P("Настройка и обучение алгоритмов машинного обучения для прогнозирования инцидентов", 
                      style={'color': 'rgba(255,255,255,0.8)', 'margin': '0', 'fontSize': '14px'})
            ]),
            # dbc.Col([
            #     html.Div([
            #         html.I(className="fas fa-brain", style={'marginRight': '8px'}),
            #         "Обновление моделей каждые 30 минут"
            #     ], style={'color': 'white', 'textAlign': 'right', 'fontSize': '14px'})
            # ], width="auto")
        ])
    ]),
    style=CARD_STYLE,
    className="mb-4"
)

# Универсальная функция для создания карточки модели
def create_model_card(model_name, model_title, icon_class, color_scheme):
    return dbc.Card(
        dbc.CardBody([
            html.H5(f"{model_title}", 
                   className="card-title", style={'color': '#2c3e50', 'marginBottom': '20px'}),
            
            dbc.Row([
                # Левая колонка - метрики и параметры
                dbc.Col([
                    # Карточка точности
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("📊 Текущая точность модели", 
                                   style={'color': '#2c3e50', 'marginBottom': '15px'}),
                            html.Div([
                                html.I(className=icon_class, style={'fontSize': '40px', 'marginBottom': '10px', 'color': color_scheme}),
                                html.H2(id=f"mae-{model_name}-learn", children="...", 
                                       style={'margin': '0', 'fontWeight': '600'}),
                                html.P("Точность прогноза", style={'margin': '0', 'fontSize': '14px'})
                            ], style={'textAlign': 'center'})
                        ])
                    ], style=METRIC_CARD_STYLE),
                    
                    # Карточка параметров
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("⚙️ Параметры обучения", 
                                   style={'color': '#2c3e50', 'marginBottom': '15px'}),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Количество итераций (5-1000):", 
                                              style={'fontWeight': '600', 'color': '#34495e', 'marginBottom': '8px'}),
                                    dbc.Input(
                                        id=f"iterations-{model_name}",
                                        type="number",
                                        min=5,
                                        max=1000,
                                        step=1,
                                        value=200,
                                        style={"marginBottom": "15px"}
                                    )
                                ]),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button(
                                        "🚀 Обучить модель", 
                                        id=f"learn-button-{model_name}", 
                                        color="primary", 
                                        className="w-100 mb-2",
                                        size="lg"
                                    )
                                ]),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button(
                                        "💾 Обучить и сохранить", 
                                        id=f"learn-save-button-{model_name}", 
                                        color="success", 
                                        className="w-100",
                                        size="lg"
                                    )
                                ]),
                            ])
                        ])
                    ], style=CONTENT_CARD_STYLE),
                ], width=4),
                
                # Правая колонка - процесс обучения
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("📈 Процесс обучения", 
                                   style={'color': '#2c3e50', 'marginBottom': '15px'}),
                            # Статус обучения - всегда видимый
                            dbc.Card([
                                dbc.CardBody([
                                    html.Div([
                                        html.I(className="fas fa-clock", 
                                              style={'fontSize': '30px', 'marginBottom': '10px', 'color': color_scheme}),
                                        html.H4(id=f"learn_{model_name}-curr", children="Ожидание запуска обучения", 
                                               style={'margin': '0', 'fontWeight': '600'}),
                                        html.P("Готов к обучению модели", style={'margin': '0', 'fontSize': '14px'})
                                    ], style={'textAlign': 'center'})
                                ])
                            ], style={'background': 'linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)', 'border': 'none', 'marginBottom': '20px'}),
                            
                            # Логи обучения - всегда видимые
                            html.Div([
                                html.Label("Лог обучения:", 
                                          style={'fontWeight': '600', 'color': '#34495e', 'marginBottom': '8px'}),
                                dcc.Textarea(
                                    id=f'log-output-{model_name}',
                                    style={
                                        'width': '100%', 
                                        'height': '150px',
                                        'borderRadius': '8px',
                                        'padding': '10px',
                                        'fontSize': '12px',
                                        'border': '1px solid #e0e0e0',
                                        'backgroundColor': '#f8f9fa',
                                        'fontFamily': 'monospace'
                                    },
                                    readOnly=True,
                                    value='Лог обучения будет отображаться здесь...'
                                )
                            ], style={'marginTop': '20px'})
                        ])
                    ], style=CONTENT_CARD_STYLE),
                ], width=8)
            ])
        ]),
        style=CONTENT_CARD_STYLE,
        className="mb-4"
    )

# Создаем карточки для каждой модели
learn_4h = create_model_card("4h", "Модель прогнозирования на 4 часа", "fas fa-clock", "#3498db")
learn_24h = create_model_card("24h", "Модель прогнозирования на 24 часа", "fas fa-calendar-day", "#e67e22")
learn_72h = create_model_card("72h", "Модель прогнозирования на 72 часа", "fas fa-calendar-alt", "#e74c3c")

layout = dbc.Container(
    [
        header,
        learn_4h,
        learn_24h,
        learn_72h,
        
        # Интервалы обновления
        dcc.Interval(
            id='interval-component-learn',
            interval=30*60*1000,
            n_intervals=0
        ),
        dcc.Interval(
            id='interval-logs',
            interval=1000,  # Обновление логов каждую секунду
            n_intervals=0
        ),
        
        # Хранилище для состояния обучения
        dcc.Store(id='training-state-4h', data={'training': False}),
        dcc.Store(id='training-state-24h', data={'training': False}),
        dcc.Store(id='training-state-72h', data={'training': False}),
    ],
    fluid=True,
    style={'backgroundColor': '#f8f9fa', 'minHeight': '100vh', 'padding': '20px'}
)

# Функции обучения моделей
def train_model_4(save=False, iterations=200):
    logs_4h.clear()
    logs_4h.append('🚀 Подготовка данных для обучения...')
    df = prepare_data_for_predict(model='4h')
    logs_4h.append('📊 Данные подготовлены, начинаем обучение...')
    mae = learn_model_get_metric(df, logs_4h, model_type='4h', iterations=iterations, save=save)
    logs_4h.append(f'✅ Обучение завершено! Точность модели: {100 - mae:.1f}%')
    return mae

def train_model_24(save=False, iterations=200):
    logs_24h.clear()
    logs_24h.append('🚀 Подготовка данных для обучения...')
    df = prepare_data_for_predict(model='24h')
    logs_24h.append('📊 Данные подготовлены, начинаем обучение...')
    mae = learn_model_get_metric(df, logs_24h, model_type='24h', iterations=iterations, save=save)
    logs_24h.append(f'✅ Обучение завершено! Точность модели: {100 - mae:.1f}%')
    return mae

def train_model_72(save=False, iterations=200):
    logs_72h.clear()
    logs_72h.append('🚀 Подготовка данных для обучения...')
    df = prepare_data_for_predict(model='72h')
    logs_72h.append('📊 Данные подготовлены, начинаем обучение...')
    mae = learn_model_get_metric(df, logs_72h, model_type='72h', iterations=iterations, save=save)
    logs_72h.append(f'✅ Обучение завершено! Точность модели: {100 - mae:.1f}%')
    return mae

# Callbacks для обновления логов
@callback(
    Output('log-output-4h', 'value'),
    Input('interval-logs', 'n_intervals')
)
def update_logs_4h(n):
    return '\n'.join(logs_4h[-5:]) if logs_4h else 'Лог обучения пуст'

@callback(
    Output('log-output-24h', 'value'),
    Input('interval-logs', 'n_intervals')
)
def update_logs_24h(n):
    return '\n'.join(logs_24h[-5:]) if logs_24h else 'Лог обучения пуст'

@callback(
    Output('log-output-72h', 'value'),
    Input('interval-logs', 'n_intervals')
)
def update_logs_72h(n):
    return '\n'.join(logs_72h[-5:]) if logs_72h else 'Лог обучения пуст'

# Callbacks для обучения моделей
@callback(
    Output("learn_4h-curr", "children"),
    Output("learn-button-4h", "disabled"),
    Output("learn-save-button-4h", "disabled"),
    Output("training-state-4h", "data"),
    Input("learn-button-4h", "n_clicks"),
    Input("learn-save-button-4h", "n_clicks"),
    State("iterations-4h", "value"),
    State("training-state-4h", "data"),
    prevent_initial_call=True
)
def on_learn_click_4h(n_clicks_learn, n_clicks_save, iterations, training_state):
    if training_state.get('training', False):
        return dash.no_update, True, True, training_state
        
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, False, False, training_state
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Устанавливаем состояние обучения
    training_state['training'] = True
    
    # Обновляем статус
    status_msg = "⏳ Обучение модели..."
    
    try:
        if button_id == "learn-save-button-4h":
            logs_4h.append('💾 Модель будет сохранена после обучения')
            result = train_model_4(save=True, iterations=iterations)
        else:
            logs_4h.append('📝 Обучение без сохранения модели')
            result = train_model_4(save=False, iterations=iterations)
        
        final_msg = f"✅ Точность модели: {100 - result:.1f}%"
        
    except Exception as e:
        logs_4h.append(f'❌ Ошибка обучения: {str(e)}')
        final_msg = "❌ Ошибка обучения"
        result = 100  # Значение по умолчанию при ошибке
    
    # Сбрасываем состояние обучения
    training_state['training'] = False
    
    return final_msg, False, False, training_state

@callback(
    Output("learn_24h-curr", "children"),
    Output("learn-button-24h", "disabled"),
    Output("learn-save-button-24h", "disabled"),
    Output("training-state-24h", "data"),
    Input("learn-button-24h", "n_clicks"),
    Input("learn-save-button-24h", "n_clicks"),
    State("iterations-24h", "value"),
    State("training-state-24h", "data"),
    prevent_initial_call=True
)
def on_learn_click_24h(n_clicks_learn, n_clicks_save, iterations, training_state):
    if training_state.get('training', False):
        return dash.no_update, True, True, training_state
        
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, False, False, training_state
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    training_state['training'] = True
    status_msg = "⏳ Обучение модели..."
    
    try:
        if button_id == "learn-save-button-24h":
            logs_24h.append('💾 Модель будет сохранена после обучения')
            result = train_model_24(save=True, iterations=iterations)
        else:
            logs_24h.append('📝 Обучение без сохранения модели')
            result = train_model_24(save=False, iterations=iterations)
        
        final_msg = f"✅ Точность модели: {100 - result:.1f}%"
        
    except Exception as e:
        logs_24h.append(f'❌ Ошибка обучения: {str(e)}')
        final_msg = "❌ Ошибка обучения"
        result = 100
    
    training_state['training'] = False
    
    return final_msg, False, False, training_state

@callback(
    Output("learn_72h-curr", "children"),
    Output("learn-button-72h", "disabled"),
    Output("learn-save-button-72h", "disabled"),
    Output("training-state-72h", "data"),
    Input("learn-button-72h", "n_clicks"),
    Input("learn-save-button-72h", "n_clicks"),
    State("iterations-72h", "value"),
    State("training-state-72h", "data"),
    prevent_initial_call=True
)
def on_learn_click_72h(n_clicks_learn, n_clicks_save, iterations, training_state):
    if training_state.get('training', False):
        return dash.no_update, True, True, training_state
        
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, False, False, training_state
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    training_state['training'] = True
    status_msg = "⏳ Обучение модели..."
    
    try:
        if button_id == "learn-save-button-72h":
            logs_72h.append('💾 Модель будет сохранена после обучения')
            result = train_model_72(save=True, iterations=iterations)
        else:
            logs_72h.append('📝 Обучение без сохранения модели')
            result = train_model_72(save=False, iterations=iterations)
        
        final_msg = f"✅ Точность модели: {100 - result:.1f}%"
        
    except Exception as e:
        logs_72h.append(f'❌ Ошибка обучения: {str(e)}')
        final_msg = "❌ Ошибка обучения"
        result = 100
    
    training_state['training'] = False
    
    return final_msg, False, False, training_state

@callback(
    Output('mae-4h-learn', 'children'),
    Output('mae-24h-learn', 'children'),
    Output('mae-72h-learn', 'children'),
    Input('interval-component-learn', 'n_intervals')
)
def get_current_mae(n):
    client = clickhouse_connect.get_client(**CLICK_CONN)

    try:
        df = client.query_df(
            f"""
                SELECT percent_dif, pred_4h, pred_24h, pred_72h from msk_database.prediction
            """
        )

        mae = df.dropna()
        
        if len(mae) > 0:
            mae_4h = mean_absolute_error(mae['percent_dif'], mae['pred_4h'])
            mae_4h = f"{100 - mae_4h:.1f}%"
            
            mae_24h = mean_absolute_error(mae['percent_dif'], mae['pred_24h'])
            mae_24h = f"{100 - mae_24h:.1f}%"
            
            mae_72h = mean_absolute_error(mae['percent_dif'], mae['pred_72h'])
            mae_72h = f"{100 - mae_72h:.1f}%"
        else:
            mae_4h = "N/A"
            mae_24h = "N/A"
            mae_72h = "N/A"
            
    except Exception as e:
        mae_4h = "Ошибка"
        mae_24h = "Ошибка"
        mae_72h = "Ошибка"

    return mae_4h, mae_24h, mae_72h
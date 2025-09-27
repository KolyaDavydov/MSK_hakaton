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

register_page(__name__, path="/", name='Аналитика')

# Создание клиента к БД
client = clickhouse_connect.get_client(**CLICK_CONN)


result = client.query('SELECT DISTINCT id FROM msk_database.analytic')
# Получим уникальные значения id в виде списка Python
unique_ids = [row[0] for row in result.result_rows]
# print(unique_ids)

# def get_data_by_id(id=1):
#     print

id_selection = dbc.Card(
    [
        dcc.Dropdown(
            id="id-dropdown",
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

period_selection = dbc.Card(
    [
        dcc.RadioItems(
            id='period-radio',
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

layout = dbc.Container(
    [
        dbc.Row([
            dbc.Col([
                dbc.Row(
                    id_selection,
                    style={
                        "height": 20,
                        "margin-left": 20,
                        "margin-right": 20,
                        "margin-top": 5}
                ),

            ]),
            dbc.Col([
                dbc.Row(
                    period_selection,
                    style={"margin-top": 5, "height": 80, "margin-right": 20,}),
            ])
        ]),
        dbc.Row([

            dcc.Graph(id='graph', style={"height": 750}),
                # Добавьте этот компонент для автоматического обновления
            dcc.Interval(
                id='interval-component',
                interval=30*60*1000,  # 30 минут в миллисекундах
                n_intervals=0
            )
            # style={
            #     "height": 20,
            #     "margin-left": 20,
            #     "margin-right": 20,
            #     "margin-top": 50}
        ]),
        dbc.Row([
            dbc.Col([
                html.H4("Горячее вооснабжение", style={"textAlign": "center", "margin-bottom": "10px"}),
                dash_table.DataTable(
                    id='hot-table',
                    columns=[
                        {"name": "Дата", "id": "datetime", "type": "datetime"},
                        {"name": "Время", "id": "hour"},
                        {"name": "Подача, м3", "id": 'input_hot', "type": "numeric", "format": {"specifier": ".2f"}},
                        {"name": "Обратка, м3", "id": 'output_hot', "type": "numeric", "format": {"specifier": ".2f"}},
                        {"name": "Потребление за период, м3", "id": 'rashod_hot', "type": "numeric", "format": {"specifier": ".2f"}},
                        {"name": "Т1 гвс, оС", "id": "temp_input"},
                        {"name": "Т2 гвс, оС", "id": "temp_output"},

                    ],
                    data=[],
                    style_header={'fontWeight': 'bold', 'textAlign': 'center'},  
                    style_table={'height': '600px', 'overflowX': 'auto'},

                    # Включение экспорта
                    export_format='xlsx',  # или 'xlsx', 'none'
                    export_headers='display',
                    
                )
            ], width=6),
            dbc.Col([
                html.H4("Холодное вооснабжение", style={"textAlign": "center", "margin-bottom": "10px"}),
                dash_table.DataTable(
                    id='cold-table',
                    columns=[
                        {"name": "Дата", "id": "datetime"},
                        {"name": "Время", "id": "hour"},
                        {"name": "Потребление накопленным итогом, м3", "id": "cumulative_rashod_cold", "type": "numeric", "format": {"specifier": ".3f"}},
                        {"name": "Потребление за период, м3", "id": "rashod_cold", "type": "numeric", "format": {"specifier": ".2f"}}

                    ],
                    data=[],  
                    style_header={'fontWeight': 'bold', 'textAlign': 'center'},
                    style_table={'height': '600px', 'overflowY': 'auto'},
                    # Включение экспорта
                    export_format='xlsx',  # или 'xlsx', 'none'
                    export_headers='display',
                    
                )
            ], width=5),

        ])

    ],
    fluid=True,
)

@callback(
    Output('graph', 'figure'),
    Output('hot-table', 'data'),
    Output('cold-table', 'data'),
    Input('id-dropdown', 'value'),
    Input('period-radio', 'value'),
    Input('interval-component', 'n_intervals')  # Добавьте этот Input
)
def update_graph(id, period, n):
    client = clickhouse_connect.get_client(**CLICK_CONN)
    df = client.query_df(f'SELECT * FROM msk_database.analytic WHERE id={id}')
    df['dif'] = df['rashod_cold'] - df['rashod_hot'] - 0.05

    df = df.sort_values(['datetime'])
    df = df.tail(period * 30 * 24)

    # Создаем subplots с 3 рядами
    fig = make_subplots(
        rows=3, 
        cols=1,
        subplot_titles=(
            'Индивидуальный тепловой пункт (ИТП)',
            f'Многокартирный жилой дом (МКД) {id}', 
            'Температура входа и выхода горячей воды в МКД'
        ),
        vertical_spacing=0.08
    )


    # Первый график - расходы
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['rashod_hot'], 
                  name='Расход горячей, м<sup>3</sup>/ч', line=dict(width=2),
                  legendgroup="group1", showlegend=True),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['rashod_cold'], 
                  name='Расход холодной, м<sup>3</sup>/ч', line=dict(width=2),
                  legendgroup="group1", showlegend=True),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['dif'], 
                  name='Утечка, м<sup>3</sup>/ч', line=dict(width=2),
                  legendgroup="group1", showlegend=True),
        row=1, col=1
    )
    
    # Второй график - подача и выход горячей воды
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['input_hot'], 
                  name='Подача горячей, м<sup>3</sup>/ч', line=dict(width=2),
                  legendgroup="group2", showlegend=True),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['output_hot'], 
                  name='Выход горячей, м<sup>3</sup>/ч', line=dict(width=3),
                  legendgroup="group2", showlegend=True),
        row=2, col=1
    )
    
    # Третий график - температуры
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['temp_input'], 
                  name='Температура входа', line=dict(width=2),
                  legendgroup="group3", showlegend=True),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['temp_output'], 
                  name='Температура выхода', line=dict(width=2),
                  legendgroup="group3", showlegend=True),
        row=3, col=1
    )
    
    # Обновляем layout
    fig.update_layout(
        height=800,
        title_text="Анализ системы водоснабжения",
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
    
        # Создаем отдельные легенды для каждого подграфика
    fig.update_layout(
        # Легенда для первого графика (верхний)
        legend1=dict(
            tracegroupgap=0,
            orientation="v",
            yanchor="middle",
            y=0.85,  # Позиция по вертикали для первого графика
            xanchor="left",
            x=1.02,  # Справа от графика
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1,
            font=dict(size=10)
        ),
        # Легенда для второго графика (средний)
        legend2=dict(
            tracegroupgap=0,
            orientation="v",
            yanchor="middle",
            y=0.50,  # Позиция по вертикали для второго графика
            xanchor="left",
            x=1.02,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1,
            font=dict(size=10)
        ),
        # Легенда для третьего графика (нижний)
        legend3=dict(
            tracegroupgap=0,
            orientation="v",
            yanchor="middle",
            y=0.15,  # Позиция по вертикали для третьего графика
            xanchor="left",
            x=1.02,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1,
            font=dict(size=10)
        )
    )
    
    # Указываем, какие traces относятся к каким легендам
    fig.update_traces(legendgroup="group1", legend="legend1", selector=dict(name='Расход горячей, м<sup>3</sup>/ч'))
    fig.update_traces(legendgroup="group1", legend="legend1", selector=dict(name='Расход холодной, м<sup>3</sup>/ч'))
    fig.update_traces(legendgroup="group1", legend="legend1", selector=dict(name='Утечка, м<sup>3</sup>/ч'))
    
    fig.update_traces(legendgroup="group2", legend="legend2", selector=dict(name='Подача горячей, м<sup>3</sup>/ч'))
    fig.update_traces(legendgroup="group2", legend="legend2", selector=dict(name='Выход горячей, м<sup>3</sup>/ч'))
    
    fig.update_traces(legendgroup="group3", legend="legend3", selector=dict(name='Температура входа'))
    fig.update_traces(legendgroup="group3", legend="legend3", selector=dict(name='Температура выхода'))


    # Обновляем оси
    fig.update_xaxes(title_text="Дата", row=3, col=1)
    fig.update_yaxes(title_text=f"м<sup>3</sup>/ч", row=1, col=1)
    fig.update_yaxes(title_text="м<sup>3</sup>/ч", row=2, col=1)
    fig.update_yaxes(title_text="Т, <sup>o</sup>C", row=3, col=1)
    
    # Добавляем сетку
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)



    df = df[['datetime', 'input_hot', 'output_hot', 'rashod_hot', 'temp_input', 'temp_output', 'cumulative_rashod_cold', 'rashod_cold']]
    df['hour_1'] = df['datetime'].dt.hour
    df['hour_2'] = df['hour_1'] + 1
    df['hour'] = (df['hour_1']).astype(str) + '-' + (df['hour_2']).astype(str)
    df['datetime'] = df['datetime'].dt.date

    df_cold = df[['datetime','hour', 'cumulative_rashod_cold', 'rashod_cold']]
    df_hot = df[['datetime', 'hour', 'input_hot', 'output_hot', 'rashod_hot', 'temp_input', 'temp_output']]
    df_cold = df_cold.to_dict('records')
    df_hot = df_hot.to_dict('records')
  
    return fig, df_hot, df_cold
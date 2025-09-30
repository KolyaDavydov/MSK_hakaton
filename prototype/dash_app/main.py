import dash
from dash import Dash, html, dcc, callback, Input, Output
import os
import dash_bootstrap_components as dbc
import base64
import warnings
warnings.filterwarnings('ignore')

app = Dash(__name__,
           suppress_callback_exceptions=True,
           external_stylesheets=[dbc.themes.BOOTSTRAP],
           use_pages=True)

analytic_image = base64.b64encode(open('images/analytic.png', 'rb').read())
predictive_image = base64.b64encode(open('images/predictive.png', 'rb').read())
learning_image = base64.b64encode(open('images/learning.png', 'rb').read())
# modeling_image = base64.b64encode(open('images/modeling.png', 'rb').read())


app.layout = html.Div(
    [
        dcc.Location(id='url', refresh=False),  # Добавляем компонент для отслеживания URL
        html.Header(
            children=[
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Link(
                                html.Img(
                                    src='data:image/png;base64,{}'.format(analytic_image.decode()),
                                    alt="Logo",
                                    id="analytic-img",  # Добавляем id
                                    style={"margin-left": 30,
                                           "margin-top": 15,
                                           "width": 155,
                                           "height": 45},
                                ),
                                href="/",
                            ), width="auto"),
                        dbc.Col(
                            dcc.Link(
                                html.Img(
                                    src='data:image/png;base64,{}'.format(predictive_image.decode()),
                                    alt="Logo",
                                    id="predictive-img",  # Добавляем id
                                    style={"margin-left": 0,
                                           "margin-top": 15,
                                           "width": 155,
                                           "height": 45},
                                ),
                                href="/predictive/",
                            ), width="auto"),
                        dbc.Col(
                            dcc.Link(
                                html.Img(
                                    src='data:image/png;base64,{}'.format(learning_image.decode()),
                                    alt="Logo",
                                    id="learning-img",  # Добавляем id
                                    style={"margin-left": 0,
                                           "margin-top": 15,
                                           "width": 155,
                                           "height": 45},
                                ),
                                href="/learning/",
                            ), width="auto"),
                        dbc.Col(
                            html.H1('Сервис аналитики водоснабжения и прогнозирования инцидентов', style={'font-weight': 'bold', "height": 30, 'text-align': 'center', 'color': '#2c3e50', 'margin-top':10}),
                                width=True
                            )
                    ]
                )
            ],
        ),
        dash.page_container,
    ]
)

# Callback для обновления стилей активной вкладки
@callback(
    [Output("analytic-img", "style"),
     Output("predictive-img", "style"),
     Output("learning-img", "style")],
    #  Output("modeling-img", "style")],
    [Input("url", "pathname")]
)
def update_active_tab(pathname):
    # Базовый стиль для всех изображений
    base_style = {
        "margin-left": 0,
        "margin-top": 15,
        "width": 155,
        "height": 45,
        "transition": "all 0.3s ease"  # Плавный переход
    }
    
    # Стиль для активной вкладки
    active_style = {
        **base_style,
        "border": "3px solid #007bff",
        "border-radius": "8px",
        "padding": "2px",
        "box-shadow": "0 4px 8px rgba(0,123,255,0.3)"
    }
    
    # Определяем активную страницу
    if pathname == "/":
        return [active_style, base_style, base_style]
    elif pathname == "/predictive/":
        return [base_style, active_style, base_style]
    elif pathname == "/learning/":
        return [base_style, base_style, active_style]
    # elif pathname == "/modeling/":
    #     return [base_style, base_style, base_style, active_style]
    else:
        return [base_style, base_style, base_style]

if __name__ == "__main__":
    # app.run(port=8124, host='0.0.0.0', debug=True)
    app.run(port=8124, host='0.0.0.0')

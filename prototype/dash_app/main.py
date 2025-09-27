import dash
from dash import Dash, html, dcc
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
modeling_image = base64.b64encode(open('images/modeling.png', 'rb').read())

app.layout = html.Div(
    [
        html.Header(
            children=[
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Link(
                                html.Img(
                                    src='data:image/png;base64,{}'.format(analytic_image.decode()),
                                    alt="Logo",
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
                                    style={"margin-left": 0,
                                           "margin-top": 15,
                                           "width": 155,
                                           "height": 45},
                                ),
                                href="/learning/",
                            ), width="auto"),
                        dbc.Col(
                            dcc.Link(
                                html.Img(
                                    src='data:image/png;base64,{}'.format(modeling_image.decode()),
                                    alt="Logo",
                                    style={"margin-left": 0,
                                           "margin-top": 15,
                                           "width": 155,
                                           "height": 45},
                                ),
                                href="/modeling/",
                            ), width="auto"),

                        dbc.Col(
                            html.H1('Прототип прогнозирования и аналитики',
                                         style={"margin-top": 5,
                                                "margin-bottom": 15}),
                                width=True
                            )
                    ]
                )
            ],
        ),
        dash.page_container,
    ]
)

if __name__ == "__main__":
    app.run(port=8124, host='0.0.0.0', debug=True)

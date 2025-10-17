# import dash
# from dash import Dash, html, dcc, callback, Input, Output
# import os
# import dash_bootstrap_components as dbc
# import base64
# import warnings
# warnings.filterwarnings('ignore')

# app = Dash(__name__,
#            suppress_callback_exceptions=True,
#            external_stylesheets=[dbc.themes.BOOTSTRAP],
#            use_pages=True)

# analytic_image = base64.b64encode(open('images/analytic.png', 'rb').read())
# predictive_image = base64.b64encode(open('images/predictive.png', 'rb').read())
# learning_image = base64.b64encode(open('images/learning.png', 'rb').read())
# # modeling_image = base64.b64encode(open('images/modeling.png', 'rb').read())


# app.layout = html.Div(
#     [
#         dcc.Location(id='url', refresh=False),  # Добавляем компонент для отслеживания URL
#         html.Header(
#             children=[
#                 dbc.Row(
#                     [
#                         dbc.Col(
#                             dcc.Link(
#                                 html.Img(
#                                     src='data:image/png;base64,{}'.format(analytic_image.decode()),
#                                     alt="Logo",
#                                     id="analytic-img",  # Добавляем id
#                                     style={"margin-left": 30,
#                                            "margin-top": 15,
#                                            "width": 155,
#                                            "height": 45},
#                                 ),
#                                 href="/",
#                             ), width="auto"),
#                         dbc.Col(
#                             dcc.Link(
#                                 html.Img(
#                                     src='data:image/png;base64,{}'.format(predictive_image.decode()),
#                                     alt="Logo",
#                                     id="predictive-img",  # Добавляем id
#                                     style={"margin-left": 0,
#                                            "margin-top": 15,
#                                            "width": 155,
#                                            "height": 45},
#                                 ),
#                                 href="/predictive/",
#                             ), width="auto"),
#                         dbc.Col(
#                             dcc.Link(
#                                 html.Img(
#                                     src='data:image/png;base64,{}'.format(learning_image.decode()),
#                                     alt="Logo",
#                                     id="learning-img",  # Добавляем id
#                                     style={"margin-left": 0,
#                                            "margin-top": 15,
#                                            "width": 155,
#                                            "height": 45},
#                                 ),
#                                 href="/learning/",
#                             ), width="auto"),
#                         dbc.Col(
#                             html.H1('Сервис аналитики и прогнозирования', style={'font-weight': 'bold', "height": 25, 'text-align': 'center', 'color': '#2c3e50', 'margin-top':10}),
#                                 width=True
#                             )
#                     ]
#                 )
#             ],
#         ),
#         dash.page_container,
#     ]
# )

# # Callback для обновления стилей активной вкладки
# @callback(
#     [Output("analytic-img", "style"),
#      Output("predictive-img", "style"),
#      Output("learning-img", "style")],
#     #  Output("modeling-img", "style")],
#     [Input("url", "pathname")]
# )
# def update_active_tab(pathname):
#     # Базовый стиль для всех изображений
#     base_style = {
#         "margin-left": 0,
#         "margin-top": 15,
#         "width": 155,
#         "height": 45,
#         "transition": "all 0.3s ease"  # Плавный переход
#     }
    
#     # Стиль для активной вкладки
#     active_style = {
#         **base_style,
#         "border": "3px solid #007bff",
#         "border-radius": "8px",
#         "padding": "2px",
#         "box-shadow": "0 4px 8px rgba(0,123,255,0.3)"
#     }
    
#     # Определяем активную страницу
#     if pathname == "/":
#         return [active_style, base_style, base_style]
#     elif pathname == "/predictive/":
#         return [base_style, active_style, base_style]
#     elif pathname == "/learning/":
#         return [base_style, base_style, active_style]
#     # elif pathname == "/modeling/":
#     #     return [base_style, base_style, base_style, active_style]
#     else:
#         return [base_style, base_style, base_style]

# if __name__ == "__main__":
#     app.run(port=8124, host='0.0.0.0', debug=True)
#     # app.run(port=8124, host='0.0.0.0')


import dash
from dash import Dash, html, dcc, callback, Input, Output, State
import os
import dash_bootstrap_components as dbc
import base64
import re
import warnings
warnings.filterwarnings('ignore')

app = Dash(__name__,
           suppress_callback_exceptions=True,
           external_stylesheets=[dbc.themes.BOOTSTRAP],
           use_pages=True)

# Загружаем изображения для навигации
analytic_image = base64.b64encode(open('images/analytic.png', 'rb').read())
predictive_image = base64.b64encode(open('images/predictive.png', 'rb').read())
learning_image = base64.b64encode(open('images/learning.png', 'rb').read())
help_image = base64.b64encode(open('images/help.png', 'rb').read())

# Функция для чтения README.md с поддержкой изображений
def read_readme_file():
    try:
        path = 'INSTRUCTION.md'
        if os.path.exists(path):
            readme_path = path
            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()
            return content, os.path.abspath(path)
        
        # Если файл не найден, возвращаем сообщение об ошибке
        return "# Файл README.md не найден\n\nПожалуйста, убедитесь, что файл README.md существует в корневой директории проекта.", None
    
    except Exception as e:
        return f"# Ошибка при чтении файла\n\nНе удалось загрузить инструкцию: {str(e)}", None

# Функция для преобразования изображений в base64
def convert_images_to_base64(markdown_content, readme_dir):
    """
    Преобразует относительные пути к изображениям в base64
    """
    if not readme_dir:
        return markdown_content
    
    # Регулярное выражение для поиска изображений в Markdown
    pattern = r'!\[(.*?)\]\((.*?)\)'
    
    def replace_image(match):
        alt_text = match.group(1)
        image_path = match.group(2)
        
        # Если это уже data URL или абсолютный URL, оставляем как есть
        if image_path.startswith('data:') or image_path.startswith('http://') or image_path.startswith('https://'):
            return match.group(0)
        
        try:
            # Полный путь к изображению
            full_image_path = os.path.join(readme_dir, image_path)
            
            # Проверяем существование файла
            if os.path.exists(full_image_path):
                # Определяем MIME тип
                ext = os.path.splitext(full_image_path)[1].lower()
                mime_types = {
                    '.png': 'image/png',
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg',
                    '.gif': 'image/gif',
                    '.svg': 'image/svg+xml',
                    '.webp': 'image/webp'
                }
                mime_type = mime_types.get(ext, 'image/png')
                
                # Читаем и кодируем изображение
                with open(full_image_path, 'rb') as img_file:
                    encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
                
                # Создаем data URL
                data_url = f"data:{mime_type};base64,{encoded_image}"
                return f'![{alt_text}]({data_url})'
            else:
                # Если файл не найден, возвращаем оригинальную ссылку
                return f'![{alt_text}]({image_path})'
                
        except Exception as e:
            print(f"Ошибка при обработке изображения {image_path}: {e}")
            return match.group(0)
    
    # Заменяем все найденные изображения
    return re.sub(pattern, replace_image, markdown_content)

# Функция для обработки Markdown и преобразования изображений
def process_markdown_with_images(markdown_content, readme_path):
    if readme_path:
        readme_dir = os.path.dirname(readme_path)
        return convert_images_to_base64(markdown_content, readme_dir)
    return markdown_content

# Модальное окно с инструкцией из README.md
instruction_modal = dbc.Modal(
    [
        dbc.ModalHeader(
            dbc.ModalTitle(
                html.Div([
                    html.I(className="fas fa-book me-2"),
                    "Инструкция по использованию сервиса"
                ])
            ),
            close_button=True
        ),
        dbc.ModalBody(
            html.Div(
                id="readme-content",
                style={
                    'maxHeight': '70vh',
                    'overflowY': 'auto',
                    'padding': '10px'
                }
            )
        ),
        dbc.ModalFooter(
            dbc.Button(
                "Закрыть",
                id="close-instruction",
                color="primary",
                className="ms-auto"
            )
        ),
    ],
    id="instruction-modal",
    size="xl",
    is_open=False,
    scrollable=True,
)

app.layout = html.Div(
    [
        dcc.Location(id='url', refresh=False),
        html.Header(
            children=[
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Link(
                                html.Img(
                                    src='data:image/png;base64,{}'.format(analytic_image.decode()),
                                    alt="Аналитика",
                                    id="analytic-img",
                                    style={
                                        "marginLeft": 30,
                                        "marginTop": 15,
                                        "width": 155,
                                        "height": 45,
                                    },
                                ),
                                href="/",
                            ), width="auto"
                        ),
                        dbc.Col(
                            dcc.Link(
                                html.Img(
                                    src='data:image/png;base64,{}'.format(predictive_image.decode()),
                                    alt="Прогнозирование",
                                    id="predictive-img",
                                    style={
                                        "marginLeft": 0,
                                        "marginTop": 15,
                                        "width": 155,
                                        "height": 45,
                                    },
                                ),
                                href="/predictive/",
                            ), width="auto"
                        ),
                        dbc.Col(
                            dcc.Link(
                                html.Img(
                                    src='data:image/png;base64,{}'.format(learning_image.decode()),
                                    alt="Обучение моделей",
                                    id="learning-img",
                                    style={
                                        "marginLeft": 0,
                                        "marginTop": 15,
                                        "width": 155,
                                        "height": 45,
                                    },
                                ),
                                href="/learning/",
                            ), width="auto"
                        ),
                        dbc.Col(
                            html.H1(
                                'Сервис аналитики и прогнозирования',
                                style={
                                    'fontWeight': 'bold',
                                    "height": 25,
                                    'textAlign': 'center',
                                    'color': '#2c3e50',
                                    'marginTop': 10
                                }
                            ),
                            width=True
                        ),
                        dbc.Col(
                            html.Div(
                                html.Img(
                                    src='data:image/png;base64,{}'.format(help_image.decode()),
                                    alt="Инструкция",
                                    id="help-img",
                                    style={
                                        "marginLeft": 0,
                                        "marginTop": 15,
                                        "width": 45,
                                        "height": 45,
                                        "cursor": "pointer",
                                        "transition": "all 0.3s ease",
                                        "borderRadius": "5px",
                                    },
                                ),
                                title="Открыть инструкцию",
                            ),
                            width="auto",
                            style={"marginRight": "30px"}
                        )
                    ],
                    className="align-items-center"
                )
            ],
        ),
        dash.page_container,
        instruction_modal
    ]
)

# Callback для обновления стилей активной вкладки
@callback(
    [Output("analytic-img", "style"),
     Output("predictive-img", "style"),
     Output("learning-img", "style")],
    [Input("url", "pathname")]
)
def update_active_tab(pathname):
    base_style = {
        "marginLeft": 0,
        "marginTop": 15,
        "width": 155,
        "height": 45,
        "transition": "all 0.3s ease"
    }
    
    active_style = {
        **base_style,
        "border": "3px solid #007bff",
        "borderRadius": "8px",
        "padding": "2px",
        "boxShadow": "0 4px 8px rgba(0,123,255,0.3)"
    }
    
    if pathname == "/":
        return [active_style, base_style, base_style]
    elif pathname == "/predictive/":
        return [base_style, active_style, base_style]
    elif pathname == "/learning/":
        return [base_style, base_style, active_style]
    else:
        return [base_style, base_style, base_style]

# Callback для открытия/закрытия модального окна с инструкцией
@callback(
    Output("instruction-modal", "is_open"),
    [Input("help-img", "n_clicks"),
     Input("close-instruction", "n_clicks")],
    [State("instruction-modal", "is_open")],
)
def toggle_instruction_modal(help_clicks, close_clicks, is_open):
    ctx = dash.callback_context
    if not ctx.triggered:
        return False
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == "help-img" or button_id == "close-instruction":
            return not is_open
    return is_open

# Callback для загрузки и отображения содержимого README.md с изображениями
@callback(
    Output("readme-content", "children"),
    [Input("help-img", "n_clicks")]
)
def load_readme_content(n_clicks):
    if n_clicks is None:
        return html.Div("Загрузка инструкции...")
    
    # Читаем содержимое README.md
    markdown_content, readme_path = read_readme_file()
    
    # Обрабатываем Markdown и преобразуем изображения
    processed_markdown = process_markdown_with_images(markdown_content, readme_path)
    
    try:
        # Используем dcc.Markdown для отображения
        return html.Div(
            dcc.Markdown(
                processed_markdown,
                dangerously_allow_html=False,
                style={
                    'fontFamily': 'Arial, sans-serif',
                    'lineHeight': '1.6',
                    'color': '#333'
                }
            ),
            style={
                'padding': '20px',
                'backgroundColor': '#f8f9fa',
                'borderRadius': '5px'
            }
        )
    
    except Exception as e:
        # Если произошла ошибка при преобразовании, показываем raw текст
        return html.Div([
            html.H4("Ошибка при загрузке инструкции"),
            html.P("Содержимое файла README.md:"),
            html.Pre(
                markdown_content,
                style={
                    'whiteSpace': 'pre-wrap',
                    'wordWrap': 'break-word',
                    'backgroundColor': '#f8f9fa',
                    'padding': '15px',
                    'borderRadius': '5px',
                    'maxHeight': '400px',
                    'overflowY': 'auto'
                }
            )
        ])

if __name__ == "__main__":
    app.run(port=8124, host='0.0.0.0', debug=True)
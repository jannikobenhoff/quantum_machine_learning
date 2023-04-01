import base64
import io
import math
import dash
import dash_bootstrap_components as dbc
import numpy as np
from dash import dcc, html
import plotly.graph_objs as go

import wolframalpha
from wolframclient.evaluation import WolframLanguageSession

from flask import Flask, Response, request
import cv2
from PIL import Image

from dash.dependencies import Output, Input, State, MATCH, ALL

from quantum_transfer import quantum_predict
from scanning import scan_process
from transfer import predict
from webcam import *
from styles import *

server = Flask(__name__)
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP,
                                                               dbc.icons.BOOTSTRAP,
                                                               dbc.icons.FONT_AWESOME])


@server.route('/video_feed', methods=['GET', 'POST'])
def video_feed():
    if request.method == "POST":
        print("POST")
    if request.method == "GET":
        print("GET")

    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


button_group = html.Div([
    html.Div(
        [
            html.Button(
                id={"type": "operation", "index": 0},
                children=html.Div(
                    [html.H5("Calculate", style={'font-size': 15, 'padding-left': 10, "padding-bottom": 10}),

                     html.Img(src="/assets/icons/calc.png",
                              height=50,
                              style={"filter": "brightness(1) invert(0)"}
                              ),
                     ]), style=button_style

            ),
            html.Button(id={"type": "operation", "index": 1},

                        children=html.Div(
                            [html.H5("Plot", style={'font-size': 15, 'padding-left': 10, "padding-bottom": 10}),

                             html.Img(src="/assets/icons/plot.png",
                                      height=50,
                                      style={"filter": "brightness(1) invert(0)"}
                                      ),
                             ]), style=button_style
                        ),
            html.Button(id={"type": "operation", "index": 2},

                        children=html.Div([
                            html.H5("Equation", style={'font-size': 15, 'padding-left': 10, "padding-bottom": 10}),
                            html.Img(src="/assets/icons/function.png",
                                     height=50,
                                     style={"filter": "brightness(1) invert(0)"}
                                     ),
                        ]), style=button_style
                        ),
            html.Button(id={"type": "operation", "index": 3},

                        children=html.Div([
                            html.H5("Quantum Machine Learning", style={'font-size': 12, 'padding-left': 10, "padding-bottom": 10}),
                            html.Img(src="/assets/icons/quantum-computing.png",
                                     height=50,
                                     style={"filter": "brightness(1) invert(0)"}
                                     ),
                        ]), style=button_style
                        ),
        ], style={"margin-top": 25, "margin-left": 25, "margin-right": 25, "display": "flex", "flex-direction": "row",
                  "width": "100%",
                  "justify-content": "center", "align-items": "center"}
    ),
], style={"display": "flex", "flex-direction": "row", "width": "100%",
          "justify-content": "center", "align-items": "center"})

webcam_group = dbc.Row(style={"margin-top": 25},
                 children=[
                     dbc.Col(width=3, children=html.Div([
                         html.Button(
                             children=dcc.Upload(id={"type": "input", "index": 1}, children=[html.Div(
                                 [html.H5('Select Files'), html.Img(src="/assets/icons/upload.png",
                                                                    height=50,
                                                                    style={
                                                                        "filter": "brightness(1) invert(0)",
                                                                        "margin-top": 5}
                                                                    ),
                                  ],
                             )]),
                             style=button_style)
                     ], style={"height": "100%", "justify-content": "center", "display": "flex",
                               "align-items": "center", "flex-direction": "column", })),
                     dbc.Col(width=6, id="screen"),
                     dbc.Col(width=3, children=html.Div([
                         html.Button(id={"type": "input", "index": 0}, n_clicks=0, style=button_style,
                                     children=html.Div([
                                         html.H5('Take Photo'),
                                         html.Img(src="/assets/icons/webcam.png",
                                                  height=50,
                                                  style={"filter": "brightness(1) invert(0)",
                                                         "margin-top": 5}
                                                  ),
                                     ])),
                     ], style={"height": "100%", "justify-content": "center", "display": "flex",
                               "align-items": "center", "flex-direction": "column", }))
                 ])

webcam = html.Div([html.H5("Webcam", style={"color": " white", "margin-bottom": 5}),
                             html.Img(src="/video_feed",
                                      style={'height': '90%', 'width': '90%', "margin-left": 25,
                                             "margin-right": 25, "border-radius": "0.3em"})])

app.layout = html.Div(
    [
        html.Div(style={"height": "10%", 'background-color': black, "border-bottom-left-radius": "2em",
                        "border-bottom-right-radius": "2em"},
                 children=dbc.Row([
                     dbc.Col([
                         html.H1("MATHTECTION",
                                 style={'margin-top': 25, 'margin-bottom': 0, "color": "white"}),
                         html.Span("Deep Learning Algorithm evaluating Mathematical Formulas", style={"color": "white"})
                     ], width=12)], justify="left"), ),

        button_group,
        webcam_group,
        html.Dialog(children=html.P("hallo")),
        dbc.Row([html.Hr(style={"margin-top":25})]),
        html.Div(id={"type": "pics", "index": 0}, n_clicks=0,
                 style={"justify-content": "center", "display": "flex",
                               "align-items": "center", "flex-direction": "row",}),
        html.Div(id={"type": "pics", "index": 1}, n_clicks=0),
        html.Div(id="pics", n_clicks=0),
        html.Div(id="pics2", n_clicks=0, style={"justify-content": "center", "display": "flex",
                               "align-items": "center", "flex-direction": "row", "margin-top": 25}),
        dbc.Row([html.Hr(style={"margin-top": 25})]),
        html.Div(id="prediction", style={"background-color": black}),
        # html.Img(id="img0", src=Image.open("screen.jpg").convert('L'), n_clicks=0),
        dcc.Store(id='store'),
        html.Div(id="delete"),
        html.Div(id="ready"),
    ], style={'background-color': black, "height": "100vh", "width": "100%", "overflow": "scroll"})


@app.callback(
    Output("prediction", "children"),
    Input("ready", "n_clicks"),
    State("store", 'data')
)
def ready(click, data):
    client = wolframalpha.Client(key)
    res = client.query("x + 5 * 4 = x * 9")
    for i in res.results:
        print(i.text)
    if click is not None and click != 0:
        img_list = []
        zahlenList = scan_process("screen.jpg")
        for i, zahl in enumerate(zahlenList):
            if i not in del_img:
                img_list.append(zahl)
        if data == 3:
            '''Quantum Machine Learning'''
            prediction = quantum_predict(img_list)
            return html.Div(
                [html.H2("Quantum Detection:"), html.Br(), html.H2(prediction, style={"color": "white"})])

        else:
            prediction = predict(img_list)
            if data == 0:
                '''Calculate'''
                print("Calculate")
                ev = eval(prediction)
                return html.Div([html.H2("Result:"), html.Br(), html.H2(prediction+" = "+str(ev), style={"color": "white"})])

            elif data == 1:
                '''Plot'''
                print("Plot")
                print(prediction)
                a = np.linspace(-50, 50, 1000)
                n = np.linspace(-50, 50, 1000)
                #ev = eval("math.sin(a)")
                fig = go.Figure(data=[go.Scatter(x=a, y=np.sin(a), line=dict(color=yellow))])
                fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')

                return [html.H2("f(a) = " + "sin(a)", style={"color": "white"}),
                        dcc.Graph(figure=fig, style={"background-color": black})]

            elif data == 2:
                '''Equation'''
                print("Equation")
                client = wolframalpha.Client(key)
                res = client.query(prediction)
                # Includes only text from the response
                answer = [prediction, html.Br()]
                for i in res.results:
                    answer.append(i.text)
                    answer.append(html.Br())

                return html.H2(answer, style={"color": "white"})



def parse_contents(contents, filename):
    data = contents.encode("utf8").split(b";base64,")[1]

    zahlenList = scan_process(io.BytesIO(base64.b64decode(data)), save=True)
    print(len(zahlenList))
    img_list = []
    for i, zahl in enumerate(zahlenList):
        img_list.append(html.Img(id={"type": "img", "index": i}, n_clicks=0, src=Image.fromarray(zahl.imagearray),
                                 style={"margin-right": "10px", "height": "5%",
                                        "width": "5%", "border": "2px white solid"})
                        )
    img_list.append(
        html.Button(id="delete",
                    children=html.Div([html.H5('Reselect'),
                                       html.Img(src="/assets/icons/reload.png",
                                                height=50,
                                                style={"filter": "brightness(1) invert(0)"}
                                                ),
                                       ], style={"flex-direction": "column", "justify-content": "center",
                                                 "align-items": "center"}),
                    style=select_button_style
                    ))
    return img_list


@app.callback(
    Output({'type': 'pics', "index": 0}, 'children'),
    Output('screen', 'children'),
    Input({'type': 'input', 'index': 0}, 'n_clicks'),
    Input({'type': 'input', 'index': 1}, 'contents'),
    State({'type': 'input', 'index': 1}, 'filename'),
)
def update_input(click, content, filename):
    try:
        index = dash.ctx.triggered_id["index"]
    except TypeError:
        return html.Div(), \
               webcam
    print(dash.ctx.triggered_id)
    print(click)
    if index == 0:
        '''Webcam'''
        if click != 0 and click != None:
            img_list = []
            print((click-1) % 2)
            if (click-1) % 2:
                return img_list, webcam
            else:
                zahlenList = VideoCamera().take_screen()

                if zahlenList != None and zahlenList[0] != None:
                    print("--plotting--")
                    for i, zahl in enumerate(zahlenList):
                        print("img{}".format(i))
                        img_list.append(html.Img(id={"type": "img", "index": i},
                                                 n_clicks=0, src=Image.fromarray(zahl.imagearray),
                                                 style={"margin-right": "10px", "height": "5%",
                                                        "width": "5%", "border": "2px white solid"})
                                        )
                    img_list.append(
                        html.Button(id="delete",
                                    children=html.Div([html.H5('Reselect'),
                                                       html.Img(src="/assets/icons/reload.png",
                                                                height=50,
                                                                style={"filter": "brightness(1) invert(0)"}
                                                                ),
                                                       ],
                                                      style={"flex-direction": "column", "justify-content": "center",
                                                             "align-items": "center"}),
                                     style= select_button_style #{"height": "100px", "border-radius": "1.5rem",
                                    #        "background-color": "#C2654E", "border": "none", "margin-left": 30}
                                    ))
                    print("HIER")
                    return img_list, html.Div([html.H5("Webcam photo", style={"color": "white","margin-bottom":5}),
                                               html.Img(src=Image.open("screen.jpg"), style={'height': '30%', 'width': '30%', "border": "2px white solid", "margin-top": 5})])
        else:
            return html.Div(), webcam
    else:
        '''Uploaded Photo'''
        if content is not None:
            return parse_contents(content, filename), html.Div([
                html.H5(filename, style={"color":"white", "margin-bottom":5}),
                html.Img(src=Image.open("__files/{}".format(filename)),
                         style={"height": "30%", "width": "30%", "border": "2px white solid", "margin-top": 5})])


del_img = []


@app.callback(
    Output({'type': 'img', 'index': MATCH}, 'style'),
    Input({'type': 'img', 'index': MATCH}, 'n_clicks'),
    State({'type': 'img', 'index': MATCH}, 'id'))
def select_images(click, index):
    if click != 0 and click != None:
        if index["index"] in del_img:
            del_img.remove(index["index"])
        else:
            del_img.append(index["index"])
        print(del_img)
        if click % 2:
            return {"margin-right": "10px", "height": "5%", "width": "5%", "border": "4px red solid"}
        else:
            return {"margin-right": "10px", "height": "5%", "width": "5%", "border": "2px white solid"}
    else:
        return {"margin-right": "10px", "height": "5%", "width": "5%", "border": "2px white solid"}


@app.callback(
    Output(component_id='pics2', component_property='children'),
    Input("delete", "n_clicks")
)
def delete_images(click):
    if click != 0 and click != None:
        img_list = []
        zahlenList = scan_process("screen.jpg")
        for i, zahl in enumerate(zahlenList):
            if i not in del_img:
                img_list.append(
                    html.Img(id={"type": "img2", "index": i}, n_clicks=0, src=Image.fromarray(zahl.imagearray),
                             style={"margin-right": "10px", "height": "5%",
                                    "width": "5%", "border": "2px white solid"})
                )
        img_list.append(
            html.Button(id="ready",
                        children=html.Div([html.H5('   Ready!   '),
                                           html.Img(src="/assets/icons/neural.png",
                                                    height=50,
                                                    style={"filter": "brightness(1) invert(0)"}
                                                    ),
                                           ], style={"flex-direction": "column", "justify-content": "center",
                                                     "align-items": "center"}),
                        style=ready_button_style
        ))
        return img_list


@app.callback(
    Output({"type": "operation", "index": ALL}, "style"),
    Output("store", "data"),
    Input({"type": "operation", "index": ALL}, "n_clicks"),
    State("store", 'data')
)
def operation_select(click, data):
    try:
        index = dash.ctx.triggered_id["index"]
    except TypeError:
        return [operation_button_style_clicked, operation_button_style, operation_button_style, operation_button_style], 0

    match index:
        case 0:
            return [operation_button_style_clicked, operation_button_style, operation_button_style, operation_button_style], 0
        case 1:
            return [operation_button_style, operation_button_style_clicked, operation_button_style, operation_button_style], 1
        case 2:
            return [operation_button_style, operation_button_style, operation_button_style_clicked, operation_button_style], 2
        case 3:
            return [operation_button_style, operation_button_style, operation_button_style,operation_button_style_clicked], 3
        case _:
            return [operation_button_style_clicked, operation_button_style, operation_button_style, operation_button_style], 0


if __name__ == '__main__':
    OPENCV_AVFOUNDATION_SKIP_AUTH = 1
    app.run_server(debug=True)


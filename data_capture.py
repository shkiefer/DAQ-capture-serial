import numpy as np
import pandas as pd

import dash
from dash.dependencies import Input, Output, State, MATCH, ALL
import dash_core_components as dcc
import dash_html_components as html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_table
import dash_table.FormatTemplate as FormatTemplate
from dash_table.Format import Format, Scheme, Sign, Symbol

import plotly.graph_objs as go

from threading import Thread
import queue
import serial
import serial.tools.list_ports

import time
from pathlib import Path
import json
import sqlite3
from datetime import datetime

# globals... yuk
FILE_DIR = ''
APP_ID = 'serial_data'
Q = queue.Queue()
SERIAL_THREAD = None

class SerialThread(Thread):

    def __init__(self, port, baud=115200):
        super().__init__()
        self.port = port
        self._isRunning = True
        self.ser_obj = serial.Serial(port=port,
                                     baudrate=baud,
                                     parity=serial.PARITY_NONE,
                                     stopbits=serial.STOPBITS_ONE,
                                     timeout=None)

    def run(self):
        while self._isRunning:
            try:
                while self.ser_obj.in_waiting > 2:
                    try:
                        line = self.ser_obj.readline()
                        split_line = line.strip().decode("utf-8")
                        Q.put(split_line)
                    except:
                        continue
            except:
                continue


    def stop(self):
        self._isRunning = False
        time.sleep(0.25)
        self.ser_obj.close()
        return None


# layout
layout = dbc.Container([
    dbc.Row(
        dbc.Col([
            dcc.Store(id=f'{APP_ID}_store'),
            dcc.Interval(id=f'{APP_ID}_interval',
                         interval=2000,
                         n_intervals=0,
                         disabled=True),
            html.H4('Serial Data Plotter'),
            html.P('This tests plotting data from serial (arduino) using a background thread to collect the data and send it to a queue.  '
                   'Data is retrieved from the queue and stored in the browser as well as written to a file')
        ])
    ),
    dbc.Row([
        dbc.Col(
            dbc.FormGroup([
                dbc.Button('COM Ports (refresh)', id=f'{APP_ID}_com_button'),
                dcc.Dropdown(id=f'{APP_ID}_com_dropdown',
                             placeholder='Select COM port',
                             options=[],
                             multi=False),
                dbc.Textarea(id=f'{APP_ID}_com_desc_label', disabled=True )
            ]),
            width=4
        ),
        dbc.Col(
            dbc.FormGroup([
                dbc.Label('Headers'),
                dbc.Button('Initialize Headers', id=f'{APP_ID}_init_header_button', block=True),
                dash_table.DataTable(
                    id=f'{APP_ID}_header_dt',
                    columns=[
                        {"name": 'Position', "id": 'pos', "type": 'numeric', 'editable': False},
                        {"name": 'Name', "id": 'name', "type": 'text'},
                        {"name": 'Format', "id": 'fmt', "type": 'text', "presentation": 'dropdown'}
                             ],
                    data=[{}],
                    editable=True,
                    row_deletable=False,
                    dropdown={
                        'fmt': {
                            'options': [
                                {'label': i, 'value': i} for i in ['text', 'real', 'integer']
                            ],
                        },
                    }
                ),
            ]),
            width=4
        ),
    ]),
    dbc.Row(
        dbc.Col([
                dbc.Toast(
                    children=[],
                    id=f'{APP_ID}_header_toast',
                    header="Initialize Headers",
                    icon="danger",
                    dismissable=True,
                    is_open=False
                ),
        ],
            width="auto"
        ),
    ),
    dbc.Row([
        dbc.Col(
            dbc.FormGroup([
                dbc.Label('Filename'),
                dbc.Input(placeholder='filename',
                          id=f'{APP_ID}_filename_input',
                          type='text',
                          value=f'data/my_data_{datetime.now().strftime("%m.%d.%Y.%H.%M.%S")}.db')
            ])
        )
    ]),
    dbc.ButtonGroup([
        dbc.Button('Start', id=f'{APP_ID}_start_button', n_clicks=0, disabled=True, size='lg', color='secondary'),
        dbc.Button('Stop', id=f'{APP_ID}_stop_button', n_clicks=0, disabled=True, size='lg', color='secondary'),
        dbc.Button('Clear', id=f'{APP_ID}_clear_button', n_clicks=0, disabled=True, size='lg'),
        dbc.Button('Download Data', id=f'{APP_ID}_download_button', n_clicks=0, disabled=True, size='lg'),
    ],
        className='mt-2 mb-2'
    ),
    html.Br(),
    dbc.ButtonGroup([
        dbc.Button("Add Figure / Readout", id=f'{APP_ID}_add_figure_button'),
    ]),
    # todo add datatable for readouts
    # todo convert figure datatable to div with 2 select boxes (X, y (multi=true))
    dash_table.DataTable(
        id=f'{APP_ID}_figure_dt',
        columns=[
            {"name": 'Name', "id": 'name', 'type': 'text'},
            {"name": 'X data', "id": 'x_data', 'type': 'text', 'presentation': 'dropdown'},
            {"name": 'Y data', "id": 'y_data', 'type': 'text', 'presentation': 'dropdown'},
            {"name": 'Width (px)', "id": 'width', 'type': 'numeric'},
        ],
        data=[{}],
        row_deletable=True,
        editable=True,
    ),
    dbc.Row([
        dbc.Col(
            dbc.FormGroup([
                dbc.Label('X axis'),
                dcc.Dropdown(id=f'{APP_ID}_x_dropdown',
                             placeholder='Select X axis',
                             options=[],
                             multi=False),
            ]),
            width=4
        ),
        dbc.Col(
            dbc.FormGroup([
                dbc.Label('Y axis'),
                dcc.Dropdown(id=f'{APP_ID}_y_dropdown',
                             placeholder='Select Y axis',
                             options=[],
                             multi=True),
            ]),
            width=4
        ),
    ]),

    dbc.CardGroup(
    id=f'{APP_ID}_fig_card_group'),
    dbc.Row(
        dbc.Col(id=f'{APP_ID}_current_div'),
    ),
])


def add_dash(app):

    @app.callback(
        [Output(f'{APP_ID}_header_dt', 'data'),
         Output(f'{APP_ID}_header_toast', 'children'),
         Output(f'{APP_ID}_header_toast', 'is_open'),
         ],
        [Input(f'{APP_ID}_init_header_button', 'n_clicks')],
        [State(f'{APP_ID}_com_dropdown', 'value')]
    )
    def serial_data_init_header(n_clicks, com):
        if n_clicks is None or com is None:
            raise PreventUpdate

        baud = 115200
        try:
            ser_obj = serial.Serial(port=com,
                                    baudrate=baud,
                                    parity=serial.PARITY_NONE,
                                    stopbits=serial.STOPBITS_ONE,
                                    timeout=10)
            split_line = '_'
            while split_line[0] != '{':
                line = ser_obj.readline()
                split_line = line.strip().decode("utf-8")
                print(split_line)

            split_line = line.strip().decode("utf-8")
            jdic = json.loads(split_line)
            data = [{'pos': i, 'name': k} for i, k in enumerate(jdic.keys())]
            ser_obj.close()
            return data, '', False
        except Exception as e:
            return [{}], html.P(str(e)), True
        return data, '', False


    @app.callback(
        Output(f'{APP_ID}_com_dropdown', 'options'),
        [Input(f'{APP_ID}_com_button', 'n_clicks')]
    )
    def serial_data_refresh_com_ports(n_clicks):
        if n_clicks is None:
            raise PreventUpdate
        ports = [{'label': comport.device, 'value': comport.device} for comport in serial.tools.list_ports.comports()]
        return ports


    @app.callback(
        Output(f'{APP_ID}_com_desc_label', 'value'),
        [Input(f'{APP_ID}_com_dropdown', 'value')]
    )
    def serial_data_com_desc(com):
        if com is None:
            raise PreventUpdate
        ports = [comport.device for comport in serial.tools.list_ports.comports()]
        idx = ports.index(com)
        descs = [comport.description for comport in serial.tools.list_ports.comports()]
        return descs[idx]

    @app.callback(
        [
            Output(f'{APP_ID}_figure_dt', 'data'),
            Output(f'{APP_ID}_figure_dt', 'dropdown'),
        ],
        [
            Input(f'{APP_ID}_add_figure_button', 'n_clicks'),
            Input(f'{APP_ID}_header_dt', 'data'),
        ],
        [
            State(f'{APP_ID}_figure_dt', 'data')
        ]
    )
    def serial_data_figure_dt(n_clicks, header_data, data):
        ctx = dash.callback_context
        if not ctx.triggered or header_data is None:
            raise PreventUpdate

        df_header = pd.DataFrame(header_data)
        df_header = df_header.dropna(axis=0, how='any')
        if df_header.empty:
            return [{}], {}

        # add row (button press)
        if ctx.triggered[0]['prop_id'].split('.')[0] == f'{APP_ID}_add_figure_button':

            data.append(
                {
                    'name': f'{len(data) + 1:d}',
                    'x_data': 'index',
                    'y_data': df_header['name'].values[-1],
                    'width': 400
                 }
            )
            dropdown = {
                'x_data': {
                    'options':
                        [{'label': 'index', 'value': 'index'}] +
                        [
                            {'label': name, 'value': name} for name in df_header['name']
                        ],
                },
                'y_data': {
                    'options':
                        [{'label': 'index', 'value': 'index'}] +
                        [
                            {'label': name, 'value': name} for name in df_header['name']
                        ],
                },
            }

            return data, dropdown

        # header data changed
        if ctx.triggered[0]['prop_id'].split('.')[0] == f'{APP_ID}_header_dt':
            data = [
                {
                    'name': f'{i + 1:d}',
                    'x_data': 'index',
                    'y_data': name,
                    'width': 400
                }
                for i, name in enumerate(df_header['name'])
            ]

            dropdown = {
                'x_data': {
                    'options':
                        [{'label': 'index', 'value': 'index'}] +
                        [
                            {'label': name, 'value': name} for name in df_header['name']
                        ],
                },
                'y_data': {
                    'options':
                        [{'label': 'index', 'value': 'index'}] +
                        [
                            {'label': name, 'value': name} for name in df_header['name']
                        ],
                },
            }

            return data, dropdown
    # todo use pattern matching callbacks to create multiple graph objects &


    @app.callback(
        [
            Output(f'{APP_ID}_interval', 'disabled'),
            Output(f'{APP_ID}_start_button', 'disabled'),
            Output(f'{APP_ID}_start_button', 'color'),
            Output(f'{APP_ID}_stop_button', 'disabled'),
            Output(f'{APP_ID}_stop_button', 'color'),
            Output(f'{APP_ID}_clear_button', 'disabled'),
            Output(f'{APP_ID}_clear_button', 'color'),
            Output(f'{APP_ID}_filename_input', 'disabled'),
            Output(f'{APP_ID}_filename_input', 'value'),
            Output(f'{APP_ID}_header_dt', 'editable'),
            Output(f'{APP_ID}_store', 'clear_data'),
            Output(f'{APP_ID}_x_dropdown', 'options'),
            Output(f'{APP_ID}_y_dropdown', 'options'),
         ],
        [
            Input(f'{APP_ID}_start_button', 'n_clicks'),
            Input(f'{APP_ID}_stop_button', 'n_clicks'),
            Input(f'{APP_ID}_clear_button', 'n_clicks'),
            Input(f'{APP_ID}_header_dt', 'data'),
         ],
        [
            State(f'{APP_ID}_com_dropdown', 'value'),
            State(f'{APP_ID}_filename_input', 'value'),
            State(f'{APP_ID}_header_dt', 'data')
         ]
    )
    def serial_data_start_stop(n_start, n_stop, n_clear, hdr_data, port, filename, data_header):
        global SERIAL_THREAD
        global Q

        ctx = dash.callback_context
        if any([n_start is None, n_stop is None, port is None, hdr_data is None, n_clear is None]):
            raise PreventUpdate

        df_hdr = pd.DataFrame(data_header).sort_values('pos')
        df_hdr['name'] = df_hdr['name'].fillna(df_hdr['pos'].astype(str))
        headers = df_hdr['name'].tolist()
        options = [{'label': c, 'value': c} for c in headers]

        trig = ctx.triggered[0]['prop_id'].split('.')[0]
        if trig == f'{APP_ID}_header_dt':
            if len(data_header[0].keys()) == 3 and ~df_hdr.isnull().values.any():
                return True, False, 'success', True, 'secondary', True, 'secondary', False, filename, True, False, options, options
            else:
                return True, True, 'secondary', True, 'secondary', True, 'secondary', False, filename, True, False, options, options


        if trig == f'{APP_ID}_start_button':
            print(f'starting: {filename}')
            if filename is None or filename == '':
                filename = f'data/my_data_{datetime.now().strftime("%m.%d.%Y.%H.%M.%S")}.db'
            if (Path(FILE_DIR) / filename).exists():
                clear = False
            else:
                clear = True
            SERIAL_THREAD = SerialThread(port, baud=115200)
            SERIAL_THREAD.start()
            return False, True, 'secondary', False, 'danger', True, 'secondary', True, filename, False, clear, options, options

        if trig == f'{APP_ID}_stop_button':
            print('stopping')
            SERIAL_THREAD.stop()
            with Q.mutex:
                Q.queue.clear()
            return True, False, 'success', True, 'secondary', False, 'warning', False, filename, True, False, options, options

        if trig == f'{APP_ID}_clear_button':
            print('clearing')
            filename = f'data/my_data_{datetime.now().strftime("%m.%d.%Y.%H.%M.%S")}.db'
            return True, False, 'success', True, 'secondary', True, 'secondary', False, filename, True, True, options, options


    @app.callback(
        Output(f'{APP_ID}_store', 'data'),
        [Input(f'{APP_ID}_interval', 'n_intervals')],
        [State(f'{APP_ID}_interval', 'disabled'),
         State(f'{APP_ID}_store', 'data'),
         State(f'{APP_ID}_filename_input', 'value'),
         State(f'{APP_ID}_header_dt', 'data')
         ]
    )
    def serial_data_update_store(n_intervals, disabled, data, filename, data_header):
        global Q
        # get data from queue
        if disabled is not None and not disabled:
            new_data = []
            while not Q.empty():
                new_data_dic = json.loads(Q.get())
                new_data.append(tuple((new_data_dic[c["name"]] for c in data_header if c["name"] in new_data_dic.keys())))

            conn = sqlite3.connect(FILE_DIR + filename)
            c = conn.cursor()

            c.execute(''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='my_data' ''')
            if c.fetchone()[0] == 1:
                c.executemany(f'INSERT INTO my_data VALUES ({(",".join(["?"] * len(data_header)) )})', new_data)
                conn.commit()
                last_row_id = c.execute("SELECT COUNT() FROM my_data").fetchone()[0]
                conn.close()
            else:
                c.execute(
                    f'''CREATE TABLE my_data
                    (''' + ', '.join([f'{hdr["name"]} {hdr["fmt"]}' for hdr in data_header])
                    + ')'
                )
                c.executemany(f'INSERT INTO my_data VALUES ({(",".join(["?"] * len(data_header)) )})', new_data)
                conn.commit()
                last_row_id = c.execute("SELECT COUNT() FROM my_data").fetchone()[0]
                conn.close()
            return last_row_id


    @app.callback(
        Output(f'{APP_ID}_fig_card_group', 'children'),
        [
            Input(f'{APP_ID}_figure_dt', 'data'),
            Input(f'{APP_ID}_store', 'data'),
         ],
        [
            State(f'{APP_ID}_filename_input', 'value'),
            State(f'{APP_ID}_fig_card_group', 'children')
         ]
    )
    def serial_data_create_figures(fig_dt_data, last_row_id, filename, cards):
        ctx = dash.callback_context
        if any([v is None for v in [fig_dt_data, filename, last_row_id]]):
            raise PreventUpdate
        trig = ctx.triggered[0]['prop_id'].split('.')[0]

        if trig == f'{APP_ID}_store' and cards is not None:
            raise PreventUpdate

        df_fig = pd.DataFrame(fig_dt_data).dropna(axis=0, how='any')
        if df_fig.empty:
            return [dbc.Alert('Add plots & readouts y adding rows to the figure table', color='warning')]


        cards = []
        for p in df_fig.itertuples():
            w = p.width

            card = dbc.Card([
                dbc.CardHeader(f"{p.name}"),
                dbc.CardBody([
                    html.H4(
                        id={
                            'type': f'{APP_ID}_card_hdr',
                            'index': p.name
                        },
                        className="card-title"
                    ),
                    dcc.Graph(
                        id={
                            'type': f'{APP_ID}_card_graph',
                            'index': p.name
                        },
                    )
                ])

            ],
                style={"max-width": f"{w}px", "min-width": f"{w}px"}
            )
            cards.append(card)
        return cards


    @app.callback(
        [
            Output({'type': f'{APP_ID}_card_graph', 'index': ALL}, 'figure'),
            Output({'type': f'{APP_ID}_card_hdr', 'index': ALL}, 'children'),
        ],
        [
            Input(f'{APP_ID}_store', 'modified_timestamp'),
         ],
        [
            State(f'{APP_ID}_store', 'data'),
            State(f'{APP_ID}_filename_input', 'value'),
            State(f'{APP_ID}_figure_dt', 'data'),
         ]
    )
    def serial_data_extend_figures(ts, data, filename, fig_dt_data):
        if any([v is None for v in [ts, data]]):
            raise PreventUpdate

        conn = sqlite3.connect(FILE_DIR + filename)
        cur = conn.cursor()
        n_estimate = cur.execute("SELECT COUNT() FROM my_data").fetchone()[0]
        n_int = n_estimate // 10000 + 1
        query = f'SELECT * FROM my_data WHERE ROWID % {n_int} = 0'
        df = pd.read_sql(query, conn)
        conn.close()

        df_fig = pd.DataFrame(fig_dt_data).dropna(axis=0, how='any')

        figs = []
        hdrs = []
        for fcb, hcb in zip(dash.callback_context.outputs_list[0], dash.callback_context.outputs_list[1]):
            if hcb['id']['index'] != fcb['id']['index']:
                print('not equal!', f'hcb: {hcb["id"]}, {fcb["id"]}')
            s_fig = df_fig.loc[df_fig['name'].astype(str) == fcb['id']['index']].iloc[0, :]
            x_data = s_fig['x_data']
            y_data = s_fig['y_data']
            if x_data == 'index':
                x = df.index
            else:
                x = df[x_data]
            if y_data == 'index':
                y = df.index
            else:
                y = df[y_data]

            fig = go.Figure()
            fig.update_xaxes(title=x_data)
            fig.update_yaxes(title=y_data)
            fig.update_layout(margin={'l': 20, 'r': 20, 't': 0, 'b': 20})
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    showlegend=False
                )
            )
            figs.append(fig)
            # hdr = dbc.Alert(
            #     [html.H3(f"X, {x_data}: {x.values[-1]:0.3f}"), html.H3(f"Y, {y_data}: {y.values[-1]:0.3f}")],
            #     color='primary'
            # )
            hdr = [
                dbc.ListGroup([
                    dbc.ListGroupItem(html.H3(f"{x_data}")),
                    dbc.ListGroupItem(html.H3(f"{x.values[-1]:0.3g}"), color='info'),
                ]),
                dbc.ListGroup([
                    dbc.ListGroupItem(html.H3(f"{y_data}")),
                    dbc.ListGroupItem(html.H3(f"{y.values[-1]:0.3g}"), color='info'),
                ]),
            ]
            hdrs.append(hdr)

        return figs, hdrs

    return app


if __name__ == '__main__':
    # app
    external_stylesheets = [
        dbc.themes.BOOTSTRAP,
    ]
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
    app.layout = layout
    app = add_dash(app)

    app.run_server(debug=True)
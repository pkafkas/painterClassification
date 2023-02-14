# -*- coding: utf-8 -*-
# We start with the import of standard ML librairies
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
# We add all Plotly and Dash necessary librairies
from dash import Dash, dcc, html, Input, Output
import dash_daq as daq
from dash.dependencies import Input, Output
#
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
#
import joblib
import validators
#
from dash_paint02_functions import conf_m, createRocCurve, pr_rec_curve, plot_pca, demo_predictor

feature_matrix = pd.read_csv('feature_matrix_demo.csv')
feature_matrix_pca = pd.read_csv('feature_matrix_demo_pca.csv')

app = Dash(__name__)

colors = {
    'background': '#E9F1FA',
    'text': '#393939'
}


models = {'Random Forest': RandomForestClassifier,
          'SVM Classifier': SVC,
          'KNN Classifier': KNeighborsClassifier}

app.layout = html.Div(style={'backgroundColor': colors['background']},children = [
    html.H1(
        children = "Paintings Classification",
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),

    # first row
    html.Div(children=[
        # first column
        html.Div(children=[
            html.P("Select model:"),
            dcc.Dropdown(
                id='dropdown',
                options=["SVM", "Random Forest", "k-NN"],
                value='Random Forest',
                clearable=False
            ),
            html.P("Select Test-train ratio:"),
            dcc.Slider(
                id='my-LED-display-slider-1',
                min=0,
                max=1,
                step=0.1,
                value=0.2
            ),
            daq.LEDDisplay(
                id='my-LED-display-1',
                label="Accuracy",
                value=0.0,
                color="#FF5E5E"
                #backgroundColor="#FF5E5E"
            ),
        ], style={'display': 'inline-block', 'vertical-align': 'top', 
                    'width': '25%','margin-left': '3vw', 'margin-top': '3vw'}),

        # second column
        html.Div(children=[
            dcc.Graph(id='indicator-graphic')      
        ], style={'display': 'inline-block', 'vertical-align': 'top', 
                    'width': '65%','margin-left': '3vw', 'margin-top': '3vw'}),
    ]),

    # second row
    html.Div(children=[
        # second column
        html.Div(children=[

            dcc.Graph(id='roc_curve') 

        ], style={'display': 'inline-block', 'vertical-align': 'top', 
                    'width': '30%','margin-left': '3vw', 'margin-top': '3vw'}),
        
        html.Div(children=[
            dcc.Graph(id='precision_recall')      
        ], style={'display': 'inline-block', 'vertical-align': 'top', 
                    'width': '34%','margin-left': '15vw', 'margin-top': '3vw'}),
    ]),

    # third tow

    html.Div(children=[
            html.Div(children=[

                dcc.Graph(id='pca_vis') 

        ], style={'display': 'inline-block', 'vertical-align': 'top', 
                    'width': '70%','margin-left': '3vw', 'margin-top': '3vw'}),
    ]),

    html.Div(
    [
        html.I("Please enter an image url. Press Enter and/or Tab to cancel the delay"),
        html.Br(),
        dcc.Input(id="in_url", type="url", placeholder="", style={'marginRight':'10px'}),
        html.Div(id="out_url", children=[]),
    ])


])



@app.callback(
    Output('my-LED-display-1', "value"),
    Output('indicator-graphic', 'figure'),
    Output('roc_curve', 'figure'),
    Output('precision_recall', 'figure'),
    Output('pca_vis', 'figure'),
    Output("out_url", "children"),

    Input('my-LED-display-slider-1', "value"),
    Input('dropdown', "value"),
    Input("in_url", "value"),
    )

def train_and_display(led_name, drop_name, in_url):

    if(validators.url(str(in_url)) == True):
        # https://uploads0.wikiart.org/00129/images/claude-monet/impression-sunrise.jpg!Large.jpg
        in_url = demo_predictor('RandomForest', in_url)   
        #pass
    else:
        in_url = 'No valid Url entered'

    X_test = feature_matrix.drop(columns = 'painter')
    y_test = feature_matrix['painter']

    if(drop_name == 'Random Forest'):
        model = joblib.load('best_rf_clf.joblib')
    elif (drop_name == 'SVM'):
        model = joblib.load('best_svm_clf.joblib')
    else:
        model = joblib.load('best_knn_clf.joblib')

    #model.fit(X_train, y_train)

    prediction = model.predict(X_test)
    y_scores = model.predict_proba(X_test)
    y_onehot = pd.get_dummies(y_test, columns=model.classes_)
    pred_score = round(accuracy_score(y_test, prediction) * 100.0, 2)

    cm = confusion_matrix(y_test, prediction, labels=model.classes_)

    conf_matrix = conf_m(cm, list(model.classes_))
    conf_matrix.update_layout(paper_bgcolor='rgba(0,0,0,0)')
    conf_matrix.update_layout(title_text='<i><b>Confusion matrix</b></i>', title_x=0.7)
    
    roc_figure = createRocCurve(y_scores, y_onehot)
    roc_figure.update_layout(paper_bgcolor='rgba(0,0,0,0)')
    roc_figure.update_layout(title_text='<i><b>Roc Curves</b></i>', title_x=0.5)
    roc_figure.update_yaxes(automargin=True)

    prec_recall = pr_rec_curve(y_onehot, y_scores)
    prec_recall.update_layout(paper_bgcolor='rgba(0,0,0,0)')
    prec_recall.update_layout(title_text='<i><b>Precision Recall</b></i>', title_x=0.5)
    prec_recall.update_yaxes(automargin=True)

    pca_vis = plot_pca(feature_matrix_pca)

    return pred_score, conf_matrix, roc_figure, prec_recall, pca_vis, u'Selected url: {}'.format(in_url)

app.run_server(debug=True)
import plotly.figure_factory as ff
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, roc_auc_score
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import precision_recall_curve, average_precision_score
import joblib
import cv2
import validators
from skimage.feature import hog

def createRocCurve(y_scores, y_onehot):

    np.random.seed(0)

    # Create an empty figure, and iteratively add new lines
    # every time we compute a new class
    fig = go.Figure()
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    for i in range(y_scores.shape[1]):
        y_true = y_onehot.iloc[:, i]
        y_score = y_scores[:, i]

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_score = roc_auc_score(y_true, y_score)

        name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

    fig.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=700, height=500
    )
    
    return fig

### Confusion Matric function

def conf_m(cm, classes_dict):
    z = cm

    # invert z idx values
    z = z[::-1]

    #x = ['healthy', 'multiple diseases', 'rust', 'scab']
    x = classes_dict
    y =  x[::-1].copy() # invert idx values of x

    # change each element of z to type string for annotations
    z_text = [[str(y) for y in x] for x in z]

    # set up figure 
    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Viridis')

    # add title
    #fig.update_layout(title_text='<i><b>Confusion matrix</b></i>',
    #                  #xaxis = dict(title='x'),
    #                  #yaxis = dict(title='x')
    #                 )

    # add custom xaxis title
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=0.5,
                            y=-0.15,
                            showarrow=False,
                            text="Predicted value",
                            xref="paper",
                            yref="paper"))

    # add custom yaxis title
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=-0.35,
                            y=0.5,
                            showarrow=False,
                            text="Real value",
                            textangle=-90,
                            xref="paper",
                            yref="paper"))

    # adjust margins to make room for yaxis title
    fig.update_layout(
        margin=dict(t=50, l=200),
        width=700, height=500
        )

    # add colorbar
    fig['data'][0]['showscale'] = True
    #fig.show()
    return fig


# Precission REcall Curves

def pr_rec_curve(y_onehot, y_scores):

    np.random.seed(0)

    # Create an empty figure, and iteratively add new lines
    # every time we compute a new class
    fig = go.Figure()
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=1, y1=0
    )

    for i in range(y_scores.shape[1]):
        y_true = y_onehot.iloc[:, i]
        y_score = y_scores[:, i]

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        auc_score = average_precision_score(y_true, y_score)

        name = f"{y_onehot.columns[i]} (AP={auc_score:.2f})"
        fig.add_trace(go.Scatter(x=recall, y=precision, name=name, mode='lines'))

    fig.update_layout(
        xaxis_title='Recall',
        yaxis_title='Precision',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=700, height=500
    )
    
    return fig


def plot_pca(feature_matrix_pca):
    fig = px.scatter_3d(feature_matrix_pca, x='pca_1', y='pca_2', z='pca_3',
              color='painter')
    return fig

def demo_predictor(model, img_url):
    if model == 'RandomForest':
        demo_clf = joblib.load('best_rf_clf.joblib')
    else:
        demo_clf = joblib.load('best_svm_clf.joblib')
        
    x_train1 = img_process(img_url)
    x_train1 = x_train1 / 255.0

    image_features1, hog_size1 = feature_extractor(x_train1)
    pd_single_img1 = create_feature_matrix(x_train1, image_features1)
    
    X_test_single = pd_single_img1.values
    probability_demo=demo_clf.predict_proba(X_test_single)

    Categories = list(demo_clf.classes_)
    #print(demo_clf.predict(X_test_single))
    #results = []
    #for ind,val in enumerate(Categories):
    #    print(f'{val} = {probability_demo[0][ind]*100}%')
    #    results.append(str(ind) + str(val))
       # results.append(val)
    return demo_clf.predict(X_test_single)

def create_feature_matrix(x_train, image_features):
    SIZE = 64
    pixels_index= SIZE*SIZE 
    image_numbers = x_train.shape[0]

    feature_matrix = pd.DataFrame()
    for i in range(0, image_numbers):
        new = pd.DataFrame()
        temp_hog = pd.DataFrame()
        df1 = image_features.iloc[:pixels_index,:]
        df2 = image_features.iloc[pixels_index:,:]

        new = pd.concat([new, df1['Pixel_Value']],axis=0, ignore_index=True) 
        new = pd.concat([new, df1['Gabor1']],axis=0, ignore_index=True) 
        new = pd.concat([new, df1['Gabor2']],axis=0, ignore_index=True) 
        new = pd.concat([new, df1['Gabor3']],axis=0, ignore_index=True) 
        new = pd.concat([new, df1['Gabor4']],axis=0, ignore_index=True)

        temp_hog = df1['hog']
        temp_hog = temp_hog[:3780]   # TODO: add hog size VAR

        new = pd.concat([new, temp_hog],axis=0, ignore_index=True)
        new = new.T

        feature_matrix = pd.concat([feature_matrix, new], axis=0, ignore_index=True) 
        image_features = df2
        new = pd.DataFrame(None)
        temp_hog = pd.DataFrame(None)
    
    return feature_matrix
#    feature_matrix.shape

def feature_extractor(dataset):
    SIZE = 64
    image_dataset = pd.DataFrame()
    for image in range(dataset.shape[0]):
        df = pd.DataFrame()
        input_img = dataset[image, :, :]  # one more dimension [:] if I have color
        img = input_img
        
        # Pixel values
        pixel_values = img.reshape(-1)
        df['Pixel_Value'] = pixel_values
        #df['Image_Name'] = image
        
        # Gabor
        num = 1
        kernels = []
        for theta in range(2):
            theta = theta / 4. * np.pi
            for sigma in range(1,3):  # range(1,3) default
                lamda = np.pi / 4.
                gamma = 0.5
                gabor_label = 'Gabor' + str(num)
                ksize = 9
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                kernels.append(kernel)
                
                fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                filtered_img = fimg.reshape(-1)
                df[gabor_label] = filtered_img  #Labels columns as Gabor1, Gabor2, etc.
#                print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                num += 1  #Increment for gabor column label
    
        # Hog
        resized_img = cv2.resize(img, (64,128))
        fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True)
        
        hog_size = fd.shape
        d_zeroes = pd.DataFrame(np.zeros((SIZE*SIZE, 1)))
        d_hog = pd.DataFrame(fd)
        df2 = d_zeroes.iloc[3780:,:]
        df3 = pd.concat([d_hog,df2],axis=0, ignore_index=True)
        df['hog'] = df3

            
        image_dataset = pd.concat([image_dataset, df],axis=0) 
    return image_dataset, hog_size

import urllib.request
def img_process(img_path):
    SIZE = 64

    img_array = []
    
    valid=validators.url(img_path)
    
    if valid==True:
        req = urllib.request.urlopen(img_path)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE,SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_array.append(img)
        img_array = np.array(img_array)
    else:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE,SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_array.append(img)
        img_array = np.array(img_array)

    return img_array
####
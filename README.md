# painterClassification

1. Project Description

The aim of this project is to create/train and tune a machine learning model that can distinguish the artist that created a painting, by looking at an image of the painting. This implementation make comparisons between four different painters, but the number could be increased or decreased based on the users requirements. The technology used for the classification is the library SK-LEARN, and the basic code was written in .py files as well as .ipynb (jupyter notebook files).

2. Installation Instructions

The project can be used "as is", meaning all the neccesary files for running it are included in this github repo. This includes the training and tuning of the classification model, the train and test images used for the creation of the feature dataset, as well as a working demo based on some demo images. The only assumption made is that the user has all the neccesary python libraries that are used. These can be found on the requirements.txt file.  

3. How to use this project

This project is divided into three (3) main jupyter notebooks: 
- 1DataProcess.ipynb, 
- 2ClassifierTraining.ipynb  
- 3DemoCode.ipynb

and two (2) .py python files that contain a dash DEMO:
-dash_paint02_demo
-dash_paint02_functions

* The user must make use of them in the above order
* All data that is used for TRAINING purposes is stored in ./RawData/train/<painter_name_folder>/*.jpg
* All data that is used for DEMO purposes is stored in ./RawData/train/<painter_name_folder>/*.jpg
* The user may change the painters that are allready present there by saving his own images into a corresponding subfolder in train/demo folders. This subfolder *must* have the painters name in order to be proper visualized later.

1) Open 1DataProcess.ipynb
    0. introduction: contains all necessary imports

    1. Data Processing
        1.1 Convert Images: contains all function used to convert the initial images into set dimensions (SIZE = **). Images are also converted to greyscale. There is the option to replace the original images with the transformed (saves disk space). The images are temporalily stored in memory as numpy arrays.

        1.2 Extract Features: The numpy arrays created in step 2.1 are used in order to extract features from them (function feature_extractor()). These features are stored as columns in a pandas dataframe (each column name correspondes to a specific feature e.g. "Gabor1" "hog" etc). Then these features are transformed into feature vectors which are stored in a pandas dataframe which is exported into a .csv file for later use. This is performed by the function "create_feature_matrix()".

        1.3 Train Data PCA: the feature matrix that was created before is used (directly or indirectly by loading it again from the .csv file) in order to perform a PCA on it. This PCA-reduced dataframe is also exported to a .csv file as well as plotted in a scatter 3d plot.

        1.4 Create Demo Feature Matrix: Step 1.2 is repeated for the DEMO data.

        1.5 Demo Data PCA: step 1.3 is repeated for DEMO data.


2) Open 2ClassifierTraining.ipynb

    2.0: contains all necessary imports

    2.1: Metric Functions: Contains all the functions that will later be used for diplaying various metrics about our classification models, e.g. Roc Curve Diplay, Precission - Recall, Confusion Matrix etc.

    2.2: Load Data: This section loads into memory the feature_matrix.csv pandas dataframe that we exported in 1.2

    2.3: Data Splitting and Model Fitting: In this section we describe in brief what kind of classifiers we will use
        
    2.4: Random Forest: Here we evaluate a Random Forest classifier on our data. The methodology followed is the following: First we perform a basic fitting on our data and measure its accuracy, its inference time and its training time.

        2.4.1: Basic Fit and Evaluation: we initialize a default RF Classifier, we train it on our data and we infer some basic metrics about its performance

        2.4.2: RF Optimization: We make use of the randomized Search CV technique from skikit-learn in order to optimize ou classifier. We export our optimized classifier to a .joblib file for later use. 

        2.4.3: Compare and Visualize: We compare our optimized classifier with the default classifier and we diplay all relevant performance measuring graphs.

    2.5: SVM: Here we evaluate an SVM classifier on our data. The methodology followed is the following: First we perform a basic fitting on our data and measure its accuracy, its inference time and its training time.

        2.5.1: Basic Fit and Evaluation: we initialize a default SVC Classifier, we train it on our data and we infer some basic metrics about its performance

        2.5.2: SVM Optimization: We make use of the randomized Search CV technique from skikit-learn in order to optimize ou classifier. We export our optimized classifier to a .joblib file for later use. 

        2.5.3: Compare and Visualize: We compare our optimized classifier with the default classifier and we diplay all relevant performance measuring graphs.

    2.6: KNN: Here we evaluate an KNN classifier on our data. The methodology followed is the following: First we perform a basic fitting on our data and measure its accuracy, its inference time and its training time.

        2.6.1: Basic Fit and Evaluation: we initialize a default KNN Classifier, we train it on our data and we infer some basic metrics about its performance

        2.6.2: KNN Optimization: We make use of the randomized Search CV technique from skikit-learn in order to optimize ou classifier. We export our optimized classifier to a .joblib file for later use. 

        2.6.3: Compare and Visualize: We compare our optimized classifier with the default classifier and we diplay all relevant performance measuring graphs.


3) Open 3DemoCode.ipynb notebook

    3.1 Introduction: contains all necessary imports

    3.2 Load All Relevant Functions: contains all the functions that will be used for displaying metrics on our classifiers performance rate.

    3.2.1: Demo Tools Function: Contains methods that are used for data processing and single image data processing. These will be used in order to implement a tool where the user can input an image of a painting from a url and get a prediction back about the corresponding painter.

    3.3 Demo Data Loading and PVA: Loads our faeture_matrix_demo.csv file in order to get the information needed for the predictions.

    3.4 Classifier Loading and Fitting: The user loads the classifier of his liking and the classifier is used on the demo data. All the performance metrics are displayed.

    3.5 Classifier URL tool: The user gives as input a url pointing to a painting. The program makes use of the classifier that was optimized and trained in previous steps in order to give back a prediction to the user.


4) run dash_paint02_demo.py file: This will make use of DASH libary in order to make a gui DEMO version of our code.
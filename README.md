# Big Data Analytics - SOEN-691-UU (Winter 2020)
## Professor: Tristan Glatard
## Project Category: Dataset Analysis
### Title: DATA ANALYSIS ON HUMAN ACTIVITY RECOGNITION USING SMARTPHONES DATASET
## Team Details:
|Student Name              |Student ID   |Email Address            |
|:------------------------:|:-----------:|:-----------------------:|
|Naren Morabagal Somasekhar|40082567     |narenms96@gmail.com      |  
|Adarsh Aravind            |40082585     |arvindadarsh891@gmail.com|
|Girish Kumar Kadapa       |40083533     |kadapagirish@gmail.com   |
|Liangzhao Lin             |40085480     |calinliangzhao@gmail.com |

## *ABSTRACT*

*Wearable devices like smartphones, smartwatches which have many potential applications in health monitoring, activity tracking, and personal assistance are being increasingly used to monitor and keep track of human activities. The data collected from these devices are processed using machine-learning algorithms for the classification of human activity. The results obtained from these algorithms are dependent on the availability of data (if it’s available for public use). In our work, we present a dataset of Human Activity Recognition Using Smartphone Dataset from UCI repository to predict the labels such as “WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING”. The dataset has captured 3-axial linear acceleration and 3-axial angular velocity at a constant rate of 50Hz, containing 561-feature vector with time and frequency domain variables and six labels. The goal of this project is to train the model, interpret and discuss the results of the data analysis with various metrics such as accuracy rate, F1 score, precision, and recall.* 

## 1. INTRODUCTION:
Human Activity Recognition is the problem of predicting a person's activity based on a trace of their movement using smartphone sensors [1]. Movements are often normal indoor activities such as “Walking, Walking_Upstairs, Walking_Downstairs, Sitting, Standing, Laying”. Sensors are often located on the subject, such as a smartphone or vest, and often record accelerometer data in three dimensions (x, y, z). Human activity recognition is a hotspot in the area of machine learning and pervasive computing and it is widely applied in healthcare. Human activity recognition can be divided into two categories by the method of data collection: Human activity recognition based on computer vision and based on wearable devices (accelerometer and gyroscope motion sensors). Activity recognition based on wearable sensors has become more popular, because it can be conducted without the limitation of time, location, etc. Human Activity Recognition can be accomplished by exploiting the information retrieved from various sources such as environmental or body-worn sensors. The idea is that once the subject’s activity is recognized and known, an intelligent computer system can then offer assistance [2].

To achieve this goal, we chose “Human Activity Recognition Using Smartphones Dataset” as the dataset to be analyzed in this project. Detailed information about this dataset is given in the following section. By analyzing this dataset, we can realize the prediction of the captured human activity data. It is a challenging problem given a large number of observations produced at every second, the temporal nature of the observations, and the lack of a clear way to relate accelerometer data to known movements and the prediction range of human activity is “Walking, Walking_Upstairs, Walking_Downstairs, Sitting, Standing, Laying”.

#### 1.1 RELATED WORK
Min et al [6] design two models, in which one uses only acceleration sensor data and the other uses the location information in addition to acceleration sensor data. The model with location information shows an accuracy of 95% and the model without location information does 90% accuracy.

Erhanet al [7] proposes several supervised machine learning algorithms such as Decision trees, Support Vector Machines, K-nearest neighbors (KNN) and ensemble classification methods such as Boosting, Bagging and Stacking.

Akram et al [8] analyze different activities of a person, using which a classification model is built based on feature selection. In Weka toolkit, Multilayer Perceptron, Random forest, LMT, SVM, Simple Logistic and LogitBoost are compared as an individual and combined classifiers then it was validated using K-fold cross-validation.

By doing some research and studying the above-mentioned works, we believe that using 3 supervised machine learning algorithms: Logistic Regression, KNN, and Random Forest using the scikit-learn library can get a high accuracy result.

## 2. MATERIALS AND METHODS:
#### 2.1 DATASET:

‘Human Activity Recognition Using Smartphones’ dataset made available in 2012 from UCI [3]. This dataset includes 10299 instances, for each record in the dataset it is provided:
-	Triaxial acceleration from the accelerometer (total acceleration) and the estimated body acceleration.
-	Triaxial Angular velocity from the gyroscope.
-	A 561-feature vector with time and frequency domain variables.
-	Its activity labels.
-	An identifier of the subject who experimented.

The movement data recorded was the x, y, and z accelerometer data (linear acceleration) and gyroscopic data (angular velocity) from the smartphone. The data was collected from 30 subjects aged between 19 and 48 years old performing one of 6 standard activities while wearing a waist-mounted smartphone that recorded the movement data. The video was recorded of each subject performing the activities and the movement data was labeled manually from these videos.

The six activities performed were “WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING”. The movement data recorded was the x, y, and z accelerometer data (linear acceleration) and gyroscopic data (angular velocity) from the smartphone. Observations were recorded at 50 Hz (i.e. 50 data points per second). Each subject performed the sequence of activities twice, once with the device on their left-hand-side and once with the device on their right-hand side [5].
 
#### 2.2 ALGORITHMS:
The project is implemented in python using 3 supervised machine learning algorithms: Logistic Regression, KNN, and Random Forest using scikit-learn library.
- KNN: An object is classified by a plurality vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors. If k = 1, then the object is simply assigned to the class of that single nearest neighbor.
- Random Forest: It is an ensemble learning method for classification that operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes of the individual trees.

#### 2.3 TECHNOLOGIES:
The technologies that we mainly used in this project are python, pandas, matplotlib, and scikit-learn.
- Pandas: Library used for data manipulation.
- Matplotlib: Data visualization library used for exploratory data analysis.
- Scikit-learn: Scikit-learn is a machine learning library for Python. It has classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means and DBSCAN. Scikit-learn integrates with other Python libraries like matplotlib and plotly for plotting, numpy for array vectorization, pandas dataframes, precision, recall, f1_score, scipy, MinMaxScaler, confusion_matrix.

## 3. RESULTS:
#### 3.1 DATA PREPROCESSING: 
We have followed a bunch of preprocessing steps to refine this dataset including Elimination of duplicate values, Removing null values, Scaling the features between the range [-1, 1], Feature selection, Removing outliers, Data visualization, Class labels balance check.

<p align="center">
  <img width="" height="" src="https://github.com/AdarshArvind/Big-Data-Analytics-SOEN-691-UU-Winter-2020/blob/master/images/DataPreprocessing.png">
</p>

#### 3.2 RANDOM FEATURE DISTRIBUTION:

<p align="center">
  <img width="" height="" src="https://github.com/AdarshArvind/Big-Data-Analytics-SOEN-691-UU-Winter-2020/blob/master/images/Random%20feature%20distribution.png">
</p>


#### 3.3 TRAINING DATA CLASS LABEL DISTRIBUTION:

<p align="center">
  <img width="" height="" src="https://github.com/AdarshArvind/Big-Data-Analytics-SOEN-691-UU-Winter-2020/blob/master/images/Training%20data%20class%20label%20distribution.png">
</p>

#### 3.4 MODEL TRAINING (Grid Search - Hyperparameter fine-tuning):
We used GridSearch with cross-validation 3 fold to tune hyperparameters for both KNN and Random Forest. 
Incase of KNN, the hyperparameter is n_neighbors, and in the case of Random Forest, the hyperparameters are Max_depth and N_estimators. We chose a range of values for all 3 hyperparameters and ran grid search with 3 fold cross-validation.

###### K-nearest neighbors:

<p align="center">
  <img width="" height="" src="https://github.com/AdarshArvind/Big-Data-Analytics-SOEN-691-UU-Winter-2020/blob/master/images/Grid%20Search%20KNN.png">
</p>

###### Random Forest:

<p align="center">
  <img width="" height="" src="https://github.com/AdarshArvind/Big-Data-Analytics-SOEN-691-UU-Winter-2020/blob/master/images/Grid%20Search%20Random%20Forest.png">
</p>

#### 3.5 EVALUATION METRICS:
As our main goal in this project was to build a model that would classify the human activities correctly into 6 different types, accuracy, precision, recall, and f1_score was considered as the evaluation metric. The dataset was split into 70% training data and 30% testing data. As this simulates real-world usage and is a reasonable approach, no further changes were made. 

Accuracy, precisio, recall, and f1_score for the 3 models (Logistic Regression, K-nearest Neighbors, and Random Forest) are below:

|Machine Learning Algorithm|Train Accuracy|Test Accuracy|Precision|Recall  |f1_score |
|:------------------------:|:------------:|:-----------:|:-------:|:------:|:-------:|
|K-nearest Neighbors       |0.969124      |0.914489     |0.916370 |0.914489|0.914102 | 
|Random Forest             |0.937296      |0.841534     |0.862797 |0.841534|0.836698 |
|Logistic Regression       |0.994015      |0.898880     |0.922631 |0.898880|0.895320 |

Random Forest performed reasonably well with 84.15% accuracy on the test data with 86.3% precision, 84% recall and 83.67% f1_score. A significant performance drop can be seen moving from training data to the test data. Confusion metrics for Random Forest is given below:

<p align="center">
  <img width="" height="" src="https://github.com/AdarshArvind/Big-Data-Analytics-SOEN-691-UU-Winter-2020/blob/master/images/RandomForestConfusionMetric.png">
</p>

Logistic Regression gave a small improvement in the accuracy of the test data to 89.88% with 92.27% precision, 89.88% recall, 89.5% f1_score and it has learned the train data very well but suffers a performance drop when looking at the test data. 

The K-nearest Neighbors performed the best with 91.44% test accuracy, 91.6% precision, 91.45% recall, 91.4% f1_score and a confusion matrix for KNN is below for the same:

<p align="center">
  <img width="" height="" src="https://github.com/AdarshArvind/Big-Data-Analytics-SOEN-691-UU-Winter-2020/blob/master/images/KNNConfusionMetric.png">
</p>

#### 3.6 CLASSIFICATOIN REPORT: 
The classification report for KNN model and Random Forest model is given below, 

###### KNN:

<p align="center">
  <img width="" height="" src="https://github.com/AdarshArvind/Big-Data-Analytics-SOEN-691-UU-Winter-2020/blob/master/images/Classification%20report%20KNN.png">
</p>

###### Random Forest:

<p align="center">
  <img width="" height="" src="https://github.com/AdarshArvind/Big-Data-Analytics-SOEN-691-UU-Winter-2020/blob/master/images/Classification%20report%20Random%20forest.png">
</p>

## 4. DISCUSSION:
#### 4.1 RESULTS INTERPRETATION:
In this course, we have seen many machine learning algorithms and have implemented a few algorithms in this project namely KNN, Random Forest and Logistic Regression. It was interesting to see that the 3 algorithms tested had broadly comparable evaluation metric results >83%. The models were able to learn the training data quite well, and the K-nearest neighbor was the best performing when extended to the test data.

One distinct observation was that the Random Forest model predicted 254 standing instances as sitting whereas the KNN model relatively did very well by only misclassifying 69 standing instances as sitting. 

As the majority of the misclassification classification is from labels standing, walking_upstairs and walking_downstairs in KNN and Random Forest models, a focus on analyzing and better differentiating these 3 activity types may prove more beneficial. In the context of activity/exercise tracking, correctly classifying sitting and walking activities with very high accuracy levels may be as significant a concern. 

It is also worth noting that, walking_upstairs and walking_downstairs class instances in the training data are relatively less compared to other class labels, this little data imbalance may also have contributed to misclassification.

#### 4.2 DRAWBACKS: 
Given the KNN and Random Forest model accuracies in this project, these models cannot be directly integrated into any hardware system to have real-time and perfect results. 

#### 4.2 FUTURE WORK:
-	A larger set of data, especially with a wider range of participants, could help the algorithm learn more of these variations and lead to significant model improvements. And, in the future, we would like to integrate trained models into hardware systems and get the live results and see the model in action which helps monitor health. 
-	We would also want to collect the raw data, understand and interpret the captured data from hardware sensors such as accelerometer and gyroscope and convert them into meaningful usable data and train our model from end to end.

## REFERENCES:
1.	Anguita D, Ghio A, Oneto L, et al. A public domain dataset for human activity recognition using smartphones[C]//Esann. 2013.

2.	Anguita D, Ghio A, Oneto L, et al. Human activity recognition on smartphones using a multiclass hardware-friendly support vector machine[C]//International workshop on ambient assisted living. Springer, Berlin, Heidelberg, 2012: 216-223.

3.	Dataset from UCI repository, https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones

4.	How to Model Human Activity From Smartphone Data by Jason Brownlee on September 17, 2018 in Deep Learning for Time Series

5.	Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using Smartphones. 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013.

6.	Min-CheolKwonandSunwoongChoi, “Recognition  of  Daily  Human Activity  Using  an  Artificial  Neural  Network  and  Smart  watch”,Wireless Communications and Mobile Computing, 2018.

7.	Erhan BÜLBÜL, Aydın Çetin and İbrahim Alper DOĞRU, “Human Activity  Recognition  Using  Smartphones”,  IEEE,  978-1-5386-4184, 2018

8.	Akram Bayat∗, Marc Pomplun, Duc A. Tran, “A Study on Human Activity Recognition Using Accelerometer Data from Smartphones”, Department of Computer Science, University of Massachusetts, Boston, 100 Morrissey Blvd Boston, MA 02125, USA, Elsevier Procedia  Computer Science,Vol 34, 450 -457 ,2014.

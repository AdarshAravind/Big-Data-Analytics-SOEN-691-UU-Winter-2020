# Big Data Analytics - SOEN-691-UU (Winter 2020)
## Professor: Tristan Glatard
## Project Category: Dataset Analysis
## Team Details:
|Student Name              |Student ID   |Email Address            |
|:------------------------:|:-----------:|:-----------------------:|
|Naren Morabagal Somasekhar|40082567     |narenms96@gmail.com      |  
|Adarsh Aravind            |40082585     |arvindadarsh891@gmail.com|
|Girish Kumar Kadapa       |40083533     |kadapagirish@gmail.com   |
|Liangzhao Lin             |40085480     |calinliangzhao@gmail.com |

### *ABSTRACT*

*As a team, we chose to do a dataset analysis on “Human Activity Recognition Using Smartphone Dataset from UCI repository” dataset using two classification algorithms (Random Forest, KNN, Decision Tree, SVM) using scikit-learn libraries. The dataset has captured 3-axial linear acceleration and 3-axial angular velocity at a constant rate of 50Hz, containing 561-feature vector with time and frequency domain variables and labels namely six activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING) wearing a smartphone on the waist performed by a person. The goal of this project is to train the model, interpret and discuss the results of the data analysis with various metrics such as accuracy rate, F1 score, precision and recall.* 

### Introduction:
Human Activity Recognition is the problem of predicting a person's activity based on a trace of their movement using smartphone sensors [1]. Movements are often normal indoor activities such as “Walking, Walking_Upstairs, Walking_Downstairs, Sitting, Standing, Laying”. Sensors are often located on the subject, such as a smartphone or vest, and often record accelerometer data in three dimensions (x, y, z). Human activity recognition is a hotspot in the area of machine learning and pervasive computing and it is widely applied in healthcare. Hum an activity recognition can be divided into two categories by the method of data collection: Human activity recognition based on computer vision and based on wearable devices (accelerometer and gyroscope motion sensors). Activity recognition based on wearable sensors has become more popular, because it can be conducted without the limitation of time, location, etc. Human Activity Recognition can be accomplished by exploiting the information retrieved from various sources such as environmental or body-worn sensors. The idea is that once the subject’s activity is recognized and known, an intelligent computer system can then offer assistance [2]. 

To achieve this goal, we chose “Human Activity Recognition Using Smartphones Data Set” as the data set to be analyzed in this project, detailed information about this dataset is given in the following section. By analyzing this data set, we can realize the prediction of the captured human activity data. It is a challenging problem given a large number of observations produced at every second, the temporal nature of the observations, and the lack of a clear way to relate accelerometer data to known movements and the prediction range of human activity is “Walking, Walking_Upstairs, Walking_Downstairs, Sitting, Standing, Laying”. 

### Materials and Methods:
#### Dataset:
The dataset we choose is the ‘Activity Recognition Using Smartphones’ dataset made available in 2012 from UCI [3]. This dataset includes 10299 instances, for each record in the dataset it is provided:
- Triaxial acceleration from the accelerometer (total acceleration) and the estimated body acceleration.
- Triaxial Angular velocity from the gyroscope.
- A 561-feature vector with time and frequency domain variables.
- Its activity labels.
- An identifier of the subject who carried out the experiment.

It was prepared and made available by Davide Anguita, et al. From the University of Genova, Italy and is described in full in their 2013 paper “A Public Domain Dataset for Human Activity Recognition Using Smartphones.” [4]
The dataset was made available and can be downloaded for free from the UCI Machine Learning Repository.

The data was collected from 30 subjects aged between 19 and 48 years old performing one of 6 standard activities while wearing a waist-mounted smartphone that recorded the movement data. The video was recorded of each subject performing the activities and the movement data was labelled manually from these videos.

The six activities performed were “WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING”. The movement data recorded was the x, y, and z accelerometer data (linear acceleration) and gyroscopic data (angular velocity) from the smartphone. Observations were recorded at 50 Hz (i.e. 50 data points per second). Each subject performed the sequence of activities twice, once with the device on their left-hand-side and once with the device on their right-hand side [5].

#### Algorithms:
The project will be implemented in python using any 2 supervised machine learning algorithms among Random Forest, SVM, and KNN using scikit-learn library. 
KNN: An object is classified by a plurality vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors. If k = 1, then the object is simply assigned to the class of that single nearest neighbor.
Random Forest: It is an ensemble learning method for classification that operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes of the individual trees.
SVM: Support-vector machine constructs a hyperplane or set of hyperplanes in a high- or infinite-dimensional space, which can be used for classification.

#### Technologies:
The technologies we will be using in this project are python, pandas, matplotlib, and scikit-learn. 
Pandas: Library used for data manipulation.
Matplotlib: Data visualization library used for exploratory data analysis.
Scikit-learn: Scikit-learn is a machine learning library for Python. It has classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means and DBSCAN. Scikit-learn integrates with other Python libraries: matplotlib and plotly for plotting, numpy for array vectorization, pandas dataframes, scipy, etc.

### References:
1.	Anguita D, Ghio A, Oneto L, et al. A public domain dataset for human activity recognition using smartphones[C]//Esann. 2013.
2.	Anguita D, Ghio A, Oneto L, et al. Human activity recognition on smartphones using a multiclass hardware-friendly support vector machine[C]//International workshop on ambient assisted living. Springer, Berlin, Heidelberg, 2012: 216-223.
3.	https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones
4.	How to Model Human Activity From Smartphone Data by Jason Brownlee on September 17, 2018 in Deep Learning for Time Series
5.	Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using Smartphones. 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013.

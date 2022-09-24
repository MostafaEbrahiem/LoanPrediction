# LoanPrediction
 
Web application made using Django framework , It uses machine learning to predict if the user is risky or not to obtain a loan from the bank by providing some information about the user

![image](https://user-images.githubusercontent.com/88105870/192112389-fe34951d-3256-4cbf-a24e-19d794ff6cdc.png)

![image](https://user-images.githubusercontent.com/88105870/192112432-3ca1170c-767f-4d31-91e7-d7c0c3d61f20.png)

![image](https://user-images.githubusercontent.com/88105870/192112442-c71107dc-2525-4d75-94b1-9ed0dafac3ca.png)

![image](https://user-images.githubusercontent.com/88105870/192112456-7cffe922-b938-4e77-93ac-800dc73fe0e5.png)

It was hosted using Heroku and you can try it throw this link:
https://loan-prediction1001.herokuapp.com/

About the machine learning part:-

1)used these dataset(https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset?datasetId=137197&sortBy=dateRun&tab=profile)
delete the first column and fill the null values based on the value of the label column (Loan_Status) if it's 1(accept) ,then it's employed ,not having kids ,not maried ,has credit history and graduated (the other numirical values filled with the average ) and vise versa if it's 0(reject).

2)Trained the data on three different algorithms (bagging classifier with Grid search , GradientBoosting Classifier with GridSearchCV and SVC with GridSearchCV)
GradientBoosting Classifier with GridSearchCV was the best one of them all gaining 100% accuracy on training and testing .

3)althogh the data is very small and some results is not realistic ,but some result are and it's good practise to train on real and huge dataset in the future.

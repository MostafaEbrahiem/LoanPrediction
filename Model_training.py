from copyreg import pickle
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit,GridSearchCV
from sklearn.ensemble import BaggingClassifier
import pickle as pk
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def train(Loan_data):

    cols = [0,12]
    X= Loan_data.drop(Loan_data.columns[cols],axis=1)
    y = Loan_data['Loan_Status']

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=21)

    
    #train using bagging classifier Grid search

    # n_estimators = [10,30,50,70,80,150,160, 170,175,180,185];
    # cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)

    # parameters = {'n_estimators':n_estimators,
              
    #     }
    # grid_bagg = GridSearchCV(BaggingClassifier(base_estimator= None, ## If None, then the base estimator is a decision tree.
    #                                   bootstrap_features=False),
    #                              param_grid=parameters,
    #                              cv=cv,
    #                              n_jobs = -1)
    # grid_bagg.fit(X_train,y_train) 
    # pk.dump(grid_bagg, open("Model_BaggingClassifierGridSearchCV", 'wb'))

    # print (grid_bagg.best_score_)
    # print (grid_bagg.best_params_)
    # print (grid_bagg.best_estimator_)
    
    # bagg_grid = grid_bagg.best_estimator_
    # bagg_train_grid = round(bagg_grid.score(X_train, y_train) * 100, 2)
    # bagg_accuracy_grid = bagg_grid.score(X_train,y_train).round(2)*100

    # print("Training Accuracy with GridSearch :",bagg_train_grid  ,"%")
    # print("Model Accuracy with GridSearch    :",bagg_accuracy_grid ,"%")
    # print("\033[1m--------------------------------------------------------\033[0m")


    # GradientBoostingClassifierGridSearchCV

#     param_grid = {
#         'learning_rate': [0.01, 0.025, 0.05, 0.075, 0.1],
#         'n_estimators':[140, 150, 160], 
#         'max_depth':[3, 4, 5]       
#         }

#     cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)
# # The grid search object
#     grid_gbc = GridSearchCV(GradientBoostingClassifier(random_state=42), 
#                       param_grid=param_grid, 
#                       scoring='f1',
#                       cv = cv,
#                       verbose=0, 
#                       n_jobs=-1)

#     grid_gbc.fit(X_train,y_train) 
#     pk.dump(grid_gbc, open("GradientBoostingClassifierGridSearchCV", 'wb'))

#     gbc_grid = grid_gbc.best_estimator_

#     gbc_train_grid = round(gbc_grid.score(X_train, y_train) * 100, 2)
#     gbc_accuracy_grid = gbc_grid.score(X_train,y_train).round(2)*100
#     print (grid_gbc.best_score_)
#     print (grid_gbc.best_params_)
#     print (grid_gbc.best_estimator_)
#     print("Training Accuracy with GridSearch :",gbc_train_grid  ,"%")
#     print("Model Accuracy with GridSearch    :",gbc_accuracy_grid ,"%")
#     print("\033[1m--------------------------------------------------------\033[0m")

    # SVC GridSearchCV 📊📈
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']} # 'precomputed','sigmoid','linear','pol

    cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)
    grid_svc = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3,cv=cv)
 
    # fitting the model for grid search
    grid_svc.fit(X_train, y_train)
    pk.dump(grid_svc, open("Model_SVCGridSearchCV", 'wb'))

    GridSearchCV(cv=StratifiedShuffleSplit(n_splits=10, random_state=15, test_size=0.3,
            train_size=None),
             estimator=SVC(),
             param_grid={'C': [0.1, 1, 10, 100, 1000],
                         'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                         'kernel': ['rbf']},
             verbose=3)
    

    print (grid_svc.best_score_)
    print (grid_svc.best_params_)
    print (grid_svc.best_estimator_)

    ### Using the best parameters from the grid-search.
    svc_grid = grid_svc.best_estimator_

    svc_train_grid = round(svc_grid.score(X_train, y_train) * 100, 2)
    svc_accuracy_grid = svc_grid.score(X_train,y_train).round(2)*100

    print("Training Accuracy with GridSearch :",svc_train_grid  ,"%")
    print("Model Accuracy with GridSearch    :",svc_accuracy_grid ,"%")
    print("\033[1m--------------------------------------------------------\033[0m")

    

#remove unnessecery data and fill the null ones
from asyncio.windows_events import NULL
from cmath import nan
import pandas as pd
import category_encoders as ce 
from sklearn.preprocessing import LabelEncoder


def clean(Loan_data):
    Loan_data.drop('Loan_ID',axis=1,inplace=True)

    #filled null values based on the value of the Loan_Status :
    #If yes then:-
    #Self_Employed is Yes and Dependents is min(0) and Married is No and vise versa
    # rest of the values filled with the mean (but integer values not float) 

    Loan_data['Gender']=Loan_data['Gender'].fillna('Female')
    Loan_data['LoanAmount']=Loan_data['LoanAmount'].fillna(146)
    Loan_data['Loan_Amount_Term']=Loan_data['Loan_Amount_Term'].fillna(360)
    for indx,i in enumerate(Loan_data['Credit_History']):
        if(pd.isna(i)):
            if(Loan_data['Loan_Status'][indx]=='N'):Loan_data['Credit_History'][indx]=0
            else:Loan_data['Credit_History'][indx]=1

    for indx,i in enumerate(Loan_data['Self_Employed']):
        if(pd.isna(i)):
            if(Loan_data['Loan_Status'][indx]=='N'):Loan_data['Self_Employed'][indx]='No'
            else:Loan_data['Self_Employed'][indx]='Yes'

    for indx,i in enumerate(Loan_data['Dependents']):
        if(pd.isna(i)):
            if(Loan_data['Loan_Status'][indx]=='N'):Loan_data['Dependents'][indx]='3+'
            else:Loan_data['Dependents'][indx]='0'
 
    for indx,i in enumerate(Loan_data['Married']):
        if(pd.isna(i)):
            if(Loan_data['Loan_Status'][indx]=='N'):Loan_data['Married'][indx]='Yes'
            else:Loan_data['Married'][indx]='No'
    
    
    #data labeled
    print(Loan_data)
    encoder = ce.OrdinalEncoder(['Gender','Married','Education','Self_Employed','Property_Area'])
    Loan_data = encoder.fit_transform(Loan_data)

    encoder = LabelEncoder()
    Loan_data['Loan_Status'] = encoder.fit_transform(Loan_data['Loan_Status'])

    print(Loan_data)
    #Loan_data.to_csv("train.csv")
 
    return Loan_data.astype(int)

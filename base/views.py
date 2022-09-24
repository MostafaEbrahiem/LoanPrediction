from urllib import request
from django.shortcuts import render
import pickle
import numpy as np
from numpy import array

def Home(request):
    return render(request,'Home.html')





def Input(request):
    return render(request,'Input.html')  

def result(request):

    models=[]
    with open('Model_BaggingClassifierGridSearchCV', 'rb') as f:
        models.append(pickle.load(f))
    
    with open('Model_GradientBoostingClassifierGridSearchCV', 'rb') as f:
        models.append(pickle.load(f))

    with open('Model_SVCGridSearchCV', 'rb') as f:
        models.append(pickle.load(f))
    
    

    res ={0:0,1:0}
    #get values from the user 

    values=[]
    values.append(np.int(request.GET.get('Gender')))
    values.append(np.int(request.GET.get('married')))
    values.append(np.int(request.GET.get('dependents')))
    values.append(np.int(request.GET.get('Graduate')))
    values.append(np.int(request.GET.get('Self_Employed')))
    values.append(np.int(request.GET.get('Salary')))
    values.append(np.int(request.GET.get('CoapplicantIncome')))
    values.append(np.int(request.GET.get('LoanAmount')))
    values.append(np.int(request.GET.get('Loan_Amount_Term')))
    values.append(np.int(request.GET.get('Credit_History')))
    values.append(np.int(request.GET.get('Property_Area')))

    values = array(values).reshape(1, -1)

    #pridict on the three models
    for model in models:
        r=model.predict(values)
        print(r)
        
        res[r[0]]+=1 
    print(res)



    f_res=""
    # if(res[0]< res[1])
    if(res[1]): f_res = " شخص غير مضمون,الرجاء الابتعاد عنه "
    else: f_res = " شخص مضمون , يمكنك البدء في عملية الاقراض "
    context = {'f_res': f_res}
    return render(request,'result.html',context)  


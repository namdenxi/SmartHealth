from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

heart_data = pd.read_csv('./home/media/files/heart.csv')
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']
result = ['Bạn không bị bệnh tim','Bạn có bị bệnh tim']

X_train, X_test, Y_train, Y_test = train_test_split(X.to_numpy(), Y, random_state=10, test_size=0.3, shuffle=True)
model = DecisionTreeClassifier(max_depth=8)
model.fit(X_train, Y_train)

prediction = model.predict(X_test)
accuracy = accuracy_score(Y_test, prediction)*100
accuracy_rounding = round(accuracy, 2)
# print(f'Accuracy: {accuracy_rounding}%')
# data = np.array([[30,0,0,100,204,0,0,100,0,0,2,0,2]])
# data_predict = model.predict(data)
# print(result[int(data_predict)])


# Create your views here.
def Heart(request):
    text_warning = ''
    show = ''
    content = ''
    if request.method == 'POST':
        age = request.POST.get('age') 
        sex = request.POST.get('sex') 
        cp = request.POST.get('cp')
        trestbps = request.POST.get('trestbps')
        chol = request.POST.get('chol')
        fbs = request.POST.get('fbs')
        restecg = request.POST.get('restecg')
        thalach = request.POST.get('thalach')
        exang = request.POST.get('exang')
        oldpeak = request.POST.get('oldpeak')
        slope = request.POST.get('slope')
        ca = request.POST.get('ca')
        thal = request.POST.get('thal')
        
        if age == '' or sex == '' or cp == '' or trestbps == '' or chol == '' or fbs == '' or restecg == '' or thalach == '' or exang == '' or oldpeak == '' or slope == '' or ca == '' or thal == '':
            text_warning = 'Hãy nhập đầy đủ thông tin trước khi gửi đi'
        else:
            try:
                data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
                data_predict = model.predict(data)
                # print(result[int(data_predict)])
                show = 'show'
                content = f'Độ chính xác {accuracy_rounding}%. {result[int(data_predict)]}'
            except:
                text_warning = 'Hãy nhập đúng giá trị'
    return render(request, 'heart.html', {'text_warning':text_warning, 'show': show, 'content': content})

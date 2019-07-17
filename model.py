import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
from sklearn.metrics import confusion_matrix
from flask import Flask, jsonify, request
import re



def preprocessiong(Text):
    Text = Text.lower()
    Text = Text.strip()
    Text = re.sub(r'[0-9]+', '', Text)
    return Text


df_jops = pd.read_csv("Job titles and industries.csv")

df_jops = df_jops.drop_duplicates(keep='first',subset =['job title'])
df_jops['job title'] = df_jops['job title'].apply(preprocessiong)

count_class = df_jops.industry.value_counts()
print(count_class)
class_IT = count_class[0]
class_Acc = count_class[3]
df_class_0 = df_jops[df_jops['industry'] == 'IT']
df_class_1 = df_jops[df_jops['industry'] == 'Marketing']
df_class_2 = df_jops[df_jops['industry'] == 'Education']
df_class_3 = df_jops[df_jops['industry'] == 'Accountancy']

df_class_1_over = df_class_1.sample(class_IT,replace=True)
df_class_2_over = df_class_2.sample(class_IT,replace=True)
df_class_3_over = df_class_3.sample(class_IT,replace=True)

df_test_over = pd.concat([df_class_0,df_class_1_over,df_class_2_over, df_class_3_over], axis=0)
df_jops = df_test_over


Label = df_jops['industry']
text = df_jops['job title']

x_train,x_test,y_train,y_test=train_test_split(text,Label,random_state=0,test_size=0.2)

count_vect = CountVectorizer()

x_train_c = count_vect.fit_transform(x_train)
tf_transform = TfidfTransformer().fit(x_train_c)
x_train_transformed = tf_transform.transform(x_train_c)

def trasform_text(Text):
    TransformedText = count_vect.transform(Text)
    TransformedText = tf_transform.transform(TransformedText)
    return TransformedText

x_test_transformed = trasform_text(x_test)

label = LabelEncoder()
label.fit(y_train)
y_train= label.transform(y_train)
Liner_svc = LinearSVC()

clf = Liner_svc.fit(x_train_transformed,y_train)
predicted = Liner_svc.predict(x_test_transformed)


print('Average accuracy on test set={}'.format(np.mean(predicted == label.transform(y_test))))


conf_mat = confusion_matrix(y_true=label.transform(y_test), y_pred=predicted)
print('Confusion matrix:\n', conf_mat)

def predict_text(text):
    text=preprocessiong(text)
    text=trasform_text([text])
    p=Liner_svc.predict(text)
    return label.classes_[p[0]]
print(predict_text('it devolper'))
app = Flask(__name__)
@app.route('/text/<string:jobTitle>',methods =['Get'])
def returnone(jobTitle):
    industry = predict_text(jobTitle)
    return jsonify({'industry':industry})

if __name__ =='__main__':
    app.run(debug=True,port=8080)
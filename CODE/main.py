import sklearn
from flask import Flask, render_template, request, url_for, flash, redirect
from werkzeug.utils import secure_filename
import os
from sklearn.model_selection import LeaveOneOut
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
import shutil
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, cross_validate
from matplotlib import pyplot as plt
from imblearn.over_sampling import SMOTE
from collections import Counter
from catboost import CatBoostClassifier
app=Flask(__name__)
app.config['UPLOAD_FOLDER']=r"uploads"
app.config['SECRET_KEY']='b0b4fbefdc48be27a6123605f02b6b86'
# D:\rupesh\uploaded_csv

def preprocessing(file):
    full_data.dropna(axis=0, how='any', inplace=True)
    return file

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/load', methods=["POST","GET"])
def load():
    if request.method=="POST":
        myfile=request.files['filename']
        ext=os.path.splitext(myfile.filename)[1]
        print("1111!!!!!!")
        print(ext)
        if ext.lower() == ".csv":
            shutil.rmtree(app.config['UPLOAD_FOLDER'])
            os.mkdir(app.config['UPLOAD_FOLDER'])
            myfile.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(myfile.filename)))
            flash('The data is loaded successfully','success')
            return render_template('load_dataset.html')
        else:
            flash('Please upload a CSV type document only','warning')
            return render_template('load_dataset.html')
    return render_template('load_dataset.html')

@app.route('/view')
def view():
    #dataset
    myfile=os.listdir(app.config['UPLOAD_FOLDER'])
    global full_data
    full_data=pd.read_csv(os.path.join(app.config["UPLOAD_FOLDER"],myfile[0]))
    return render_template('view_dataset.html', col=full_data.columns.values, df=list(full_data.values.tolist()))

@app.route('/split', methods=['POST','GET'])
def split():
    if request.method=="POST":
        test_size=float(request.form['size'])
        # test_size=test_size/100
        global df_encoded
        #preprocessing
        df_encoded=preprocessing(full_data)
        #split
        global X, y, X_train, X_test, y_train, y_test
        X = df_encoded.iloc[:, :-1]
        y = df_encoded.iloc[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,random_state=2)
        flash('The dataset is transformed and split successfully','success')
        return redirect(url_for('train_model'))
    return render_template('split_dataset.html')

@app.route('/train_model', methods=['GET','POST'])
def train_model():
    if request.method=="POST":
        model_no=int(request.form['algo'])
        if model_no==0:
            print("U have not selected any model")
        elif model_no==1:
            acc_km = logistic_regression()
            return render_template('train_model.html', acc=acc_km, model=model_no)
        elif model_no==2:
            acc_agg= Naive_Bayes()
            return render_template('train_model.html', acc=acc_agg, model=model_no)
        elif model_no==3:
            acc_gnb=K_Nearest_Neighbors()
            return render_template('train_model.html', acc=acc_gnb, model=model_no)
        elif model_no==4:
            acc_dct=Decision_Tree()
            return render_template('train_model.html', acc=acc_dct, model=model_no)
        elif model_no==5:
            acc_gnb=Random_Forest()
            return render_template('train_model.html',acc=acc_gnb,model=model_no)
        elif model_no==6:
            acc_dct=Multi_layer_Perceptron ()
            return render_template('train_model.html', acc=acc_dct, model=model_no)
        elif model_no==7:
            acc_gnb_km=Support_Vector_Machine()
            return render_template('train_model.html', acc=acc_gnb_km, model=model_no)
        elif model_no == 8:
            acc_gnb_cbc = Cat_Boost_Classifier()
            return render_template('train_model.html', acc=acc_gnb_cbc, model=model_no)
    return render_template('train_model.html')

d={}
d1={}

def logistic_regression():
        scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score),
               'recall': make_scorer(recall_score),
                'f1_score': make_scorer(f1_score)}
        kfold = KFold(n_splits=10)
        loo = LeaveOneOut()
        model = linear_model.LogisticRegression(solver='saga')
        smt=SMOTE()
        x_train_sm,y_train_sm=smt.fit_resample(X_train,y_train)
        loo.get_n_splits(x_train_sm)
        print(loo.split(x_train_sm))
        results1=cross_validate(estimator=model,X=x_train_sm,y=y_train_sm,cv=10,scoring=scoring)
        global d1
        n11=np.mean(results1['test_accuracy'])
        n12=np.mean(results1['test_precision'])
        n13=np.mean(results1['test_recall'])
        n14=np.mean(results1['test_f1_score'])
        d1['logistic']=[n11,n12,n13,n14]
        print(d1)
        results = cross_validate(estimator=model,X=x_train_sm,y=y_train_sm,cv=kfold,scoring=scoring)
        m11=np.mean(results['test_accuracy'])
        m12=np.mean(results['test_precision'])
        m13=np.mean(results['test_recall'])
        m14=np.mean(results['test_f1_score'])
        global d
        d['Logistic']=[m11, m12, m13, m14]
        model.fit(x_train_sm, y_train_sm)
        y_pred = model.predict(X_test)
        acc_km = accuracy_score(y_test, y_pred)
        flash("LogisticRegression performed Successfully", 'secondary')
        return acc_km

def Naive_Bayes():
    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score),
               'recall': make_scorer(recall_score),
               'f1_score': make_scorer(f1_score)}
    kfold = KFold(n_splits=10)
    loo = LeaveOneOut()
    model = GaussianNB()
    smt = SMOTE()
    x_train_sm, y_train_sm = smt.fit_resample(X_train, y_train)
    loo.get_n_splits(x_train_sm)
    results1 = cross_validate(estimator=model, X=x_train_sm, y=y_train_sm, cv=10, scoring=scoring)
    n21 = np.mean(results1['test_accuracy'])
    n22 = np.mean(results1['test_precision'])
    n23 = np.mean(results1['test_recall'])
    n24 = np.mean(results1['test_f1_score'])
    d1['naivey_bayes'] = [n21, n22, n23, n24]
    results = cross_validate(estimator=model, X=x_train_sm, y=y_train_sm, cv=kfold, scoring=scoring)
    m21 = np.mean(results['test_accuracy'])
    m22 = np.mean(results['test_precision'])
    m23 = np.mean(results['test_recall'])
    m24 = np.mean(results['test_f1_score'])
    d["naivey_bayes"]=[m21,m22,m23,m24]
    model.fit(x_train_sm, y_train_sm)
    y_pred = model.predict(X_test)
    acc_gnb = accuracy_score(y_test, y_pred)
    flash("Naive Bayes model created Successfully", 'secondary')
    return acc_gnb

def K_Nearest_Neighbors():
    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score),
               'recall': make_scorer(recall_score),
               'f1_score': make_scorer(f1_score)}
    kfold = KFold(n_splits=10)
    loo = LeaveOneOut()
    classifier = KNeighborsClassifier(n_neighbors=8)
    smt = SMOTE()
    x_train_sm, y_train_sm = smt.fit_resample(X_train, y_train)
    loo.get_n_splits(x_train_sm)
    results1 = cross_validate(estimator=classifier, X=x_train_sm, y=y_train_sm, cv=10, scoring=scoring)
    n31 = np.mean(results1['test_accuracy'])
    n32 = np.mean(results1['test_precision'])
    n33 = np.mean(results1['test_recall'])
    n34 = np.mean(results1['test_f1_score'])
    d1['K_Nearest_Neighbors'] = [n31, n32, n33, n34]
    results = cross_validate(estimator=classifier, X=x_train_sm, y=y_train_sm, cv=kfold, scoring=scoring)
    m31 = np.mean(results['test_accuracy'])
    m32 = np.mean(results['test_precision'])
    m33 = np.mean(results['test_recall'])
    m34 = np.mean(results['test_f1_score'])
    d["K_Nearest_Neighbors"] = [m31, m32, m33, m34]
    classifier.fit(x_train_sm, y_train_sm)
    y_pred = classifier.predict(X_test)
    acc_gnb = accuracy_score(y_test, y_pred)
    flash("knn model created Successfully", 'secondary')
    return acc_gnb

def Decision_Tree():
    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score),
               'recall': make_scorer(recall_score),
               'f1_score': make_scorer(f1_score)}
    kfold = KFold(n_splits=10)
    loo = LeaveOneOut()
    classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
    smt = SMOTE()
    x_train_sm, y_train_sm = smt.fit_resample(X_train, y_train)
    loo.get_n_splits(x_train_sm)
    results1 = cross_validate(estimator=classifier, X=x_train_sm, y=y_train_sm, cv=10, scoring=scoring)
    n41 = np.mean(results1['test_accuracy'])
    n42 = np.mean(results1['test_precision'])
    n43 = np.mean(results1['test_recall'])
    n44 = np.mean(results1['test_f1_score'])
    d1['Decision_Tree'] = [n41, n42, n43, n44]
    counter=Counter(x_train_sm)
    counter1=Counter(y_train_sm)
    results = cross_validate(estimator=classifier, X=x_train_sm, y=y_train_sm, cv=kfold, scoring=scoring)
    m41 = np.mean(results['test_accuracy'])
    m42 = np.mean(results['test_precision'])
    m43 = np.mean(results['test_recall'])
    m44 = np.mean(results['test_f1_score'])
    d["Decision_Tree"] = [m41, m42, m43, m44]
    classifier.fit(x_train_sm, y_train_sm)
    y_pred = classifier.predict(X_test)
    acc_gnb = accuracy_score(y_test, y_pred)
    flash("decision tree model created Successfully", 'secondary')
    return acc_gnb

def Random_Forest():
    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score),
               'recall': make_scorer(recall_score),
               'f1_score': make_scorer(f1_score)}
    kfold = KFold(n_splits=10)
    loo = LeaveOneOut()
    model = RandomForestClassifier(n_estimators=20)
    smt = SMOTE()
    global x_train_sm,y_train_sm
    x_train_sm, y_train_sm = smt.fit_resample(X_train, y_train)
    loo.get_n_splits(x_train_sm)
    results1 = cross_validate(estimator=model, X=x_train_sm, y=y_train_sm, cv=10, scoring=scoring)
    n51 = np.mean(results1['test_accuracy'])
    n52 = np.mean(results1['test_precision'])
    n53 = np.mean(results1['test_recall'])
    n54 = np.mean(results1['test_f1_score'])
    d1['Random_Forest'] = [n51, n52, n53, n54]
    results = cross_validate(estimator=model, X=x_train_sm, y=y_train_sm, cv=kfold, scoring=scoring)
    m51 = np.mean(results['test_accuracy'])
    m52 = np.mean(results['test_precision'])
    m53 = np.mean(results['test_recall'])
    m54 = np.mean(results['test_f1_score'])
    d["Random_Forest"] = [m51, m52, m53, m54]
    model.fit(x_train_sm, y_train_sm)
    y_pred = model.predict(X_test)
    acc_dct = accuracy_score(y_test, y_pred)
    flash("random forest model created Successfully", 'secondary')
    return acc_dct

def Multi_layer_Perceptron():
    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score),
               'recall': make_scorer(recall_score),
               'f1_score': make_scorer(f1_score)}
    kfold = KFold(n_splits=10)
    loo = LeaveOneOut()
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
    smt = SMOTE()
    x_train_sm, y_train_sm = smt.fit_resample(X_train, y_train)
    loo.get_n_splits(x_train_sm)
    results1 = cross_validate(estimator=clf, X=x_train_sm, y=y_train_sm, cv=10, scoring=scoring)
    n61 = np.mean(results1['test_accuracy'])
    n62 = np.mean(results1['test_precision'])
    n63 = np.mean(results1['test_recall'])
    n64 = np.mean(results1['test_f1_score'])
    d1['Multi_layer_Perceptron'] = [n61, n62, n63, n64]
    results = cross_validate(estimator=clf, X=x_train_sm, y=y_train_sm, cv=kfold, scoring=scoring)
    m61 = np.mean(results['test_accuracy'])
    m62 = np.mean(results['test_precision'])
    m63 = np.mean(results['test_recall'])
    m64 = np.mean(results['test_f1_score'])
    d["Multi_layer_Perceptron"] = [m61, m62, m63, m64]
    clf.fit(x_train_sm,y_train_sm)
    y_pred = clf.predict(X_test)
    acc_dct = accuracy_score(y_test, y_pred)
    print(acc_dct)
    flash("multi layer perceptron model created Successfully", 'secondary')
    return acc_dct

def Support_Vector_Machine():
    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score),
               'recall': make_scorer(recall_score),
               'f1_score': make_scorer(f1_score)}
    kfold = KFold(n_splits=10)
    loo = LeaveOneOut()
    model = SVC()
    smt = SMOTE()
    x_train_sm, y_train_sm = smt.fit_resample(X_train, y_train)
    loo.get_n_splits(x_train_sm)
    results1 = cross_validate(estimator=model, X=x_train_sm, y=y_train_sm, cv=10, scoring=scoring)
    n71 = np.mean(results1['test_accuracy'])
    n72 = np.mean(results1['test_precision'])
    n73 = np.mean(results1['test_recall'])
    n74 = np.mean(results1['test_f1_score'])
    d1['Support_Vector_Machine'] = [n71, n72, n73, n74]
    results = cross_validate(estimator=model, X=x_train_sm, y=y_train_sm, cv=kfold, scoring=scoring)
    print(results)
    m71 = np.mean(results['test_accuracy'])
    m72 = np.mean(results['test_precision'])
    m73 = np.mean(results['test_recall'])
    m74 = np.mean(results['test_f1_score'])
    d["Support_Vector_Machine"] = [m71, m72, m73, m74]
    print("d1")
    print(d1)
    model.fit(x_train_sm, y_train_sm)
    y_pred=model.predict(X_test)
    acc_dct=accuracy_score(y_test,y_pred)
    flash("support vector classifier model created Successfully", 'secondary')
    return acc_dct

def Cat_Boost_Classifier():
    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score),
               'recall': make_scorer(recall_score),
               'f1_score': make_scorer(f1_score)}
    kfold = KFold(n_splits=10)
    loo = LeaveOneOut()
    model = CatBoostClassifier()
    smt = SMOTE()
    x_train_sm, y_train_sm = smt.fit_resample(X_train, y_train)
    loo.get_n_splits(x_train_sm)
    results1 = cross_validate(estimator=model, X=x_train_sm, y=y_train_sm, cv=10, scoring=scoring)
    n81 = np.mean(results1['test_accuracy'])
    n82 = np.mean(results1['test_precision'])
    n83 = np.mean(results1['test_recall'])
    n84 = np.mean(results1['test_f1_score'])
    d1['Cat_Boost_Classifier'] = [n81, n82, n83, n84]
    results = cross_validate(estimator=model, X=x_train_sm, y=y_train_sm, cv=kfold, scoring=scoring)
    print(results)
    m81 = np.mean(results['test_accuracy'])
    m82 = np.mean(results['test_precision'])
    m83 = np.mean(results['test_recall'])
    m84 = np.mean(results['test_f1_score'])
    d["Cat_Boost_Classifier"] = [m81, m82, m83, m84]
    print("d1")
    print(d1)
    model.fit(x_train_sm, y_train_sm)
    y_pred=model.predict(X_test)
    acc_dct=accuracy_score(y_test,y_pred)
    flash("CatBoostClassifier model created Successfully", 'secondary')
    return acc_dct


@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method=='POST':
        f1 = request.form['f1']
        f2 = request.form['f2']
        f3 = request.form['f3']
        f4 = request.form['f4']
        f5 = request.form['f5']
        f6 = request.form['f6']
        f7 = request.form['f7']
        all_obj_vals=[[(f1),(f2),(f3),(f4),(f5),(f6),(f7)]]
        model=RandomForestClassifier(n_estimators=20)

        model.fit(x_train_sm.drop("PatientNumber",axis=1),y_train_sm)
        pred=model.predict(all_obj_vals)
        print(pred)
        return render_template('prediction.html', pred=pred)
    return render_template('prediction.html')
@app.route("/chart")
def chart():
        data_frame=pd.DataFrame(d)
        data_frame.to_csv("all_metrics.csv")
        data_frame.plot(kind="bar")
        plt.title("Algorithm comparision 10_fold cross validation")
        plt.xlabel("algorithms")
        plt.ylabel("score")
        print(data_frame)
        graph_data=plt.show()
        plt.show()
        print(data_frame)
        return redirect(url_for('train_model'))

@app.route("/chart1")
def chart1():
        data_frame=pd.DataFrame(d1)
        data_frame.to_csv("CHART1_metrics.csv")
        data_frame.plot(kind="bar")
        plt.title("Algorithm comparision in LOO validation")
        plt.xlabel("algorithms")
        plt.ylabel("score")
        print(data_frame)
        graph_data=plt.show()
        plt.show()
        print(data_frame)
        return redirect(url_for('train_model'))


if __name__=='__main__':
    app.run(debug=True)
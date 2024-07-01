from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

def process_file(filepath):
    df = pd.read_csv(filepath)

    # Data cleaning and preprocessing
    df.drop('datetime', axis=1, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    
    for col in ['model', 'failure']:
        df[col] = df[col].astype('category')
    
    num_atr = df.select_dtypes(['int64', 'float64']).columns
    scaler = StandardScaler()
    scaler.fit(df[num_atr])
    df[num_atr] = scaler.transform(df[num_atr])
    
    y = df['failure']
    X = df.drop('failure', axis=1)
    X = pd.get_dummies(X, drop_first=True)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=123)
    
    results = {}

    # Logistic Regression
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred_train_lr = lr.predict(X_train)
    y_pred_test_lr = lr.predict(X_val)
    results['logistic_regression'] = {
        'train_accuracy': accuracy_score(y_train, y_pred_train_lr),
        'test_accuracy': accuracy_score(y_val, y_pred_test_lr),
        'train_f1': f1_score(y_train, y_pred_train_lr, average='weighted'),
        'test_f1': f1_score(y_val, y_pred_test_lr, average='weighted')
    }

    # Decision Tree
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    y_pred_train_dtc = dtc.predict(X_train)
    y_pred_test_dtc = dtc.predict(X_val)
    results['decision_tree'] = {
        'train_accuracy': accuracy_score(y_train, y_pred_train_dtc),
        'test_accuracy': accuracy_score(y_val, y_pred_test_dtc),
        'train_f1': f1_score(y_train, y_pred_train_dtc, average='weighted'),
        'test_f1': f1_score(y_val, y_pred_test_dtc, average='weighted')
    }

    # Random Forest
    rfc = RandomForestClassifier(n_estimators=40, random_state=0)
    rfc.fit(X_train, y_train)
    y_pred_train_rfc = rfc.predict(X_train)
    y_pred_test_rfc = rfc.predict(X_val)
    results['random_forest'] = {
        'train_accuracy': accuracy_score(y_train, y_pred_train_rfc),
        'test_accuracy': accuracy_score(y_val, y_pred_test_rfc),
        'train_f1': f1_score(y_train, y_pred_train_rfc, average='weighted'),
        'test_f1': f1_score(y_val, y_pred_test_rfc, average='weighted')
    }

    # Confusion Matrix and Classification Report for Random Forest
    results['random_forest_confusion_matrix'] = confusion_matrix(y_val, y_pred_test_rfc)
    results['random_forest_classification_report'] = classification_report(y_val, y_pred_test_rfc)

    return results

@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            results = process_file(filepath)
            return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)

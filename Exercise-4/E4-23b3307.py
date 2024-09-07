import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve
import os

if not os.path.exists('Images'):
    os.makedirs('Images')

if not os.path.exists('Metrics'):
    os.makedirs('Metrics')

data_v0 = pd.read_csv('clusters-4-v0.csv')
data_v1 = pd.read_csv('clusters-4-v1.csv')
data_v2 = pd.read_csv('clusters-4-v2.csv')

x_train_v0, x_test_v0, y_train_v0, y_test_v0 = train_test_split(data_v0[['x1','x2']], data_v0['y'], test_size=0.2, random_state=42)
x_train_v1, x_test_v1, y_train_v1, y_test_v1 = train_test_split(data_v1[['x1','x2']], data_v1['y'], test_size=0.2, random_state=42)
x_train_v2, x_test_v2, y_train_v2, y_test_v2 = train_test_split(data_v2[['x1','x2']], data_v2['y'], test_size=0.2, random_state=42)

datasets = [(data_v0, "Dataset 1"), (data_v1, "Dataset 2"), (data_v2, "Dataset 3")]

for i, (data, label) in enumerate(datasets):
    plt.figure(figsize=(16, 10))
    
    plt.scatter(data['x1'], data['x2'], c=data['y'], cmap='viridis', s=10)
    plt.title(f'Scatter Plot of {label}')
    plt.xlabel('x1')
    plt.ylabel('x2')
    
    plt.tight_layout()
    plt.savefig(f'Images/dataset-{i+1}-overview.png', dpi=400)
    plt.show()
    
    class_counts = data['y'].value_counts()
    print(f"Class balance in {label}:\n{class_counts}\n")

def draw_decision_boundary(model, X, y, resolution=100, size=10, edgecolor='k'):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap='viridis', alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors=edgecolor, cmap='viridis', s=size)  

classifiers = {
    'Logistic Regression': LogisticRegression(),
    'SVC Linear': SVC(kernel='linear', probability=True),
    'SVC RBF': SVC(kernel='rbf', probability=True),
    'Random Forest (min_samples_leaf=1)': RandomForestClassifier(min_samples_leaf=1),
    'Random Forest (min_samples_leaf=3)': RandomForestClassifier(min_samples_leaf=3),
    'Random Forest (min_samples_leaf=5)': RandomForestClassifier(min_samples_leaf=5),
    'Neural Network (hidden_layer_sizes=(5,))': MLPClassifier(hidden_layer_sizes=(5,)),
    'Neural Network (hidden_layer_sizes=(5,5))': MLPClassifier(hidden_layer_sizes=(5,5)),
    'Neural Network (hidden_layer_sizes=(5,5,5))': MLPClassifier(hidden_layer_sizes=(5,5,5)),
    'Neural Network (hidden_layer_sizes=(10,))': MLPClassifier(hidden_layer_sizes=(10,))
}

for i, (data, label) in enumerate(datasets):
    print(f"Classification results for {label}:\n")
    x_train, x_test, y_train, y_test = train_test_split(data[['x1','x2']], data['y'], test_size=0.2, random_state=42)
    
    for name, clf in classifiers.items():
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        acc = accuracy_score(y_test, y_pred) # model.score(x_test, y_test)
        print(f"{name} Accuracy: {acc:.2f}, {clf.score(x_test, y_test):.2f}")
        print(classification_report(y_test, y_pred))

        plt.figure(figsize=(16, 10))
        draw_decision_boundary(clf, x_train.values, y_train.values, size=10)
        plt.title(f'Decision Boundary for {name} on {label}')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.tight_layout()
        plt.savefig(f'Images/dataset-{i+1}-{name}-decision-boundary.png', dpi=400)
        plt.show()
    print("\n")

def get_metrics(y_true, y_pred, y_prob):
    report = classification_report(y_true, y_pred, output_dict=True)
    acc = accuracy_score(y_true, y_pred)
    
    precisions = [report[f'{i+1}']['precision'] for i in range(4)]
    recalls = [report[f'{i+1}']['recall'] for i in range(4)]
    f1s = [report[f'{i+1}']['f1-score'] for i in range(4)]
    
    aucs = roc_auc_score(pd.get_dummies(y_true), y_prob, multi_class='ovr', average=None)
    
    precision_avg = sum(precisions) / 4
    recall_avg = sum(recalls) / 4
    f1_avg = sum(f1s) / 4
    auc_avg = sum(aucs) / 4
    
    return [acc, *precisions, precision_avg, *recalls, recall_avg, *f1s, f1_avg, *aucs, auc_avg]

def get_metrics_df(data, classifiers):
    x_train, x_test, y_train, y_test = train_test_split(data[['x1', 'x2']], data['y'], test_size=0.2, random_state=42)
    metrics = []
    
    for name, clf in classifiers.items():
        clf.fit(x_train, y_train)
        
        y_train_pred = clf.predict(x_train)
        if hasattr(clf, 'predict_proba'):
            y_train_prob = clf.predict_proba(x_train)
        else:
            decision_values = clf.decision_function(x_train)
            y_train_prob = (decision_values - decision_values.min()) / (decision_values.max() - decision_values.min())
        
        train_metrics = get_metrics(y_train, y_train_pred, y_train_prob)
        
        y_test_pred = clf.predict(x_test)
        if hasattr(clf, 'predict_proba'):
            y_test_prob = clf.predict_proba(x_test)
        else:
            decision_values = clf.decision_function(x_test)
            y_test_prob = (decision_values - decision_values.min()) / (decision_values.max() - decision_values.min())
        
        test_metrics = get_metrics(y_test, y_test_pred, y_test_prob)
        
        metrics.append([name, 'train', *train_metrics])
        metrics.append([name, 'test', *test_metrics])
    
    columns = ['algorithm_name', 'train_or_test_data', 'accuracy',
               'precision_1', 'precision_2', 'precision_3', 'precision_4', 'precision_avg',
               'recall_1', 'recall_2', 'recall_3', 'recall_4', 'recall_avg',
               'f1_1', 'f1_2', 'f1_3', 'f1_4', 'f1_avg',
               'auc_1', 'auc_2', 'auc_3', 'auc_4', 'auc_avg']
    
    ret = pd.DataFrame(metrics, columns=columns)
    
    return ret

for i, (data, label) in enumerate(datasets):
    metrics_df = get_metrics_df(data, classifiers)
    metrics_df.to_csv(f'Metrics/metrics-{i+1}.csv', index=False)

def draw_roc_curve(model, X, y, label):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=['x1', 'x2'])

    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X)
    else:
        y_prob = model.decision_function(X)
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())

    fpr, tpr, _ = roc_curve(pd.get_dummies(y).values.ravel(), y_prob.ravel())
    auc_score = roc_auc_score(pd.get_dummies(y), y_prob, multi_class='ovr')

    plt.plot(fpr, tpr, label=f'{label} (AUC = {auc_score:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {label}')
    plt.legend()

def draw_plots(classifiers, datasets):
    for i, (data, label) in enumerate(datasets):
        plt.figure(figsize=(18, 15))
        
        for row, (name, clf) in enumerate(classifiers.items()):
            x_train, x_test, y_train, y_test = train_test_split(data[['x1', 'x2']], data['y'], test_size=0.2, random_state=42)
            
            x_train_df = pd.DataFrame(x_train, columns=['x1', 'x2'])
            x_test_df = pd.DataFrame(x_test, columns=['x1', 'x2'])
            y_train = pd.Series(y_train)
            
            clf.fit(x_train_df, y_train.values)

            plt.subplot(len(classifiers)//2, 3, (row % 5) * 3 + 1)
            draw_decision_boundary(clf, x_train_df.values, y_train.values, size=5, edgecolor=None)
            plt.title(f'{name} - {label}')
            plt.xlabel('x1')
            plt.ylabel('x2')

            plt.subplot(len(classifiers)//2, 3, (row % 5) * 3 + 2)
            draw_roc_curve(clf, x_train_df, y_train, 'Train')

            plt.subplot(len(classifiers)//2, 3, (row % 5) * 3 + 3)
            draw_roc_curve(clf, x_test_df, y_test, 'Test')

            if row == 4:
                plt.tight_layout()
                plt.savefig(f'Images/dataset-{i+1}-roc-curves-1.png', dpi=400)
                plt.show()
                plt.figure(figsize=(18, 15))


        plt.tight_layout()
        plt.savefig(f'Images/dataset-{i+1}-roc-curves-2.png', dpi=400)
        plt.show()

draw_plots(classifiers, datasets)

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from imblearn.over_sampling import SMOTE
rng_seed = 42

#data = pd.read_csv("EmployeeAttritionAllData.csv")         ### Unfiltered data, not using for now
data = pd.read_csv("EmployeeAttrition.csv")
### checking the data to make sure all data are valid to use for training
#print(data.head()) 
#print(data.info())
#print(data.isnull().sum())


### One Hot Encode categorical data
categorical_columns = data.select_dtypes(include=['object']).columns
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)  ###drop_first avoid redundency


### Test Train Split before Normalization to ensure no data leakage
X = data.drop(['Attrition_Yes'], axis=1)                #Attrition converted to Yes from OHE
y = data['Attrition_Yes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rng_seed)

### Scaling numerical Data
scaler = StandardScaler()
numerical_columns = X_train.select_dtypes(include=['int64']).columns            ### Only normalize numerical values
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns]) 
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])         ### Make sure to normalize the x_train with same param as x_test

'''
# Balance data
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
'''

### Simple Logistic Regression Model Training
simple_model = LogisticRegression(solver='liblinear', random_state=rng_seed)
simple_model.fit(X_train, y_train)
simple_yhat = simple_model.predict_proba(X_test)[:, 1]


### Evaluate model with cross validation
acc_norm = np.mean(cross_val_score(simple_model, X_train, y_train, scoring='accuracy', cv=3))



### Lasso Regularized Logistic Regression
C = np.logspace(-2,1,20)
acc = np.empty(40)
models = list()

for c in range(len(C)):   
    
    model = Pipeline([('scaler', StandardScaler()), 
                 ('logreg', LogisticRegression(C=C[c],
                   penalty='l1',
                   solver='liblinear',
                   random_state=rng_seed))])
    model.fit(X_train,y_train)
    
    models.append(model)

    # Validation accuracy
    acc[c] = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy').mean()


# Find best model
cstar = np.argmax(acc)
print(f"best Lasso model index: {cstar}")
# Find best model's accuracy
acc_star = acc[cstar]
print(f"best Lasso model accuracy: {acc_star}")

theta = np.vstack([model.named_steps['logreg'].coef_[0,:] for model in models])
theta_star = theta[cstar,:]
print(f"Regularized weights: {theta_star}")
lasso_model = models[cstar]


yhat = lasso_model.predict(X_test)
lasso_test = accuracy_score(y_test, yhat)
print(f"Accuracy for Regularized Model: {lasso_test}")


###Poly Features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)


# Train a logistic regression model on the new dataset
poly_model = LogisticRegression(random_state=42)
poly_model.fit(X_train_poly, y_train)
poly_yhat = poly_model.predict(X_test_poly)
poly_test = accuracy_score(y_test, poly_yhat)
print(f"Accuracy for Poly Model: {poly_test}")


### Testing / Accuracy for 50% threshold
simple_ypred = (simple_yhat > 0.5).astype(int)
accuracy = accuracy_score(y_test, simple_ypred)
print(f"Unregularized Accuracy: {accuracy}")

'''
#Classification Report
print("Classification Report:")
print(classification_report(y_test, yhat))
'''

fpr, tpr, th = roc_curve(y_test, simple_yhat) 
auc = roc_auc_score(y_test, simple_yhat)

lw = 2
plt.plot(fpr, tpr, color='blue', lw=lw, label='AUC = %0.2f' % auc)
plt.plot([0, 1], [0, 1], color='red', lw=lw, linestyle='--') 
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

print("AUC-ROC Score:", auc)
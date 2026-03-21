import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,ConfusionMatrixDisplay
import pickle
fig,ax=plt.subplots(3,3,figsize=(9,9))
df=pd.read_csv("/storage/emulated/0/Download/Work1/EcommerceClean.csv")
le=LabelEncoder()
df["Gender"]=le.fit_transform(df["Gender"])
df["ProductCategory"]=le.fit_transform(df["ProductCategory"])
df["State"]=le.fit_transform(df["State"])
X=df[["Age","Gender","State","ProductCategory","TotalSpent(NGN)","NumOrders","DaysSinceLastOrder","NumComplaints","DiscountsUsed","SatisfactionScore","DeliveryDays","NumReturns","LoyaltyYears","SatisfactionIndex","LoyaltyIndex"]]
y=df["Churned"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=7)
#logistic_regression
model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
classification=classification_report(y_test,y_pred)
print(f"Accuracy(LogisticRegression): {accuracy:.4f}")
print(f"ClassificationReport(LogisticRegression): {classification}")
cm=confusion_matrix(y_test,y_pred)
disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=("No","Yes"))
disp.plot(ax=ax[0,0],cmap="Blues")
ax[0,0].set_title("LogisticRegression")
#Decision_Tree
model1=DecisionTreeClassifier(max_depth=5,random_state=7)
model1.fit(X_train,y_train)
y_pred1=model1.predict(X_test)
accuracy1=accuracy_score(y_test,y_pred1)
classification1=classification_report(y_test,y_pred1)
print(f"Accuracy(DecisionTree): {accuracy1:.4f}")
print(f"Classication(DecisionTree): {classification1}")
cm1=confusion_matrix(y_test,y_pred1)
disp1=ConfusionMatrixDisplay(confusion_matrix=cm1,display_labels=("No","Yes"))
disp1.plot(ax=ax[0,1],cmap="Blues")
ax[0,1].set_title("DecisionTree")
#RandomForest
model2=RandomForestClassifier(n_estimators=120,random_state=7)
model2.fit(X_train,y_train)
y_pred2=model2.predict(X_test)
accuracy2=accuracy_score(y_test,y_pred2)
classification2=classification_report(y_test,y_pred2)
print(f"Accuracy(RandomForest): {accuracy2:.4f}")
print(f"Classication(RandomForest): {classification2}")
cm2=confusion_matrix(y_test,y_pred2)
disp2=ConfusionMatrixDisplay(confusion_matrix=cm2,display_labels=("No","Yes"))
disp2.plot(ax=ax[0,2],cmap="Blues")
ax[0,2].set_title("RandomForest")
importance=model2.feature_importances_
features=X.columns
im_df=pd.DataFrame({"Feature":features, "Importance":importance})
im_df=im_df.sort_values(by="Importance")
ax[1,0].barh(im_df["Feature"],im_df["Importance"])
#fineTuneForest(highestAccuracy)
params={"n_estimators":[50,80,150],"min_samples_split":[3,7,10,15,17],"max_depth":[3,7,10,13,15]}
grid=GridSearchCV(RandomForestClassifier(random_state=7),params,cv=7,scoring="accuracy",verbose=1)
grid.fit(X_train,y_train)
print(f"accuracyscore {grid.best_score_:.4f}")
print(f"bestparameter: {grid.best_params_}")
y_pred3=grid.best_estimator_.predict(X_test)
accuracy3=accuracy_score(y_test,y_pred3)
classification3=classification_report(y_test,y_pred3)
print(f"FinetunnedRandomForest(accuracy): {accuracy3:.4f}")
print(f"FinetunnedRandomForest(classifcationreport): {classification3}")
cm3=confusion_matrix(y_test,y_pred3)
disp3=ConfusionMatrixDisplay(confusion_matrix=cm3,display_labels=("No","Yes"))
disp3.plot(ax=ax[1,2],cmap="Blues")
ax[1,2].set_title("FineTunedRandomForest")
#best(RandonForrest)
pickle.dump(model2,open("/storage/emulated/0/Download/Work1/model2.pkl","wb"))
plt.show()
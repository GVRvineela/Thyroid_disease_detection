import pandas as pd
import numpy as np
# pip install scikit-learn
# pip install xgboost
# pip install imbalanced-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from  xgboost import XGBClassifier
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import pickle
allHyperTest = pd.read_csv("allhyperTestEDIT.CSV")
allHyperTrain = pd.read_csv("allhyperTrainEDIT.CSV")
allHypoTest = pd.read_csv("allhypoTEST.csv")
allHypoTrain = pd.read_csv("allhypoDATA.CSV")
from IPython.display import display
display(allHypoTest.head(10))
display(allHypoTrain.dtypes)
def handleDuplicated(df):
    if df["ID"].duplicated().sum() == 0 :
        print("There aren't duplicates")
    elif (df["ID"].duplicated().sum()) < len(df) / 100:
        df["ID"].drop_duplicates(keep="first", inplace=True)
        print("duplicates were less than the 1% of all the data, they have been dropped")
    else:
        index_duplicated = df["ID"].duplicated().index
        print("duplicates are more than the 1% of all the data, they have been preserved")
        print(index_duplicated)

handleDuplicated(allHyperTest)
handleDuplicated(allHyperTrain)
handleDuplicated(allHypoTest)
handleDuplicated(allHypoTrain)
del allHyperTest["ID"]
del allHyperTrain["ID"]
del allHypoTest["ID"]
del allHypoTrain["ID"]
del allHyperTest["TBG"]
del allHyperTrain["TBG"]
del allHypoTest["TBG"]
del allHypoTrain["TBG"]
del allHyperTest["TBG_measured"]
del allHyperTrain["TBG_measured"]
del allHypoTest["TBG_measured"]
del allHypoTrain["TBG_measured"]
del allHyperTest["psych"]
del allHyperTrain["psych"]
del allHypoTest["psych"]
del allHypoTrain["psych"]
del allHyperTest["hypopituitary"]
del allHyperTrain["hypopituitary"]
del allHypoTest["hypopituitary"]
del allHypoTrain["hypopituitary"]
# del allHyperTest["tumor"]
# del allHyperTrain["tumor"]
# del allHypoTest["tumor"]
# del allHypoTrain["tumor"]
# del allHyperTest["goitre"]
# del allHyperTrain["goitre"]
# del allHypoTest["goitre"]
# del allHypoTrain["goitre"]
del allHyperTest["lithium"]
del allHyperTrain["lithium"]
del allHypoTest["lithium"]
del allHypoTrain["lithium"]
del allHyperTest["query_hyperthyroid"]
del allHyperTrain["query_hyperthyroid"]
del allHypoTest["query_hyperthyroid"]
del allHypoTrain["query_hyperthyroid"]
del allHyperTest["I131_treatment"]
del allHyperTrain["I131_treatment"]
del allHypoTest["I131_treatment"]
del allHypoTrain["I131_treatment"]
# del allHyperTest["thyroid_surgery"]
# del allHyperTrain["thyroid_surgery"]
# del allHypoTest["thyroid_surgery"]
# del allHypoTrain["thyroid_surgery"]
# del allHyperTest["sick"]
# del allHyperTrain["sick"]
# del allHypoTest["sick"]
# del allHypoTrain["sick"]
del allHyperTest["on_antithyroid_medication"]
del allHyperTrain["on_antithyroid_medication"]
del allHypoTest["on_antithyroid_medication"]
del allHypoTrain["on_antithyroid_medication"]
del allHyperTest["query_on_thyroxine"]
del allHyperTrain["query_on_thyroxine"]
del allHypoTest["query_on_thyroxine"]
del allHypoTrain["query_on_thyroxine"]
del allHyperTest["on_thyroxine"]
del allHyperTrain["on_thyroxine"]
del allHypoTest["on_thyroxine"]
del allHypoTrain["on_thyroxine"]
del allHyperTest["query_hypothyroid"]
del allHyperTrain["query_hypothyroid"]
del allHypoTest["query_hypothyroid"]
del allHypoTrain["query_hypothyroid"]
allHyperTest
def notCorrect_TargetFilter(df,correct_Target,target):
    df = df[df.Target.isin(correct_Target)]
    df.replace(correct_Target,target,inplace = True)
    return df
    
allHyperTest = notCorrect_TargetFilter(allHyperTest,["hyperthyroid","T3_toxic","goitre","secondary_toxic"],"hyperthyroid")
allHyperTrain = notCorrect_TargetFilter(allHyperTrain,["hyperthyroid","T3_toxic","goitre","secondary_toxic"],"hyperthyroid")
allHypoTest = notCorrect_TargetFilter(allHypoTest,["hypothyroid", "primary_hypothyroid", "compensated_hypothyroid", "secondary_hypothyroid"],"hypothyroid")
allHypoTrain = notCorrect_TargetFilter(allHypoTrain,["hypothyroid", "primary_hypothyroid", "compensated_hypothyroid", "secondary_hypothyroid"],"hypothyroid")
allDataset = pd.concat([allHyperTest,allHyperTrain,allHypoTest,allHypoTrain], ignore_index = True)
display(allDataset.shape)
allDataset
thyroid0387 = pd.read_csv("thyroid0387EDIT.CSV")
display(thyroid0387.head(10))
display(thyroid0387.dtypes)
handleDuplicated(thyroid0387)
del thyroid0387["ID"]
del thyroid0387["TBG"]
del thyroid0387["TBG_measured"]
del thyroid0387["psych"]
del thyroid0387["hypopituitary"]
#del thyroid0387["tumor"]
#del thyroid0387["goitre"]
del thyroid0387["lithium"]
del thyroid0387["query_hyperthyroid"]
del thyroid0387["I131_treatment"]
#del thyroid0387["thyroid_surgery"]
#del thyroid0387["sick"]
del thyroid0387["on_thyroxine"]
del thyroid0387["query_on_thyroxine"]
del thyroid0387["query_hypothyroid"]
del thyroid0387["on_antithyroid_medication"]
thyroid0387
thyroid0387['sex'] = thyroid0387['sex'].map({'F': 1, 'M': 0})

thyroid0387.replace(['A', 'B', 'C', 'D'], "hyperthyroid", inplace=True)
thyroid0387.replace(['E', 'F', 'G', 'H'], "hypothyroid", inplace=True)

for value in set(thyroid0387['Target']):
    if value != 'hypothyroid' and value != 'hyperthyroid':
        thyroid0387.replace(value, 'negative', inplace=True)
thyroid0387
hypothyroid = pd.read_csv("hypothyroid.csv")
display(hypothyroid.shape)
display(hypothyroid.head(10))
display(hypothyroid.dtypes)
hypothyroid = hypothyroid.rename(columns={hypothyroid.columns[0]:"Target",hypothyroid.columns[1]:"age",hypothyroid.columns[2]:"sex" })
hypothyroid = hypothyroid[hypothyroid.Target.isin(['hypothyroid'])]
del hypothyroid["TBG"]
del hypothyroid["TBG_measured"]
#del hypothyroid["tumor"]
#del hypothyroid["goitre"]
del hypothyroid["lithium"]
del hypothyroid["query_hyperthyroid"]
#del hypothyroid["thyroid_surgery"]
#del hypothyroid["sick"]
del hypothyroid["on_thyroxine"]
del hypothyroid["query_on_thyroxine"]
del hypothyroid["query_hypothyroid"]
del hypothyroid["on_antithyroid_medication"]
hypothyroid
sick_euthyroid = pd.read_csv("sick-euthyroid.CSV")
display(sick_euthyroid.shape)
display(sick_euthyroid.head(10))
display(sick_euthyroid.dtypes)
sick_euthyroid = sick_euthyroid[sick_euthyroid.Target.isin(['negative'])]
display(sick_euthyroid.shape)
del sick_euthyroid["TBG"]
del sick_euthyroid["TBG_measured"]
#del sick_euthyroid["tumor"]
#del sick_euthyroid["goitre"]
del sick_euthyroid["lithium"]
del sick_euthyroid["query_hyperthyroid"]
#del sick_euthyroid["thyroid_surgery"]
#del sick_euthyroid["sick"]
del sick_euthyroid["on_thyroxine"]
del sick_euthyroid["query_on_thyroxine"]
del sick_euthyroid["query_hypothyroid"]
del sick_euthyroid["on_antithyroid_medication"]
sick_euthyroid
ann_train = pd.read_csv("ann-train.CSV")
ann_test = pd.read_csv("ann-test.CSV")
display(ann_test.head(10))
display(ann_test.dtypes)
del ann_train["I131_treatment"]
#del ann_train["tumor"]
#del ann_train["goitre"]
del ann_train["lithium"]
del ann_train["query_hyperthyroid"]
#del ann_train["thyroid_surgery"]
#del ann_train["sick"]
del ann_train["on_thyroxine"]
del ann_train["query_on_thyroxine"]
del ann_train["query_hypothyroid"]
del ann_train["on_antithyroid_medication"]
del ann_train["hypopituitary"]
del ann_train["psych"]
ann_train
del ann_test["I131_treatment"]
#del ann_test["tumor"]
#del ann_test["goitre"]
del ann_test["lithium"]
del ann_test["query_hyperthyroid"]
#del ann_test["thyroid_surgery"]
#del ann_test["sick"]
del ann_test["on_thyroxine"]
del ann_test["query_on_thyroxine"]
del ann_test["query_hypothyroid"]
del ann_test["on_antithyroid_medication"]
del ann_test["hypopituitary"]
del ann_test["psych"]
ann_test
target1 = pd.Series(ann_test[ann_test.columns[-1]].values)
display(target1.value_counts())
target2 = pd.Series(ann_train[ann_train.columns[-1]].values)
display(target2.value_counts())
#3 is referring to the 'negative' class
#2 is referring to the 'hypothyroid' class
# 1 is referring to the 'hyperthyroid' class
print("Sex thyroid0387 1=F,0=M:")
sex_series1 = pd.Series(thyroid0387[thyroid0387.columns[1]].values)
display(sex_series1.value_counts())
print("Sick-euthyroid:")
sex_series2 = pd.Series(sick_euthyroid[sick_euthyroid.columns[2]].values)
display(sex_series2.value_counts())
sex1 = pd.Series(ann_test[ann_test.columns[1]].values)
display(sex1.value_counts())
sex2 = pd.Series(ann_train[ann_train.columns[1]].values)
display(sex2.value_counts())
for column in ann_train.columns:
    listOfValues=set(ann_train[column])
    print(column,": ",listOfValues)
ann = pd.concat([ann_train,ann_test], ignore_index = True)
ann['sex'] = ann['sex'].map({0:'F',1:'M'})
ann['Target'] = ann['Target'].map({3:'negative',2:'hypothyroid',1:'hyperthyroid'})

continuos_attributes = ['age','TSH','T3','TT4','T4U','FTI']
for attribute in continuos_attributes:
    ann[attribute] = ann[attribute] * 100

def fillNewAttributes(row,attribute):
    if row[attribute] > 0:
        return 'y'
    else:
        return 'n'

ann['TSH_measured'] = ann.apply(lambda row: fillNewAttributes(row,'TSH'), axis=1)
ann['T3_measured'] = ann.apply(lambda row: fillNewAttributes(row,'T3'), axis=1)
ann['TT4_measured'] = ann.apply(lambda row: fillNewAttributes(row,'TT4'), axis=1)
ann['T4U_measured'] = ann.apply(lambda row: fillNewAttributes(row,'T4U'), axis=1)
ann['FTI_measured'] = ann.apply(lambda row: fillNewAttributes(row,'FTI'), axis=1)
display(ann.dtypes)
data = pd.concat([allDataset,thyroid0387,hypothyroid,sick_euthyroid,ann], ignore_index = True)
display(data.shape)
display(data.dtypes)
data
 # data preprocessing
for column in data.columns:
    listOfValues=set(data[column])
    print(column,": ",listOfValues)
data=data.replace({"?":np.NAN})
data.isna().sum()
del data['referral_source']
del data['sex']
data
data = data.replace({"t":1,"f":0, "y":1, "n":0, "hypothyroid":1, "negative":0,"hyperthyroid":2, "F":1, "M":0})
display(data.dtypes)
cols = data.columns[data.dtypes.eq('object')]
data[cols] = data[cols].apply(pd.to_numeric, errors='coerce')
display(data.dtypes)
corr_values = abs(data[data.columns[0:]].corr()['Target'][:])
corr_values = corr_values.drop('Target')
corr_values = corr_values[corr_values > 0.04]
display(corr_values)
def holdout(dataframe):
    x = dataframe[corr_values.index]
    y = dataframe['Target']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=50) 
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = holdout(data)
classifiers = {
    "XGBClassifier" : XGBClassifier(learning_rate=0.01),
}
def classification(classifiers, X_train, X_test, y_train, y_test):
    #res as a DataFrame
    res = pd.DataFrame(columns=["Classifier", "Accuracy", "Precision", "Recall", "FScore"])
    
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        pr, rc, fs, sup = metrics.precision_recall_fscore_support(y_test, y_pred, average='macro')
        
        # loc to append a row to the DataFrame
        res.loc[len(res)] = {
            "Classifier": name,
            "Accuracy": round(metrics.accuracy_score(y_test, y_pred), 4),
            "Precision": round(pr, 4),
            "Recall": round(rc, 4),
            "FScore": round(fs, 4)
        }
        
        print("Confusion matrix for: ", name)
        display(confusion_matrix(y_test, y_pred))
    
    res.set_index("FScore", inplace=True)
    res.sort_values(by="FScore", ascending=False, inplace=True)
    return res

# Call the function
result = classification(classifiers, X_train, X_test, y_train, y_test)

# label_mapping = {0: "negative", 1: "hypothyroid", 2: "hyperthyroid"}
# for clf_name, clf in classifiers.items():
#     clf.classes_ = [label_mapping[label] for label in clf.classes_]


display(result)
display(data.shape)
data.Target.value_counts()
#0 is negative
#1 is hypothyroid
#2 is hyperthyroid
classifiers = {
    "XGBClassifier": XGBClassifier(learning_rate=0.01)
}

# train-test split
X = data.drop("Target", axis=1)  # Assuming "Target" is the target variable
y = data["Target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

#classification function
result = classification(classifiers, X_train, X_test, y_train, y_test)

# Map numerical labels to string labels
# label_mapping = {0: "negative", 1: "hypothyroid", 2: "hyperthyroid"}
# for clf_name, clf in classifiers.items():
#     clf.classes_ = [label_mapping[label] for label in clf.classes_]

# input details from the user for new testing set
# input_data = {}
# for column in X.columns:
#     value = input(f"Enter value for {column}: ")
#     input_data[column] = [float(value)]

# DataFrame
# user_input_df = pd.DataFrame(input_data)

# user_predictions = classifiers["XGBClassifier"].predict(user_input_df)
# Display the user input and the predicted class
# print("\nUser Input:")
# display(user_input_df)
# print("\nPredicted Class:", user_predictions[0])
# print("\nUser Input:")
# display(user_input_df)

# class_labels = {0: "Negative", 1: "Hypothyroid", 2: "Hyperthyroid"}

# user_predictions = classifiers["XGBClassifier"].predict(user_input_df)

# predicted_label = class_labels[user_predictions[0]]

# Display
# print("\n MOST EXPECTED RESULT:", predicted_label)

pickle.dump(classifiers,open("model.pkl","wb"))
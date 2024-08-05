import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
import pickle as pickle


def get_clean_data():
     data = pd.read_csv("data/data.csv")
     data = data.drop(['Unnamed: 32','id'],axis=1)
     #ENCODING THE DIAGNOSIS COLUMN
     data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})
     return data 


def create_model(data):
    X_features = data.drop(['diagnosis'],axis=1)
    target = data['diagnosis']
    #feature scaling the data
    scaler = MinMaxScaler()
    X_features = scaler.fit_transform(X_features)
    #splitting the data into train test split
    x_train,x_test,y_train,y_test = train_test_split(X_features,target,test_size=0.2,random_state=42)
    #creating the model and training the model
    model = LogisticRegression()
    model.fit(x_train,y_train)
    return model,x_test,y_test,scaler


def evaluate_model(model,x_test,y_test):
    y_pred = model.predict(x_test)
    print("Accuracy of the model",accuracy_score(y_test,y_pred))
    print("Classification report",classification_report(y_test,y_pred))
    return model.score(x_test,y_test)    



def main():
    #data cleaning
    data = get_clean_data()
    #training the model
    model,x_test,y_test,scaler = create_model(data)
    #evaluating the model
    score = evaluate_model(model,x_test,y_test)
    #saving the model as binary file
    with open('model/model.pkl','wb') as file:
        pickle.dump(model,file)
    with open('model/scaler.pkl','wb') as file1:
        pickle.dump(scaler,file1)
    with open('model/data.pkl','wb') as file2:
        pickle.dump(data,file2)        


if __name__ == '__main__':
    main()    
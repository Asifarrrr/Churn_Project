import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
#------------------------------------------- (Data Cleaning) ----------------------------------------------
df = None
try:
    df = pd.read_csv(r"C:\Users\pc\PycharmProjects\WelcomeScreen\Mini Projects\Churn_Project\data\customer churn.csv")
except FileNotFoundError:
    pass

df = df.drop(columns = 'Customer_ID')
df = df.drop_duplicates()

strings = ['Gender','Region','Subscription_Plan','Payment_Method']
for col in strings:
    df[col] = df[col].str.strip().str.lower()

df.loc[(df['Age'] <= 10) | (df['Age'] >= 101) ,'Age'] = None
df['Age'] = df['Age'].fillna(df['Age'].median()).astype(int)

df.loc[df['Customer_Service_Calls'] == -1,'Customer_Service_Calls'] = 0

df.loc[(df['Monthly_Spend'] < 0) | (df['Monthly_Spend'] > 10000),'Monthly_Spend'] = None
df['Monthly_Spend'] = df['Monthly_Spend'].fillna(df['Monthly_Spend'].mean().round(2))

df = df.dropna(subset = ['Payment_Method'])

df = pd.get_dummies(df,columns = ['Gender','Region','Subscription_Plan','Payment_Method'],dtype = int)

df = df.rename(columns = {
    'Gender_non-binary': 'Gender_non_binary',
    'Payment_Method_e-wallet': 'Payment_Method_e_wallet',
    'Payment_Method_credit card': 'Payment_Method_credit_card'
})
df = df[df['Monthly_Spend'] <= 130]
#------------------------------------------- (Model Training) ----------------------------------------------
x = df.drop(columns = ['Churn'])
y = df['Churn']

x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size = 0.18,
    random_state = 42
)
model = RandomForestClassifier(
    n_estimators = 150,
    max_depth = 8,
    min_samples_leaf = 15,
    min_samples_split = 20,
    class_weight = 'balanced',
    random_state = 42
)
smote = SMOTE(random_state = 42)
x_train_sampled,y_train_sampled = smote.fit_resample(x_train,y_train)

model.fit(x_train_sampled,y_train_sampled)

joblib.dump(model,r"C:\Users\pc\PycharmProjects\WelcomeScreen\Mini Projects\Churn_Project\data\churn_model.pkl")
joblib.dump(df,r"C:\Users\pc\PycharmProjects\WelcomeScreen\Mini Projects\Churn_Project\models\data_frame.pkl")
joblib.dump(list(x.columns),r"C:\Users\pc\PycharmProjects\WelcomeScreen\Mini Projects\Churn_Project\models\model_columns.pkl")
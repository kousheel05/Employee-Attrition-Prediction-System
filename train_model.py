import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------
# Load Dataset
# -----------------------

df = pd.read_csv("dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# -----------------------
# Keep only required columns
# -----------------------

selected_columns = [

    "Age",
    "BusinessTravel",
    "Department",
    "DistanceFromHome",
    "EnvironmentSatisfaction",
    "JobRole",
    "JobSatisfaction",
    "MonthlyIncome",
    "OverTime",
    "TotalWorkingYears",
    "WorkLifeBalance",
    "YearsAtCompany",
    "Attrition"

]

df = df[selected_columns]

# -----------------------
# Encode Target
# -----------------------

target_encoder = LabelEncoder()

df["Attrition"] = target_encoder.fit_transform(df["Attrition"])

joblib.dump(target_encoder, "models/target_encoder.pkl")

# -----------------------
# Encode categorical columns
# -----------------------

categorical_columns = df.select_dtypes(include="object").columns

encoders = {}

for column in categorical_columns:

    encoder = LabelEncoder()

    df[column] = encoder.fit_transform(df[column])

    encoders[column] = encoder

joblib.dump(encoders, "models/label_encoders.pkl")

# -----------------------
# Features & Target
# -----------------------

X = df.drop("Attrition", axis=1)

y = df["Attrition"]

# -----------------------
# Train Test Split
# -----------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y

)

# -----------------------
# Scaling
# -----------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

joblib.dump(scaler, "models/scaler.pkl")

# -----------------------
# Train Model
# -----------------------

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy :", round(accuracy*100,2), "%")

joblib.dump(model, "models/model.pkl")

print("\nModel Saved Successfully")
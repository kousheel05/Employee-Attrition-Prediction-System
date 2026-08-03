import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------------------
# Load Dataset
# -------------------------------

current_dir = os.path.dirname(__file__)

dataset_path = os.path.join(
    current_dir,
    "..",
    "dataset",
    "WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

df = pd.read_csv(dataset_path)

# -------------------------------
# Create output folder
# -------------------------------

output_folder = os.path.join(current_dir, "..", "static", "images")

os.makedirs(output_folder, exist_ok=True)

sns.set_style("whitegrid")
plt.figure(figsize=(6,5))

sns.countplot(x='Attrition', data=df)

plt.title("Employee Attrition Distribution")

plt.savefig(os.path.join(output_folder,"attrition_distribution.png"))

plt.show()

plt.figure(figsize=(8,5))

sns.countplot(x='Department', hue='Attrition', data=df)

plt.xticks(rotation=20)

plt.title("Department Wise Attrition")

plt.savefig(os.path.join(output_folder,"department_attrition.png"))

plt.show()

plt.figure(figsize=(6,5))

sns.countplot(x='Gender', data=df)

plt.title("Gender Distribution")

plt.savefig(os.path.join(output_folder,"gender_distribution.png"))

plt.show()

plt.figure(figsize=(8,5))

sns.histplot(df["MonthlyIncome"], bins=30, kde=True)

plt.title("Monthly Income Distribution")

plt.savefig(os.path.join(output_folder,"monthly_income.png"))

plt.show()

plt.figure(figsize=(8,5))

sns.histplot(df["Age"], bins=25, kde=True)

plt.title("Age Distribution")

plt.savefig(os.path.join(output_folder,"age_distribution.png"))

plt.show()

plt.figure(figsize=(6,5))

sns.countplot(x='OverTime', hue='Attrition', data=df)

plt.title("Overtime vs Attrition")

plt.savefig(os.path.join(output_folder,"overtime_attrition.png"))

plt.show()

plt.figure(figsize=(6,5))

sns.countplot(x='JobSatisfaction', data=df)

plt.title("Job Satisfaction")

plt.savefig(os.path.join(output_folder,"job_satisfaction.png"))

plt.show()

plt.figure(figsize=(16,12))

numeric_df = df.select_dtypes(include=['int64','float64'])

sns.heatmap(numeric_df.corr(), cmap='coolwarm')

plt.title("Correlation Heatmap")

plt.savefig(os.path.join(output_folder,"correlation_heatmap.png"))

plt.show()
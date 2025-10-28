import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv('student_lifestyle_dataset.csv')

# ------------------- Rule-Based AI 1 -------------------
def rule_based_ai1(row):
    if (row['Study_Hours_Per_Day'] >= 4 and 
        row['Sleep_Hours_Per_Day'] >= 6 and 
        row['GPA'] >= 3.0):
        return 1
    else:
        return 0

df['AI1_Pred'] = df.apply(rule_based_ai1, axis=1)

# ------------------- Rule-Based AI 2 -------------------
def rule_based_ai2(row):
    score = 0
    if row['Study_Hours_Per_Day'] >= 4: score += 2
    if row['Sleep_Hours_Per_Day'] >= 7: score += 1
    if row['Extracurricular_Hours_Per_Day'] <= 3: score += 1
    if row['Physical_Activity_Hours_Per_Day'] >= 1: score += 1
    if row['Social_Hours_Per_Day'] <= 4: score += 1
    if row['GPA'] >= 3.0: score += 2
    return 1 if score >= 6 else 0

df['AI2_Pred'] = df.apply(rule_based_ai2, axis=1)

# ------------------- Ground Truth -------------------
df['Actual'] = df['GPA'].apply(lambda x: 1 if x >= 3.0 else 0)

# ------------------- Evaluation -------------------
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
results = {'Metric': metrics, 'AI_1': [], 'AI_2': []}

for col in ['AI1_Pred', 'AI2_Pred']:
    acc = accuracy_score(df['Actual'], df[col])
    prec = precision_score(df['Actual'], df[col])
    rec = recall_score(df['Actual'], df[col])
    f1 = f1_score(df['Actual'], df[col])
    if col == 'AI1_Pred':
        results['AI_1'] = [acc, prec, rec, f1]
    else:
        results['AI_2'] = [acc, prec, rec, f1]

results_df = pd.DataFrame(results)
print(results_df)

# ------------------- Visualization -------------------
plt.figure(figsize=(8,5))
x = range(len(metrics))
plt.bar([i - 0.15 for i in x], results_df['AI_1'], width=0.3, label='Rule-Based AI 1')
plt.bar([i + 0.15 for i in x], results_df['AI_2'], width=0.3, label='Rule-Based AI 2')

plt.xticks(x, metrics)
plt.ylim(0,1)
plt.title('Performance Comparison of Rule-Based AI Models')
plt.ylabel('Score')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

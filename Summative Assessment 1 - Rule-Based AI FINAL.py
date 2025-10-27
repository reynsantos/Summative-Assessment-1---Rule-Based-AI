import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
df = pd.read_csv('star_classification.csv')

# Rule-based AI 1
def check_galaxy_ai1(row):
    # Example threshold for classification
    if row['u'] < 22 and row['g'] < 22 and row['r'] < 21 and row['redshift'] < 0.8:
        return 1  # Classify as GALAXY
    else:
        return 0  # Classify as non-GALAXY

# Apply the rule-based AI 1 to the dataset
df['ai_1_pred'] = df.apply(check_galaxy_ai1, axis=1)

# Rule-based AI 2 (using a different threshold combination)
def check_galaxy_ai2(row):
    # Example threshold for classification
    if row['z'] < 18 and row['i'] < 20 and row['redshift'] > 0.5:
        return 1  # Classify as GALAXY
    else:
        return 0  # Classify as non-GALAXY

# Apply the rule-based AI 2 to the dataset
df['ai_2_pred'] = df.apply(check_galaxy_ai2, axis=1)

# Evaluation for both AI models
accuracy_1 = accuracy_score(df['class'] == 'GALAXY', df['ai_1_pred'])
precision_1 = precision_score(df['class'] == 'GALAXY', df['ai_1_pred'])
recall_1 = recall_score(df['class'] == 'GALAXY', df['ai_1_pred'])
f1_1 = f1_score(df['class'] == 'GALAXY', df['ai_1_pred'])

accuracy_2 = accuracy_score(df['class'] == 'GALAXY', df['ai_2_pred'])
precision_2 = precision_score(df['class'] == 'GALAXY', df['ai_2_pred'])
recall_2 = recall_score(df['class'] == 'GALAXY', df['ai_2_pred'])
f1_2 = f1_score(df['class'] == 'GALAXY', df['ai_2_pred'])

# Print the evaluation results for both AI models
print('AI Model 1 Evaluation:')
print(f'Accuracy: {accuracy_1:.4f}')
print(f'Precision: {precision_1:.4f}')
print(f'Recall: {recall_1:.4f}')
print(f'F1 Score: {f1_1:.4f}\n')

print('AI Model 2 Evaluation:')
print(f'Accuracy: {accuracy_2:.4f}')
print(f'Precision: {precision_2:.4f}')
print(f'Recall: {recall_2:.4f}')
print(f'F1 Score: {f1_2:.4f}\n')

import matplotlib.pyplot as plt
import numpy as np

# Data for evaluation results
ai_models = ['AI Model 1', 'AI Model 2']
accuracy = [0.4526, 0.4045]
precision = [0.5676, 0.3653]
recall = [0.3324, 0.0024]
f1_score = [0.4193, 0.0047]

# Bar Width
bar_width = 0.2
index = np.arange(len(ai_models))

# Plotting the first graph (Accuracy, Precision, Recall, F1 Score for both AI models)
fig, ax = plt.subplots(figsize=(8, 6))

bar1 = ax.bar(index, accuracy, bar_width, label='Accuracy')
bar2 = ax.bar(index + bar_width, precision, bar_width, label='Precision')
bar3 = ax.bar(index + 2*bar_width, recall, bar_width, label='Recall')
bar4 = ax.bar(index + 3*bar_width, f1_score, bar_width, label='F1 Score')

ax.set_xlabel('AI Models')
ax.set_ylabel('Scores')
ax.set_title('Comparison of Performance Metrics (Accuracy, Precision, Recall, F1 Score)')
ax.set_xticks(index + 1.5*bar_width)
ax.set_xticklabels(ai_models)
ax.legend()

# Display the first graph
plt.tight_layout()
plt.show()

# Plotting the second graph (just for F1 Score comparison)
fig, ax = plt.subplots(figsize=(8, 6))

ax.bar(ai_models, f1_score, color=['blue', 'orange'])

ax.set_xlabel('AI Models')
ax.set_ylabel('F1 Score')
ax.set_title('F1 Score Comparison of AI Models')

# Display the second graph
plt.tight_layout()
plt.show()


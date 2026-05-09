import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,classification_report, recall_score, precision_score, f1_score, roc_auc_score

from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')
print('All libraries imported successfully! ')

df = pd.read_csv("C:/Users/Chinmay/OneDrive/Documents/creditcard.csv")

print(f'Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns')
df.head()

print('Missing values per column:')
print(df.isnull().sum().sum(), 'total missing values')  # should be 0
n_duplicates = df.duplicated().sum()
print(f'\nDuplicate rows: {n_duplicates} ({n_duplicates / len(df) * 100:.2f}%)')
df.drop_duplicates(inplace=True)
print(f'Rows after removing duplicates: {len(df):,}')

class_counts = df['Class'].value_counts()
fraud_pct = df['Class'].mean() * 100
print(f'Legitimate transactions : {class_counts[0]:,}')
print(f'Fraudulent transactions : {class_counts[1]:,}')
print(f'Fraud rate             : {fraud_pct:.3f}%')
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(['Legitimate (0)', 'Fraud (1)'], class_counts.values,color=['steelblue', 'tomato'], width=0.5)
ax.set_title('Transaction Class Distribution', fontsize=14)
ax.set_ylabel('Number of transactions')
for i, v in enumerate(class_counts.values):
    ax.text(i, v + 500, f'{v:,}', ha='center', fontsize=11)
plt.tight_layout()
plt.show()
print('\n The data is heavily imbalanced we must account for this!')
fraud     = df[df['Class'] == 1]
legit     = df[df['Class'] == 0]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(legit['Amount'],  bins=50, alpha=0.6, label='Legitimate', color='steelblue')
axes[0].hist(fraud['Amount'],  bins=50, alpha=0.8, label='Fraud',      color='tomato')
axes[0].set_xlim(0, 500)
axes[0].set_title('Transaction Amount')
axes[0].set_xlabel('Amount (€)')
axes[0].legend()
axes[1].hist(legit['V1'], bins=50, alpha=0.6, label='Legitimate', color='steelblue')
axes[1].hist(fraud['V1'], bins=50, alpha=0.8, label='Fraud',      color='tomato')
axes[1].set_title('Feature V1')
axes[1].legend()
axes[2].hist(legit['V4'], bins=50, alpha=0.6, label='Legitimate', color='steelblue')
axes[2].hist(fraud['V4'], bins=50, alpha=0.8, label='Fraud',      color='tomato')
axes[2].set_title('Feature V4')
axes[2].legend()
plt.suptitle('Feature Distributions: Legitimate vs Fraud', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()
print('Notice how the fraud distributions look very different the model will learn from these differences.')

def cap_outliers(dataframe, columns):
    """
    Cap outliers to [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    Values below the lower bound are set to the lower bound.
    Values above the upper bound are set to the upper bound.
    """
    df_capped = dataframe.copy()
    for col in columns:
        Q1  = df_capped[col].quantile(0.25)
        Q3  = df_capped[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_capped[col] = df_capped[col].clip(lower, upper)
    return df_capped
feature_cols = [c for c in df.columns if c != 'Class']
df_capped = cap_outliers(df, feature_cols)
print('Outlier capping done ')
print(f'Rows: {len(df_capped):,}  |  Columns: {df_capped.shape[1]}')
scaler = StandardScaler()
df_capped[['Amount', 'Time']] = scaler.fit_transform(df_capped[['Amount', 'Time']])
print('Scaling complete ')
print('Amount mean:', round(df_capped['Amount'].mean(), 4),'  std:', round(df_capped['Amount'].std(), 4))
X = df_capped.drop('Class', axis=1)
y = df_capped['Class']
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
print(f'Training set  : {len(X_train):,} rows  (fraud: {y_train.sum():,})')
print(f'Test set      : {len(X_test):,} rows   (fraud: {y_test.sum():,})')

def show_results(model_name, y_true, y_pred):
    """
    Print key metrics and plot the confusion matrix.
    
    Why these metrics instead of accuracy?
    - Recall     : of all actual frauds, how many did we catch?
    - Precision  : of all flagged transactions, how many were really fraud?
    - F1-score   : balance between recall and precision
    - ROC-AUC    : overall model discrimination ability
    """
    recall    = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred)

    print(f'\n===== {model_name} =====')
    print(f'  Recall    (fraud caught) : {recall}')
    print(f'  Precision               : {precision}')
    print(f'  F1-score                : {f1}')

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['Legitimate', 'Fraud'])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=12)
    plt.tight_layout()
    plt.show()

    return {'model': model_name, 'recall': recall, 'precision': precision, 'f1': f1}

all_results = []
print('Helper function ready ')

rf_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_base.fit(X_train, y_train)
rf_base_preds = rf_base.predict(X_test)

result = show_results('Random Forest (no resampling)', y_test, rf_base_preds)
all_results.append(result)

xgb_base = XGBClassifier(objective='binary:logistic', random_state=42,eval_metric='logloss', n_jobs=-1)
xgb_base.fit(X_train, y_train)
xgb_base_preds = xgb_base.predict(X_test)

result = show_results('XGBoost (no resampling)', y_test, xgb_base_preds)
all_results.append(result)

under = RandomUnderSampler(random_state=42)
X_under, y_under = under.fit_resample(X_train, y_train)

print(f'After undersampling: {len(X_under):,} rows')
print(f'  Legitimate : {(y_under == 0).sum():,}')
print(f'  Fraud      : {(y_under == 1).sum():,}')

rf_under = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_under.fit(X_under, y_under)
rf_under_preds = rf_under.predict(X_test)

result = show_results('Random Forest (undersampling)', y_test, rf_under_preds)
all_results.append(result)

xgb_under = XGBClassifier(objective='binary:logistic', random_state=42,eval_metric='logloss', n_jobs=-1)
xgb_under.fit(X_under, y_under)
xgb_under_preds = xgb_under.predict(X_test)

result = show_results('XGBoost (undersampling)', y_test, xgb_under_preds)
all_results.append(result)

smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)

print(f'After SMOTE: {len(X_smote):,} rows')
print(f'  Legitimate : {(y_smote == 0).sum():,}')
print(f'  Fraud      : {(y_smote == 1).sum():,}')

rf_smote = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_smote.fit(X_smote, y_smote)
rf_smote_preds = rf_smote.predict(X_test)

result = show_results('Random Forest (SMOTE)', y_test, rf_smote_preds)
all_results.append(result)

xgb_smote = XGBClassifier(objective='binary:logistic', random_state=42,eval_metric='logloss', n_jobs=-1)
xgb_smote.fit(X_smote, y_smote)
xgb_smote_preds = xgb_smote.predict(X_test)

result = show_results('XGBoost (SMOTE)', y_test, xgb_smote_preds)
all_results.append(result)

neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count

print(f'scale_pos_weight = {neg_count:,} / {pos_count:,} = {scale_pos_weight:.1f}')

xgb_weighted = XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=scale_pos_weight,   # ← key parameter
    random_state=42,
    eval_metric='logloss',
    n_jobs=-1
)
xgb_weighted.fit(X_train, y_train)
xgb_weighted_preds = xgb_weighted.predict(X_test)

result = show_results('XGBoost (scale_pos_weight)', y_test, xgb_weighted_preds)
all_results.append(result)

importances = pd.Series(xgb_smote.feature_importances_,index=X.columns).sort_values(ascending=False).head(15)

fig, ax = plt.subplots(figsize=(8, 5))
importances.plot(kind='barh', ax=ax, color='steelblue')
ax.invert_yaxis()
ax.set_title('Top 15 Most Important Features (XGBoost + SMOTE)', fontsize=13)
ax.set_xlabel('Feature importance score')
plt.tight_layout()
plt.show()

results_df = pd.DataFrame(all_results).set_index('model')
results_df = results_df.sort_values('recall', ascending=False)

print('\n===== Model Comparison =====')
print(results_df.round(3).to_string())

ax = results_df[['recall', 'precision', 'f1']].plot(kind='bar', figsize=(12, 5),color=['tomato', 'steelblue', 'seagreen'])
ax.set_title('Model Comparison: Recall, Precision, F1-Score', fontsize=13)
ax.set_ylabel('Score')
ax.set_ylim(0, 1.05)
ax.axhline(0.80, color='black', linestyle='--', linewidth=1, label='80% target')
ax.legend(loc='lower right')
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
plt.show()
import os, sys, warnings, json, pickle, glob
warnings.filterwarnings('ignore')
BASE = r'C:\Users\ivanq\Desktop\Talento tech\trabajo final'
sys.path.insert(0, BASE)

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib

OUT = os.path.join(BASE, 'models')
os.makedirs(OUT, exist_ok=True)

print("=== ENTRENANDO MODELOS ===")

# --- 1. RIDGE (Regresion Lineal) ---
print("\n[1/5] Ridge - Regresion Lineal...")
df_lr = pd.read_csv(os.path.join(BASE, 'suelos_regresion_lineal.csv'), nrows=None)
X_lr = df_lr.drop(columns=['IPS'])
y_lr = df_lr['IPS']
feature_names_lr = list(X_lr.columns)

scaler = StandardScaler()
X_lr_s = scaler.fit_transform(X_lr)

ridge = Ridge()
params = {'alpha': [0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(ridge, params, cv=3, scoring='r2', n_jobs=-1, verbose=0)
grid.fit(X_lr_s, y_lr)
print(f"  Mejor alpha: {grid.best_params_['alpha']}, R² CV: {grid.best_score_:.4f}")

joblib.dump(grid.best_estimator_, os.path.join(OUT, 'ridge.pkl'))
joblib.dump(scaler, os.path.join(OUT, 'scaler_ridge.pkl'))
with open(os.path.join(OUT, 'features_ridge.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(feature_names_lr))
print(f"  ridge.pkl + scaler_ridge.pkl guardados ({len(feature_names_lr)} features)")

# --- 2. LOGISTIC REGRESSION ---
print("\n[2/5] Regresion Logistica...")
df_log = pd.read_csv(os.path.join(BASE, 'suelos_regresion_logistica.csv'))
X_log = df_log.drop(columns=['prod_alta'])
y_log = df_log['prod_alta']
feature_names_log = list(X_log.columns)

scaler_log = StandardScaler()
X_log_s = scaler_log.fit_transform(X_log)

params = {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2'], 'solver': ['liblinear']}
grid = GridSearchCV(LogisticRegression(max_iter=2000, random_state=42), params, cv=3, scoring='roc_auc', n_jobs=-1, verbose=0)
grid.fit(X_log_s, y_log)
print(f"  Mejor C: {grid.best_params_['C']}, AUC CV: {grid.best_score_:.4f}")

joblib.dump(grid.best_estimator_, os.path.join(OUT, 'logistic.pkl'))
joblib.dump(scaler_log, os.path.join(OUT, 'scaler_logistic.pkl'))
with open(os.path.join(OUT, 'features_logistic.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(feature_names_log))
print(f"  logistic.pkl + scaler_logistic.pkl guardados")

# --- 3. RANDOM FOREST ---
print("\n[3/5] Random Forest - Clasificacion multiclase...")
df_tree = pd.read_csv(os.path.join(BASE, 'suelos_arboles_decision.csv'))
X_tree = df_tree.drop(columns=['calidad'])
y_tree = df_tree['calidad']
feature_names_tree = list(X_tree.columns)

le = LabelEncoder()
y_tree_enc = le.fit_transform(y_tree)
scaler_tree = StandardScaler()
X_tree_s = scaler_tree.fit_transform(X_tree)

rf = RandomForestClassifier(random_state=42, n_jobs=-1)
params = {'n_estimators': [50, 100], 'max_depth': [10, 15, None], 'min_samples_split': [2, 5]}
grid = GridSearchCV(rf, params, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=0)
grid.fit(X_tree_s, y_tree_enc)
print(f"  Mejores params: {grid.best_params_}, F1 CV: {grid.best_score_:.4f}")

joblib.dump(grid.best_estimator_, os.path.join(OUT, 'random_forest.pkl'))
joblib.dump(scaler_tree, os.path.join(OUT, 'scaler_rf.pkl'))
joblib.dump(le, os.path.join(OUT, 'label_encoder.pkl'))
with open(os.path.join(OUT, 'features_rf.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(feature_names_tree))
print(f"  random_forest.pkl + scaler_rf.pkl + label_encoder.pkl guardados")

# --- 4. K-MEANS ---
print("\n[4/5] K-Means Clustering...")
num_feats = ['pH','MO','fosforo','azufre','calcio','magnesio','potasio','CIC','CE','anios_est']
X_kmeans = df_lr[num_feats].copy()
scaler_k = StandardScaler()
X_k_s = scaler_k.fit_transform(X_kmeans)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X_k_s)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_k_s)

joblib.dump(kmeans, os.path.join(OUT, 'kmeans.pkl'))
joblib.dump(pca, os.path.join(OUT, 'pca.pkl'))
joblib.dump(scaler_k, os.path.join(OUT, 'scaler_kmeans.pkl'))
print(f"  kmeans.pkl + pca.pkl + scaler_kmeans.pkl guardados")
print(f"  Inercia: {kmeans.inertia_:.2f}, Componentes PCA: {pca.explained_variance_ratio_}")

# --- 5. DATOS LIGEROS PARA KNN ---
print("\n[5/5] Preparando datos para KNN...")
knn_sample = df_lr[num_feats + ['IPS']].sample(min(5000, len(df_lr)), random_state=42)
knn_sample.to_csv(os.path.join(OUT, 'knn_sample.csv'), index=False)
print(f"  knn_sample.csv guardado ({len(knn_sample)} filas)")

# DATOS PARA MAPA
csv_raw = glob.glob(os.path.join(BASE, 'Resultados_de_An*lisis_de_Laboratorio_Suelos_en_Colombia_20260526.csv'))[0]
df_raw = pd.read_csv(csv_raw, encoding='latin1')
cols_mapping = {0:'id',1:'fecha',2:'depto',3:'municipio',4:'cultivo',5:'estado',6:'tiempo_est',7:'topografia',8:'drenaje',9:'riego',10:'fertilizantes',11:'pH',12:'MO',13:'fosforo',14:'azufre',15:'acidez',16:'aluminio',17:'calcio',18:'magnesio',19:'potasio',20:'sodio',21:'CIC',22:'CE',23:'Fe_olsen',24:'Cu',25:'Mn_olsen',26:'Zn_olsen',27:'B',28:'Fe_doble',29:'Cu_doble',30:'Mn_doble',31:'Zn_doble'}
df_raw.columns = [cols_mapping[i] for i in range(len(df_raw.columns))]
df_raw['depto'] = df_raw['depto'].str.strip().str.title()
depto_ips = df_raw.groupby('depto').agg(IPS_medio=('pH', lambda x: np.nanmean(pd.to_numeric(x, errors='coerce')))).to_dict()['IPS_medio']
depto_dict = dict(sorted(depto_ips.items(), key=lambda x: x[1], reverse=True))
with open(os.path.join(OUT, 'depto_ips.json'), 'w', encoding='utf-8') as f:
    json.dump(depto_dict, f, ensure_ascii=False)
print(f"  depto_ips.json guardado ({len(depto_dict)} departamentos)")

# RESUMEN
outputs = ['ridge.pkl', 'scaler_ridge.pkl', 'features_ridge.txt',
           'logistic.pkl', 'scaler_logistic.pkl', 'features_logistic.txt',
           'random_forest.pkl', 'scaler_rf.pkl', 'label_encoder.pkl', 'features_rf.txt',
           'kmeans.pkl', 'pca.pkl', 'scaler_kmeans.pkl',
           'knn_sample.parquet', 'depto_ips.json']
print("\n=== ARCHIVOS GENERADOS ===")
for f in outputs:
    path = os.path.join(OUT, f)
    if os.path.exists(path):
        print(f"  {os.path.getsize(path):>10,} bytes  {f}")
print("\n=== ENTRENAMIENTO COMPLETADO ===")

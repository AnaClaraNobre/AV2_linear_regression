import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan


df = pd.read_csv("dataset_1.csv")

# PARTE 1:
print("Dados basicos:\n", df.describe())

print("\nMedianas:\n", df.median(numeric_only=True))

print("\nValores ausentes:\n", df.isnull().sum())

# FIM DA PARTE 1 -----------------------------------------

# PARTE 2:
df_clean = df.dropna()
df_encoded = pd.get_dummies(df_clean, drop_first=True)

# 2.2 Ajuste um modelo de regressão linear múltipla considerando:    
y = df_encoded['tempo_resposta']
X = df_encoded.drop(columns=['tempo_resposta'])

X = sm.add_constant(X)

X = X.astype(float)
y = y.astype(float)

model = sm.OLS(y, X).fit()

#2.3. Informe (de acordo com as técnicas, abordagens e testes vistos em sala de aula): 
print(model.summary())

# #2.5. Faça o diagnóstico de multicolinearidade: 
vif_data = pd.DataFrame()
vif_data["Variável"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)

# #2.6. Faça o diagnóstico de heterocedasticidade: 
residuos = model.resid
valores_ajustados = model.fittedvalues

plt.figure(figsize=(10, 5))
sns.scatterplot(x=valores_ajustados, y=residuos)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Valores ajustados")
plt.ylabel("Resíduos")
plt.title("Resíduos vs Valores Ajustados")
plt.show()

bp_test = het_breuschpagan(residuos, X)
labels = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
bp_result = dict(zip(labels, bp_test))
print("\nResultado do Teste de Breusch-Pagan:")
for k, v in bp_result.items():
    print(f"{k}: {v:.4f}")

#FIM DA PARTE 2 -------------------


#PARTE 3 – MODELO 2 (sem latência e armazenamento) =====

X_reduced = X.drop(columns=['latencia_ms', 'armazenamento_tb'])

model2 = sm.OLS(y, X_reduced).fit()

print("\n=== MODELO 2: Sem 'latencia_ms' e 'armazenamento_tb' ===")
print(model2.summary())

#FIM DA PARTE 3 -------------------


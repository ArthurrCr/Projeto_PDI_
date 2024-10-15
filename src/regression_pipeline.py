import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def preprocess_data(df, target_column, columns_to_treat):
    # Cria uma cópia
    df = df.copy()
    
    # Converter data para o formato datetime
    df['data'] = pd.to_datetime(df['data'], format='%Y-%m-%d', errors='coerce')
    
    # Criar as colunas com dia, mês e ano
    df['dia'] = df['data'].dt.day
    df['mes'] = df['data'].dt.month
    df['ano'] = df['data'].dt.year
    
    # Converter hora para string se não for float
    if df['hora'].dtype != float:
        df['hora'] = df['hora'].astype(str)
        # Verificar e ajustar o formato da hora
        df['hora'] = df['hora'].apply(lambda x: x.replace(':', '') if ':' in x else x)
        # Converter hora para inteiro e normalizar
        df['hora'] = df['hora'].astype(int)
        df['hora'] = df['hora'] / 10000
    
    # Calcular e imprimir a porcentagem de valores <= 0 antes do tratamento
    print("Porcentagem de valores <= 0 antes do tratamento:")
    for col in columns_to_treat:
        if col in df.columns:
            total = df[col].shape[0]
            count_le_zero = (df[col] <= 0).sum()
            percentage_le_zero = (count_le_zero / total) * 100
            print(f"{col}: {percentage_le_zero:.2f}%")
        else:
            print(f"Aviso: A coluna {col} não está presente no DataFrame.")
    
    # Transformar em NaN as ocorrências de valores <= 0 nas colunas especificadas
    df[columns_to_treat] = df[columns_to_treat].applymap(lambda x: np.nan if x <= 0 else x)
    
    # Calcular e imprimir a porcentagem de NaNs após o tratamento
    print("\nPorcentagem de NaNs após o tratamento:")
    for col in columns_to_treat:
        if col in df.columns:
            total = df[col].shape[0]
            count_nan = df[col].isna().sum()
            percentage_nan = (count_nan / total) * 100
            print(f"{col}: {percentage_nan:.2f}%")
        else:
            print(f"Aviso: A coluna {col} não está presente no DataFrame.")
    
    # Inicializar o codificador de rótulos
    label_encoder = LabelEncoder()
    
    # Aplicar a codificação de rótulos
    df['id_estacao_encoded'] = label_encoder.fit_transform(df['id_estacao'])
    
    # Remover colunas desnecessárias e com NaN
    df = df.drop(columns=['data', 'ano', 'id_estacao'])
    df = df.dropna()
    
    # Separar variáveis independentes (X) e dependente (y)
    if target_column == 'temperatura_max':
        X = df.drop(columns=[target_column, 'temperatura_min'])  # Remover também temperatura_min
        y = df['temperatura_max']
    elif target_column == 'temperatura_min':
        X = df.drop(columns=[target_column, 'temperatura_max'])  # Remover também temperatura_max
        y = df['temperatura_min']
    else:
        # Caso o target_column não seja um dos especificados
        X = df.drop(columns=[target_column])
        y = df[target_column]
    
    # Dividir em conjuntos de treino e teste, mantendo os NaNs apenas no teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Escalonar os dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return df, X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder

def train_and_evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test):
    # Definir o grid de parâmetros para o XGBoost
    param_grid = {
        'objective': ['reg:squarederror'],
        'colsample_bytree': [0.3, 0.5, 0.7],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'alpha': [0, 10, 20],
        'n_estimators': [50, 100, 150]
    }
    
    # Configurar o modelo XGBoost
    xg_reg = xgb.XGBRegressor()
    
    # Configurar o GridSearchCV
    grid_search = GridSearchCV(
        estimator=xg_reg,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=3,
        verbose=1,
        n_jobs=2
    )
    
    # Executar a busca em grade
    grid_search.fit(X_train_scaled, y_train)
    
    # Obter o melhor modelo
    best_model = grid_search.best_estimator_
    
    # Prever e avaliar o modelo no conjunto de teste
    y_pred = best_model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'\nMean Squared Error: {mse}')
    print(f'Mean Absolute Error: {mae}')
    print(f'R² Score: {r2}\n')
    
    return best_model

def impute_missing_values(df, target_column, best_model, scaler, label_encoder, columns_to_treat):
    # Cria uma cópia do DataFrame para evitar modificar o original
    df = df.copy()
    
    # Converter data para o formato datetime
    df['data'] = pd.to_datetime(df['data'], format='%Y-%m-%d', errors='coerce')
    
    # Criar as colunas com dia, mês e ano
    df['dia'] = df['data'].dt.day
    df['mes'] = df['data'].dt.month
    df['ano'] = df['data'].dt.year
    
    # Converter hora para string se não for float
    if df['hora'].dtype != float:
        df['hora'] = df['hora'].astype(str)
        # Verificar e ajustar o formato da hora
        df['hora'] = df['hora'].apply(lambda x: x.replace(':', '') if ':' in x else x)
        # Converter hora para inteiro e normalizar
        df['hora'] = df['hora'].astype(int)
        df['hora'] = df['hora'] / 10000
    
    
    # Transformar em NaN as ocorrências de valores <= 0 nas colunas especificadas
    df[columns_to_treat] = df[columns_to_treat].applymap(lambda x: np.nan if x <= 0 else x)
    
    # Aplicar a codificação de rótulos
    df['id_estacao_encoded'] = label_encoder.transform(df['id_estacao'])
    
    # Armazenar colunas removidas para adicionar de volta posteriormente
    original_columns = df[['data', 'id_estacao', 'ano']]
    
    # Manter uma cópia das colunas de temperatura
    temp_min_col = df['temperatura_min'].copy() if 'temperatura_min' in df.columns else None
    temp_max_col = df['temperatura_max'].copy() if 'temperatura_max' in df.columns else None
    
    # Remover colunas que não são necessárias para a previsão
    df = df.drop(columns=['data', 'id_estacao', 'ano'])
    
    # Remover a coluna de temperatura que não é o target para manter consistência
    if target_column == 'temperatura_max' and 'temperatura_min' in df.columns:
        df = df.drop(columns=['temperatura_min'])
    elif target_column == 'temperatura_min' and 'temperatura_max' in df.columns:
        df = df.drop(columns=['temperatura_max'])
    
    # Identificar as linhas com valores ausentes no target
    nan_rows = df[df[target_column].isna()]
    non_nan_rows = df.dropna(subset=[target_column])
    
    # Separar variáveis independentes
    X_nan = nan_rows.drop(columns=[target_column])
    
    # Usar o escalonador já ajustado para transformar os dados
    X_nan_scaled = scaler.transform(X_nan)
  
    # Prever os valores ausentes
    y_nan_pred = best_model.predict(X_nan_scaled)
    
    # Substituir os NaNs pelas previsões
    df.loc[df[target_column].isna(), target_column] = y_nan_pred
    
    # Adicionar de volta as colunas removidas
    df['data'] = original_columns['data']
    df['id_estacao'] = original_columns['id_estacao']
    df['ano'] = original_columns['ano']
    
    # Re-adicionar a coluna de temperatura que foi removida
    if target_column == 'temperatura_max' and temp_min_col is not None:
        df['temperatura_min'] = temp_min_col
    elif target_column == 'temperatura_min' and temp_max_col is not None:
        df['temperatura_max'] = temp_max_col
    
    # Remover colunas temporárias
    df = df.drop(columns=['id_estacao_encoded', 'dia'])
    
    return df

# Função principal para o pipeline
def run_pipeline(df, target_columns, columns_to_treat):
    df = df.copy()
    for target_column in target_columns:
        print(f'\nProcessing target column: {target_column}')
        
        # Processar os dados
        df_preprocessed, X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder = preprocess_data(df, target_column, columns_to_treat)
        
        # Treinar o modelo e avaliar seu desempenho
        best_model = train_and_evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # Imputar os valores ausentes no dataset original
        df = impute_missing_values(df, target_column, best_model, scaler, label_encoder, columns_to_treat)
        
        # Verificar se há valores ausentes restantes
        missing_values = df[target_column].isnull().sum()
        print(f'Valores ausentes em {target_column}: {missing_values}')
    
    return df

def plot_average_temperatures(df_before, df_after, year):
    """
    Plota as temperaturas máximas e mínimas médias mensais antes e depois da imputação.

    Args:
        df_before (pd.DataFrame): DataFrame antes da imputação.
        df_after (pd.DataFrame): DataFrame após a imputação.
        year (int): Ano referente aos dados.
    """
    print(f'\nIniciando plotagem para o ano {year}...')
    
    # Filtrar os dados para o ano específico
    df_before_year = df_before[df_before['data'].dt.year == year]
    df_after_year = df_after[df_after['data'].dt.year == year]
    
    # Verificar se os DataFrames filtrados não estão vazios
    if df_before_year.empty:
        print(f"Aviso: df_before_year está vazio para o ano {year}.")
    if df_after_year.empty:
        print(f"Aviso: df_after_year está vazio para o ano {year}.")
    
    # Agrupar por mês e calcular a temperatura máxima e mínima média
    temperatura_anual_before = df_before_year.groupby('mes').agg(
        temperatura_max_before=('temperatura_max', 'mean'),
        temperatura_min_before=('temperatura_min', 'mean')
    ).reset_index()
    
    temperatura_anual_after = df_after_year.groupby('mes').agg(
        temperatura_max_after=('temperatura_max', 'mean'),
        temperatura_min_after=('temperatura_min', 'mean')
    ).reset_index()
    
    # Combinar os dados antes e depois para facilitar a plotagem
    temperatura_comparativa = pd.merge(
        temperatura_anual_before, 
        temperatura_anual_after, 
        on='mes', 
        how='inner'
    )
    
    # Plotar a série temporal anual
    plt.figure(figsize=(12, 6))  # Ajusta o tamanho do gráfico

    # Temperatura Máxima
    plt.plot(
        temperatura_comparativa['mes'], 
        temperatura_comparativa['temperatura_max_before'], 
        label='Máxima Antes da Imputação', 
        marker='o', 
        linestyle='--'
    )
    plt.plot(
        temperatura_comparativa['mes'], 
        temperatura_comparativa['temperatura_max_after'], 
        label='Máxima Depois da Imputação', 
        marker='o'
    )

    # Temperatura Mínima
    plt.plot(
        temperatura_comparativa['mes'], 
        temperatura_comparativa['temperatura_min_before'], 
        label='Mínima Antes da Imputação', 
        marker='s', 
        linestyle='--'
    )
    plt.plot(
        temperatura_comparativa['mes'], 
        temperatura_comparativa['temperatura_min_after'], 
        label='Mínima Depois da Imputação', 
        marker='s'
    )

    plt.xlabel('Mês')
    plt.ylabel('Temperatura Média (°C)')
    plt.title(f'Temperaturas Máxima e Mínima Mensais - Ano {year}')
    plt.xticks(range(1, 13))  # Define os ticks do eixo x para cada mês
    plt.legend()
    plt.grid(True)  # Adiciona uma grade para melhor visualização
    plt.tight_layout()  # Ajusta o layout para evitar sobreposição
    plt.show()


import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # Necessário para o IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math
import matplotlib.pyplot as plt
import seaborn as sns


def optimize_data_types(df):
    """
    Otimiza os tipos de dados para reduzir o uso de memória, convertendo tipos de 64 bits para 32 bits.
    """
    df = df.copy()
    
    # Converter float64 para float32
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    
    # Converter int64 para int32
    for col in df.select_dtypes(include=['int64', 'Int64']).columns:
        df[col] = df[col].astype('int32')
    
    return df

def preprocess_data_MICE(df, exclude_cols=None):
    """
    Pré-processa o dataset para preparação antes da imputação.
    """
    df = df.copy()

    # Otimizar os tipos de dados para reduzir o uso de memória
    df = optimize_data_types(df)

    # Combinar 'data' e 'hora' em 'data_hora'
    df['data_hora'] = pd.to_datetime(df['data'] + ' ' + df['hora'], format='%Y-%m-%d %H:%M:%S')

    # Extrair dia e hora
    df['dia'] = df['data_hora'].dt.day.astype('int32')
    df['hora_dia'] = df['data_hora'].dt.hour.astype('int32')

    # Remover colunas desnecessárias
    df = df.drop(['data', 'hora'], axis=1)

    # Ordenar o DataFrame por 'data_hora'
    df = df.sort_values('data_hora').reset_index(drop=True)

    # Definir colunas a serem excluídas da imputação
    if exclude_cols is None:
        exclude_cols = ['data_hora', 'id_estacao']
    else:
        exclude_cols.extend(['data_hora', 'id_estacao'])

    # Selecionar colunas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Definir as colunas a serem imputadas (excluindo as colunas que não precisam de imputação)
    cols_to_impute = [col for col in numeric_cols if col not in exclude_cols]

    # Definir as colunas imputáveis (todas as colunas numéricas que serão usadas como preditoras)
    imputable_cols = cols_to_impute + ['ano', 'mes', 'dia', 'hora_dia']

    # Garantir que todas as colunas em imputable_cols estão no DataFrame
    imputable_cols = [col for col in imputable_cols if col in df.columns]

    return df, cols_to_impute, imputable_cols

def clean_temperature_columns(df, columns):
    """
    Transforma valores 0 e abaixo de 0 em NaN nas colunas especificadas.
    Além disso, calcula e imprime a quantidade e a porcentagem de valores 0 ou abaixo antes da limpeza.
    """
    df = df.copy()
    total_rows = len(df)
    
    for col in columns:
        if col in df.columns:
            # Calcular quantidade e porcentagem de valores <= 0 antes da limpeza
            count_zero_or_below = (df[col] <= 0).sum()
            percent_zero_or_below = (count_zero_or_below / total_rows) * 100
            print(f"Coluna '{col}': {count_zero_or_below} ({percent_zero_or_below:.2f}%) valores eram 0 ou abaixo de 0 antes da limpeza.")
            
            # Transformar valores <= 0 em NaN
            original_nan_count = df[col].isna().sum()
            df.loc[df[col] <= 0, col] = np.nan
            new_nan_count = df[col].isna().sum()
            nan_added = new_nan_count - original_nan_count
            print(f"Coluna '{col}': {nan_added} valores transformados em NaN.")
        else:
            print(f"Coluna '{col}' não encontrada no DataFrame.")
    return df

def introduce_missing_values(df, cols_to_introduce, missing_rate=0.1):
    """
    Introduz valores faltantes artificiais nas colunas especificadas.
    """
    df_missing = df.copy()
    missing_info = {}

    np.random.seed(42)  # Para reprodutibilidade

    for col in cols_to_introduce:
        # Obter índices não nulos
        non_null_indices = df_missing[df_missing[col].notnull()].index

        # Número de valores a serem removidos
        n_missing = int(len(non_null_indices) * missing_rate)

        # Selecionar índices aleatórios para remoção
        if n_missing > 0:
            missing_indices = np.random.choice(non_null_indices, n_missing, replace=False)

            # Armazenar os valores reais
            missing_info[col] = {
                'indices': missing_indices,
                'values': df_missing.loc[missing_indices, col]
            }

            # Introduzir valores faltantes
            df_missing.loc[missing_indices, col] = np.nan
        else:
            print(f"Coluna '{col}' não tem valores suficientes para introduzir valores faltantes.")
            missing_info[col] = {
                'indices': [],
                'values': []
            }

    return df_missing, missing_info

def apply_mice(df, cols_to_impute, imputable_cols):
    """
    Aplica o MICE para imputação de valores faltantes nas colunas especificadas.
    """
    df = df.copy()

    from sklearn.ensemble import RandomForestRegressor

    imputer = IterativeImputer(estimator=RandomForestRegressor(
                                   n_estimators=100,  # Número de árvores na floresta
                                   max_depth=None,    # Profundidade máxima das árvores
                                   random_state=24,
                                   n_jobs=-1),
                               max_iter=50,  # Número de iterações
                               random_state=24,
                               verbose=2)

    # Aplicar o imputador em todas as colunas imputáveis
    imputed_values = imputer.fit_transform(df[imputable_cols])

    # Verificar a convergência
    print(f"Convergência alcançada em {imputer.n_iter_} iterações.")

    # Criar um DataFrame com os valores imputados
    df_imputed = pd.DataFrame(imputed_values, columns=imputable_cols, index=df.index)

    # Substituir as colunas imputadas no DataFrame original
    for col in cols_to_impute:
        df[col] = df_imputed[col]

    return df

def evaluate_imputation(df_original, df_imputed, missing_info):
    """
    Avalia a imputação comparando os valores imputados com os valores reais removidos.
    """
    metrics = {}

    for col, info in missing_info.items():
        indices = info['indices']
        if len(indices) == 0:
            print(f"Sem valores removidos para a coluna '{col}'.")
            continue
        true_values = info['values']
        imputed_values = df_imputed.loc[indices, col]

        # Calcular MAE
        mae = mean_absolute_error(true_values, imputed_values)

        # Calcular RMSE
        rmse = mean_squared_error(true_values, imputed_values, squared=False)

        # Calcular R²
        r2 = r2_score(true_values, imputed_values)

        metrics[col] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }

    return metrics

def print_nan_percentage(df, title="Porcentagem de NaNs por Coluna"):
    """
    Calcula e imprime a porcentagem de valores NaN para cada coluna do DataFrame.
    """
    total_rows = len(df)
    nan_counts = df.isna().sum()
    nan_percent = (nan_counts / total_rows) * 100
    nan_summary = pd.DataFrame({
        'Total NaNs': nan_counts,
        'Percentagem NaNs (%)': nan_percent
    })
    nan_summary = nan_summary.sort_values(by='Percentagem NaNs (%)', ascending=False)
    
    print(f"\n{title}:")
    print(nan_summary)
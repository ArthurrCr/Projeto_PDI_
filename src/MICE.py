import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # Necessário para o IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def optimize_data_types(df):
    """
    Otimiza os tipos de dados para reduzir o uso de memória, convertendo tipos de 64 bits para 32 bits.
    
    Parâmetros:
    - df: DataFrame a ser otimizado.
    
    Retorna:
    - df: DataFrame com tipos de dados otimizados.
    """
    df = df.copy()
    
    # Converter float64 para float32
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    
    # Converter int64 para int32
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    
    return df

def preprocess_data_MICE(df, station_id=None, exclude_cols=None):
    """
    Pré-processa o dataset para preparação antes da imputação.

    Parâmetros:
    - df: DataFrame original.
    - station_id: ID da estação para filtrar (opcional).
    - exclude_cols: Lista de colunas a serem excluídas da imputação (opcional).

    Retorna:
    - df_preprocessed: DataFrame pré-processado.
    - cols_to_impute: Lista de colunas a serem imputadas.
    - imputable_cols: Lista de colunas a serem usadas como preditoras.
    - label_encoder: Objeto LabelEncoder usado na codificação de 'id_estacao'.
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

    # Codificar 'id_estacao' usando LabelEncoder
    label_encoder = LabelEncoder()
    df['id_estacao_cod'] = label_encoder.fit_transform(df['id_estacao']).astype('int32')

    # Remover 'id_estacao' original
    df = df.drop('id_estacao', axis=1)

    # Ordenar o DataFrame por 'data_hora'
    df = df.sort_values('data_hora').reset_index(drop=True)

    # Definir colunas a serem excluídas da imputação
    if exclude_cols is None:
        exclude_cols = ['ano', 'mes', 'dia', 'hora_dia', 'data_hora', 'id_estacao_cod']
    else:
        exclude_cols.extend(['data_hora', 'id_estacao_cod'])

    # Selecionar colunas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Todas as colunas numéricas serão usadas como preditoras
    imputable_cols = numeric_cols + ['ano', 'mes', 'dia', 'hora_dia', 'id_estacao_cod']

    # Remover duplicatas (caso as colunas já estejam em numeric_cols)
    imputable_cols = list(set(imputable_cols))

    # Garantir que todas as colunas em imputable_cols estão no DataFrame
    imputable_cols = [col for col in imputable_cols if col in df.columns]

    # Definir as colunas a serem imputadas (excluindo as colunas que não precisam de imputação)
    cols_to_impute = [col for col in numeric_cols if col not in exclude_cols]

    return df, cols_to_impute, imputable_cols, label_encoder

def introduce_missing_values(df, cols_to_introduce, missing_rate=0.1):
    """
    Introduz valores faltantes artificiais nas colunas especificadas.

    Parâmetros:
    - df: DataFrame pré-processado.
    - cols_to_introduce: Lista de colunas onde serão introduzidos valores faltantes.
    - missing_rate: Proporção de valores a serem removidos (entre 0 e 1).

    Retorna:
    - df_missing: DataFrame com valores faltantes introduzidos.
    - missing_info: Dicionário com informações dos valores removidos (índices e valores reais).
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
        missing_indices = np.random.choice(non_null_indices, n_missing, replace=False)

        # Armazenar os valores reais
        missing_info[col] = {
            'indices': missing_indices,
            'values': df_missing.loc[missing_indices, col]
        }

        # Introduzir valores faltantes
        df_missing.loc[missing_indices, col] = np.nan

    return df_missing, missing_info

def apply_mice(df, cols_to_impute, imputable_cols):
    """
    Aplica o MICE para imputação de valores faltantes nas colunas especificadas.

    Parâmetros:
    - df: DataFrame pré-processado (com valores faltantes artificiais introduzidos).
    - cols_to_impute: Lista de colunas a serem imputadas.
    - imputable_cols: Lista de colunas a serem usadas como preditoras.

    Retorna:
    - df_imputed: DataFrame com valores imputados.
    """
    df = df.copy()

    # Configurar o imputador com BayesianRidge para reduzir o uso de memória
    from sklearn.linear_model import BayesianRidge

    imputer = IterativeImputer(estimator=BayesianRidge(),
                               max_iter=10, random_state=0)

    # Aplicar o imputador em todas as colunas imputáveis
    imputed_values = imputer.fit_transform(df[imputable_cols])

    # Criar um DataFrame com os valores imputados
    df_imputed = pd.DataFrame(imputed_values, columns=imputable_cols, index=df.index)

    # Substituir as colunas imputadas no DataFrame original
    for col in cols_to_impute:
        df[col] = df_imputed[col]

    return df

def postprocess_data(df, label_encoder):
    """
    Realiza pós-processamento após a imputação (se necessário).

    Parâmetros:
    - df: DataFrame após imputação.
    - label_encoder: Objeto LabelEncoder usado no pré-processamento.

    Retorna:
    - df_postprocessed: DataFrame pós-processado.
    """
    df = df.copy()

    # Reverter codificação de 'id_estacao'
    df['id_estacao'] = label_encoder.inverse_transform(df['id_estacao_cod'].astype(int))

    return df

def evaluate_imputation(df_original, df_imputed, missing_info):
    """
    Avalia a imputação comparando os valores imputados com os valores reais removidos.

    Parâmetros:
    - df_original: DataFrame original (antes de introduzir valores faltantes artificiais).
    - df_imputed: DataFrame após imputação.
    - missing_info: Dicionário com informações dos valores removidos.

    Retorna:
    - metrics: Dicionário com métricas de erro para cada coluna.
    """
    metrics = {}

    for col, info in missing_info.items():
        indices = info['indices']
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

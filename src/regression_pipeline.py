import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def preprocess_data(df, target_column):
    # cria uma cópia
    df = df.copy()
    # Converter data para o formato datetime
    df['data'] = pd.to_datetime(df['data'], format='%Y-%m-%d')

    # Criar a coluna com apenas o dia
    df['dia'] = df['data'].dt.day

    # Converter hora para string se não for float
    if df['hora'].dtype != float:
        df['hora'] = df['hora'].astype(str)
        # Verificar e ajustar o formato da hora
        df['hora'] = df['hora'].apply(lambda x: x.replace(':', '') if ':' in x else x)
        # Converter hora para inteiro e normalizar
        df['hora'] = df['hora'].astype(int)
        df['hora'] = df['hora'] / 10000

    # Inicializar o codificador de rótulos
    label_encoder = LabelEncoder()

    # Aplicar a codificação de rótulos
    df['id_estacao_encoded'] = label_encoder.fit_transform(df['id_estacao'])

    # Remover colunas desnecessárias
    df = df.drop(columns=['data', 'ano', 'id_estacao'])
    df = df.dropna()

    # Separar variáveis independentes (X) e dependente (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Dividir em conjuntos de treino e teste, mantendo os NaNs apenas no treino
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Escalonar os dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return df, X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder

# Função para treinar o modelo e avaliar seu desempenho
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
    grid_search = GridSearchCV(estimator=xg_reg, param_grid=param_grid, scoring='neg_mean_squared_error', 
                               cv=3, verbose=1, n_jobs=-1)

    # Executar a busca em grade
    grid_search.fit(X_train_scaled, y_train)

    # Obter o melhor modelo
    best_model = grid_search.best_estimator_

    # Prever e avaliar o modelo no conjunto de teste
    y_pred = best_model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'Mean Absolute Error: {mae}')
    print(f'R² Score: {r2}')

    return best_model

# Função para imputar valores ausentes no dataset
def impute_missing_values(df, target_column, best_model, scaler, label_encoder):
    # Cria uma cópia do DataFrame para evitar modificar o original
    df = df.copy()
    
    # Converter data para o formato datetime
    df['data'] = pd.to_datetime(df['data'], format='%Y-%m-%d')

    # Criar a coluna com apenas o dia
    df['dia'] = df['data'].dt.day

    # Converter hora para string se não for float
    if df['hora'].dtype != float:
        df['hora'] = df['hora'].astype(str)
        # Verificar e ajustar o formato da hora
        df['hora'] = df['hora'].apply(lambda x: x.replace(':', '') if ':' in x else x)
        # Converter hora para inteiro e normalizar
        df['hora'] = df['hora'].astype(int)
        df['hora'] = df['hora'] / 10000

    # Aplicar a codificação de rótulos
    df['id_estacao_encoded'] = label_encoder.transform(df['id_estacao'])

    # Armazenar colunas removidas para adicionar de volta posteriormente
    original_columns = df[['data', 'id_estacao','ano']]

    # Remover colunas que não são necessárias para a previsão
    df = df.drop(columns=['data', 'id_estacao','ano'])

    # Identificar as linhas com valores ausentes no target
    nan_rows = df[df[target_column].isna()]
    non_nan_rows = df.dropna(subset=[target_column])

    # Separar variáveis independentes
    X_nan = nan_rows.drop(columns=[target_column])
    X_non_nan = non_nan_rows.drop(columns=[target_column])

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

    # Remover colunas temporárias
    df = df.drop(columns=['id_estacao_encoded','dia'])

    return df

# Função principal para o pipeline
def run_pipeline(df, target_columns):
    df = df.copy()
    for target_column in target_columns:
        print(f'Processing target column: {target_column}')
        
        # Processar os dados
        df_preprocessed, X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder = preprocess_data(df, target_column)
        
        # Treinar o modelo e avaliar seu desempenho
        best_model = train_and_evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # Imputar os valores ausentes no dataset original
        df = impute_missing_values(df, target_column, best_model, scaler, label_encoder)
        
        # Verificar se há valores ausentes restantes
        missing_values = df[target_column].isnull().sum()
        print(f'Remaining missing values in {target_column}: {missing_values}')

    return df
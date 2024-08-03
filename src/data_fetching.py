import basedosdados as bd
import pandas as pd

def fetch_inmet_data(years):
    """
    Função para buscar dados do INMET a partir de uma lista de anos.
    
    Parameters:
        years (list of int): Lista de anos para filtrar os dados.
        
    Returns:
        pd.DataFrame: DataFrame com os dados filtrados.
    """
    # Convertendo a lista de anos para string para usar na query
    years_str = ', '.join(map(str, years))
    
    station_ids_str = """
    'A323', 'A327', 'A371', 'A408', 'A412', 'A413', 'A415', 'A416', 'A418',
    'A423', 'A424', 'A425', 'A426', 'A428', 'A429', 'A430', 'A432', 'A433',
    'A435', 'A436', 'A439', 'A440', 'A441', 'A442', 'A443', 'A448', 'A449',
    'A450', 'A458', 'A305', 'A306', 'A314', 'A315', 'A319', 'A324', 'A325',
    'A332', 'A339', 'A342', 'A347', 'A358', 'A359', 'A360', 'A368', 'A369',
    'A310', 'A313', 'A321', 'A333', 'A334', 'A348', 'A373', 'A307', 'A309',
    'A322', 'A328', 'A329', 'A349', 'A350', 'A351', 'A366', 'A370', 'A308',
    'A330', 'A331', 'A336', 'A337', 'A343', 'A345', 'A354', 'A365', 'A316',
    'A317', 'A318', 'A340', 'A367', 'A372', 'A417', 'A419', 'A420', 'A451',
    'A453', 'A526', 'A539', 'A543', 'A559', 'A563', 'A454'
    """
    
    query = f"""
    SELECT
        ano,
        mes,
        data,
        hora,
        id_estacao,
        precipitacao_total,
        pressao_atm_hora,
        pressao_atm_max,
        pressao_atm_min,
        radiacao_global,
        temperatura_bulbo_hora,
        temperatura_orvalho_hora,
        temperatura_max,
        temperatura_min,
        temperatura_orvalho_max,
        temperatura_orvalho_min,
        umidade_rel_max,
        umidade_rel_min,
        umidade_rel_hora,
        vento_direcao,
        vento_rajada_max,    
        vento_velocidade
    FROM
        `basedosdados.br_inmet_bdmep.microdados`
    WHERE
        ano IN ({years_str})
        AND id_estacao IN ({station_ids_str});
    """
    
    billing_project_id = "projetopdi-430718"
    df_meta_dados = bd.read_sql(query, billing_project_id=billing_project_id)
    return df_meta_dados
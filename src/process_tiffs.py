import os
import pandas as pd
import rasterio
from rasterio.mask import mask
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from rasterio.plot import show

def save_masked_tiff(raster_path, estacao_id, output_dir, gdf_voronoi_clipped):
    """
    Mascarar um GeoTIFF com base em um polígono de Voronoi e salvar o resultado.

    Args:
    raster_path (str): Caminho para o arquivo GeoTIFF de entrada.
    estacao_id (str): ID da estação para selecionar o polígono de Voronoi correspondente.
    output_dir (str): Diretório onde o GeoTIFF resultante será salvo.
    gdf_voronoi_clipped (GeoDataFrame): GeoDataFrame contendo os polígonos de Voronoi.

    Returns:
    str: Caminho para o arquivo GeoTIFF salvo.
    """

    # Carregar o arquivo GeoTIFF
    with rasterio.open(raster_path) as src:
        input_crs = src.crs  # Obter o CRS do GeoTIFF

    # Verificar e transformar o CRS do GeoDataFrame de Voronoi se necessário
    if gdf_voronoi_clipped.crs != input_crs:
        gdf_voronoi_clipped = gdf_voronoi_clipped.to_crs(input_crs)

    # Filtrar o polígono específico do ID da estação
    polygon = gdf_voronoi_clipped.loc[gdf_voronoi_clipped['id_estacao'] == estacao_id, 'geometry'].values[0]

    with rasterio.open(raster_path) as src:
        # Aplicar a máscara usando o polígono
        out_image, out_transform = mask(src, [polygon], crop=True)
        out_meta = src.meta.copy()

        # Atualizar metadados para o novo arquivo
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        # Extrair o ano do nome do arquivo GeoTIFF original
        year = os.path.basename(raster_path).split('_')[-1].split('.')[0]
        output_tif = os.path.join(output_dir, f"coverage_{estacao_id}_{year}.tif")

        # Salvar o GeoTIFF resultante
        with rasterio.open(output_tif, "w", **out_meta) as dest:
            dest.write(out_image)


def extract_classification_info(tiff_path):
    """
    Extrair as classificações e a contagem de pixels por classe de um GeoTIFF.

    Args:
    tiff_path (str): Caminho para o arquivo GeoTIFF.

    Returns:
    tuple: Um conjunto contendo as classificações únicas e um dicionário com a contagem de pixels por classe.
    """

    with rasterio.open(tiff_path) as src:
        out_image = src.read(1)  # Ler a primeira banda do GeoTIFF

    # Contagem de pixels por classe
    unique, counts = np.unique(out_image, return_counts=True)
    class_counts = dict(zip(unique, counts))

    # Ordenar pela contagem em ordem crescente
    sorted_class_counts = dict(sorted(class_counts.items(), key=lambda item: item[1]))

    return sorted_class_counts


def plot_tiff_with_classes(tiff_path):
    """
    Plotar um GeoTIFF com um colormap personalizado para as classes.

    Args:
    tiff_path (str): Caminho para o arquivo GeoTIFF.
    """

    # Abrir o arquivo GeoTIFF
    with rasterio.open(tiff_path) as src:
        out_image = src.read(1)  # Ler a primeira banda do raster
        out_transform = src.transform

    # Definir as cores das classes
    class_colors = {
        1: "#32a65e", 6: "#026975", 10: "#ad975a", 13: "#d89f5c", 14: "#FFFFB2", 
        18: "#E974ED", 19: "#C27BA0", 40: "#c71585", 36: "#d082de", 46: "#d68fe2", 
        3: "#1f8d49", 4: "#7dc975", 5: "#04381d", 49: "#02d659", 20: "#db7093",
        11: "#519799", 12: "#d6bc74", 32: "#fc8114", 29: "#ffaa5f", 50: "#ad5100",
        15: "#edde8e", 39: "#f5b3c8", 62: "#ff69b4", 41: "#f54ca9", 46: "#d68fe2",
        48: "#e6ccff", 9: "#7a5900", 21: "#ffefc3", 23: "#ffa07a", 24: "#d4271e", 
        30: "#9c0027", 25: "#db4d4f", 33: "#2532e4", 31: "#091077", 0: "#FFFFFF", 
        27: "#FFFFFF", 35:"#9065d0", 22: "#d4271e", 26: "#0000FF"
    }

    # Verificar as classificações presentes no raster extraído
    unique_classes = np.unique(out_image)

    # Criar uma lista de cores para o colormap na ordem dos valores únicos
    colors = [class_colors[key] for key in unique_classes] 

    # Criar o colormap e o normalizador
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries=unique_classes, ncolors=len(unique_classes))

    # Plotar o raster com as cores das classes
    fig, ax = plt.subplots(figsize=(10, 10))
    show(out_image, transform=out_transform, cmap=cmap, norm=norm, ax=ax)
    ax.set_title(f"Classes do raster {os.path.basename(tiff_path)}")
    plt.show()


def plot_time_series(df_time_series, estacao_id):
    """
    Plotar a série temporal de contagem de superpixels por classe.

    Args:
    df_time_series (pd.DataFrame): DataFrame contendo a série temporal.
    estacao_id (str): ID da estação para o título do gráfico.
    """

    plt.figure(figsize=(14, 8))
    
    # Plotar cada classe de superpixel
    for class_id in df_time_series.columns:
        plt.plot(df_time_series.index, df_time_series[class_id], marker='o', label=f'Classe {class_id}')
    
    # Adicionar título e rótulos
    plt.title(f'Série Temporal das Contagens de Superpixels - Estação {estacao_id}', fontsize=16)
    plt.xlabel('Ano', fontsize=14)
    plt.ylabel('Contagem de Superpixels', fontsize=14)
    
    # Definir os ticks do eixo x para mostrar apenas anos inteiros
    plt.xticks(df_time_series.index, [str(int(year)) for year in df_time_series.index])
    
    # Adicionar uma grade
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adicionar a legenda fora do gráfico
    plt.legend(title='Classes de Superpixels', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Ajustar layout para evitar corte das legendas
    plt.tight_layout()
    
    # Exibir o gráfico
    plt.show()

def create_time_series(output_base_dir, estacao_id, start_year, end_year):
    """
    Criar uma série temporal de contagem de superpixels por classe para uma estação específica,
    removendo IDs de classe com valor zero.

    Args:
    output_base_dir (str): Diretório base onde os GeoTIFFs resultantes foram salvos.
    estacao_id (str): ID da estação para criar a série temporal.
    start_year (int): Ano inicial da série temporal.
    end_year (int): Ano final da série temporal.

    Returns:
    pd.DataFrame: DataFrame contendo a série temporal.
    """

    # Criar um DataFrame para armazenar as contagens por ano
    time_series_data = {}

    for year in range(start_year, end_year + 1):
        tiff_path = os.path.join(output_base_dir, str(year), f'coverage_{estacao_id}_{year}.tif')

        if os.path.exists(tiff_path):
            sorted_class_counts = extract_classification_info(tiff_path)
            # Remover a entrada com ID de classe 0, se existir
            if 0 in sorted_class_counts:
                del sorted_class_counts[0]
            time_series_data[year] = sorted_class_counts
        else:
            print(f"Arquivo {tiff_path} não encontrado.")

    # Converter o dicionário em DataFrame
    df_time_series = pd.DataFrame(time_series_data).T.fillna(0)

    return df_time_series

def calcular_media_temperatura(dados, estacao_id):
    """
    Calcula a média anual das temperaturas máximas e mínimas para uma estação específica.

    Args:
    dados (pd.DataFrame): DataFrame contendo as colunas ['id_estacao', 'data', 'temperatura_max', 'temperatura_min', 'ano'].
    estacao_id (str): ID da estação para filtrar os dados.

    Returns:
    pd.DataFrame: DataFrame com a média anual das temperaturas.
    """
    # Filtrar os dados para a estação específica
    dados_estacao = dados[dados['id_estacao'] == estacao_id]

    # Agrupar por ano e calcular a média das temperaturas máxima e mínima
    temperatura_anual = dados_estacao.groupby('ano').agg(
        temperatura_max=('temperatura_max', 'mean'),
        temperatura_min=('temperatura_min', 'mean')
    ).reset_index()

    return temperatura_anual

def combinar_datasets(temperatura_anual, df_time_series):
    """
    Combina a média anual das temperaturas com a série temporal de contagem de superpixels.

    Args:
    temperatura_anual (pd.DataFrame): DataFrame com a média anual das temperaturas.
    df_time_series (pd.DataFrame): DataFrame contendo a série temporal de contagem de superpixels.

    Returns:
    pd.DataFrame: DataFrame combinado com as temperaturas e a contagem de superpixels por ano.
    """
    # Mesclar os dois DataFrames usando o ano como chave
    dataset_combinado = pd.merge(temperatura_anual, df_time_series, left_on='ano', right_index=True, how='inner')
    
    return dataset_combinado

# Função para criar o dataset completo para uma estação
def criar_dataset_completo(dados, output_base_dir, estacao_id, start_year, end_year):
    """
    Cria um dataset completo combinando as médias de temperatura e a contagem de superpixels.

    Args:
    dados (pd.DataFrame): DataFrame contendo as colunas ['id_estacao', 'data', 'temperatura_max', 'temperatura_min', 'ano'].
    output_base_dir (str): Diretório base onde os GeoTIFFs resultantes foram salvos.
    estacao_id (str): ID da estação para criar o dataset.
    start_year (int): Ano inicial da série temporal.
    end_year (int): Ano final da série temporal.

    Returns:
    pd.DataFrame: DataFrame completo com as temperaturas e a contagem de superpixels por ano.
    """
    # Calcular a média anual das temperaturas
    temperatura_anual = calcular_media_temperatura(dados, estacao_id)

    # Criar a série temporal de superpixels
    df_time_series = create_time_series(output_base_dir, estacao_id, start_year, end_year)

    # Combinar os datasets
    dataset_completo = combinar_datasets(temperatura_anual, df_time_series)

    return dataset_completo
    

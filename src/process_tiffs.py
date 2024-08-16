import os
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
        3: "#1f8d49", 4: "#7dc975", 5: "#04381d", 49: "#02d659", 20: "#db7093",
        11: "#519799", 12: "#d6bc74", 32: "#fc8114", 29: "#ffaa5f", 50: "#ad5100",
        15: "#edde8e", 39: "#f5b3c8", 62: "#ff69b4", 41: "#f54ca9", 46: "#d68fe2",
        48: "#e6ccff", 9: "#7a5900", 21: "#ffefc3", 23: "#ffa07a", 24: "#d4271e", 
        30: "#9c0027", 25: "#db4d4f", 33: "#2532e4", 31: "#091077", 0: "#FFFFFF"
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

    

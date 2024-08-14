import os
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np

def save_extracted_tiff(raster_path, biomes_shp, biome_name='Caatinga', output_tiff_path='../dados/caatinga_extracted.tif'):
    """
    Esta função salva a imagem extraída do bioma selecionado em um novo GeoTiff.
    """
    # Carregar o shapefile dos biomas
    biomes = gpd.read_file(biomes_shp)

    # Filtrar para obter apenas o bioma desejado (por padrão, 'Caatinga')
    biome = biomes[biomes['Bioma'] == biome_name]

    # Carregar o arquivo GeoTiff e aplicar a máscara do bioma
    with rasterio.open(raster_path) as src:
        out_image, out_transform = mask(src, biome.geometry, crop=True, nodata=0)
        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform,
                         "nodata": 0})

    # Salvar o resultado em um novo GeoTiff
    with rasterio.open(output_tiff_path, "w", **out_meta) as dest:
        dest.write(out_image)

def get_class_counts(output_tiff_path):
    """
    Esta função retorna o dicionário com a contagem de classes para o ano selecionado,
    assumindo que o arquivo já está salvo na pasta especificada.
    """
    if not os.path.exists(output_tiff_path):
        raise FileNotFoundError(f"O arquivo {output_tiff_path} não foi encontrado.")
    
    # Carregar o arquivo GeoTiff salvo
    with rasterio.open(output_tiff_path) as src:
        out_image = src.read(1)  # Ler a primeira banda

    # Contagem de superpixels por classe
    unique, counts = np.unique(out_image, return_counts=True)
    class_counts = dict(zip(unique, counts))

    return class_counts


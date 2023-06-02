import geopandas as gp
import pandas as pd
import shapely
from shapely.geometry.polygon import Polygon, Point
import numpy as np

def fill_gdf_classification(gdf, fill_none = 'Cyst'):
    gdf_result = gdf.copy()
    gdf_list = gdf['classification'].tolist()
    gdf_list = [x if type(x)== dict else dict([('name',fill_none)]) for x in gdf_list]
    gdf_class = pd.DataFrame(gdf_list)
    gdf_result['classification']= gdf_class.to_dict(orient='records')
    return(gdf_result)

def calculate_distance_matrix(gdf,coordinate_array):
    distance_rows = []
    for pair in coordinate_array:
        ppair0 = pair[0]
        ppair1 = pair[1]
        distance_rows.append(gdf.distance(Point((ppair0,ppair1))).to_numpy())
    distance_mat=np.stack(distance_rows, axis=0)
    return(distance_mat)

def create_envelope():
    """This method creates a polygon that encapsulates the adata subset of interest.
    Args:
        adata (AnnData): An anndata object containing the observations with the points of interest.

    Returns:
        gdf_dict (dict): a dict which stores a nested dict (value) for each library sample (key). 
        The nested dict contains geopandas objects
    """
    pass

def transform_coordinates(gdf, scale_factor, x_shift, y_shift, TR_center, rotation=90):
    gdf_rotated = gdf.copy()

    for index, row in gdf_rotated.iterrows():
        rotated = shapely.affinity.rotate(row['geometry'], rotation, origin=TR_center)
        gdf_rotated.loc[index, 'geometry'] = rotated

    gdf_transformed = gdf_rotated.translate(xoff=x_shift, yoff=y_shift)
    #gdf_transformed = gdf_rotated
    gdf_transformed = gdf_transformed.scale(xfact=scale_factor, yfact=scale_factor, origin=(0,0))
    gdf_final=gdf.copy()
    gdf_final['geometry'] = gdf_transformed
    return(gdf_final)

def get_center(infoTable, library_id):
    sample_info = infoTable.set_index('sample_names').loc[library_id]
    original_x=sample_info['fullres_dimensions_y']
    original_y=sample_info['fullres_dimensions_x']
    return((original_x/2, original_y/2))

def filter_gdf(gdf, feature_name, fill_none = 'Cyst'):
    gdf_result = gdf.copy()
    gdf_list = gdf['classification'].tolist()
    gdf_list = [x if type(x)== dict else dict([('name',fill_none)]) for x in gdf_list]
    gdf_class = pd.DataFrame(gdf_list)
    gdf_result['classification']= gdf_class.to_dict(orient='records')

    gdf_filtered=gdf_result.iloc[gdf_class[gdf_class['name']==feature_name].index,:]
    return(gdf_filtered)

def load_geojson_annotations(infoTable, adata, feature_list, fill_none = 'Cyst'):
    adata_result = adata.copy()
    infoTable_annotations = infoTable.dropna(subset='annotations')
    result_dict = {}
    for ind,row in infoTable_annotations.iterrows():
        library_id = row['sample_names']
        gdf = gp.read_file(row['annotations'])
        feature_gdfs_fr = {}
        for feature in feature_list:
            feature_gdfs_fr[feature] = filter_gdf(gdf, feature, fill_none = fill_none)

        center_sample=get_center(infoTable_annotations, library_id)
        sf_dict = adata_result.uns['spatial'][library_id]['scalefactors']

        spatial_array=adata_result[adata_result.obs['sample_names'] == library_id].obsm['spatial_fr']

        feature_gdfs_fr_90 = {}
        distance_mats = {}
        feature_gdfs_sp = {}

        gdf_transformed = transform_coordinates(gdf, scale_factor=sf_dict['tissue_hires_scalef_FR'], x_shift=-sf_dict['X_shift_FR'], y_shift = -sf_dict['Y_shift_FR'], TR_center=center_sample, rotation=90)

        for feature in feature_list:
            feature_gdfs_fr_90[feature] = transform_coordinates(feature_gdfs_fr[feature], scale_factor = 1, x_shift=0, y_shift=0, TR_center = center_sample, rotation=90)
            distance_mats[feature] = calculate_distance_matrix(feature_gdfs_fr_90[feature], spatial_array)
            feature_gdfs_sp[feature] = filter_gdf(gdf_transformed, feature)
            cleaned_feature_name = feature.lower()
            cleaned_feature_name = cleaned_feature_name.replace(" ","_")
            cleaned_feature_name = cleaned_feature_name.replace(",","_")
            adata_result.obs.loc[adata_result.obs['sample_names'] == library_id, f"distance_to_any_{cleaned_feature_name}_px"] = distance_mats[feature].min(axis=1)
            adata_result.obs.loc[adata_result.obs['sample_names'] == library_id, f"distance_to_any_{cleaned_feature_name}_um"] = distance_mats[feature].min(axis=1)/(sf_dict['spot_diameter_fullres_FR']/65)


        result_dict[library_id] = {}
        result_dict[library_id]['gdf'] = gdf
        result_dict[library_id]['gdf_sp'] = gdf_transformed
        result_dict[library_id]['feature_gdfs_sp'] = feature_gdfs_sp
        result_dict[library_id]['feature_gdfs_fr'] = feature_gdfs_fr
        result_dict[library_id]['feature_gdfs_fr_90'] = feature_gdfs_fr_90
        result_dict[library_id]['distance_mats'] = distance_mats
        result_dict[library_id]['px2um_factor'] = sf_dict['spot_diameter_fullres_FR']/65 #number of pixels in a 65 um spot/65um = 1.626 px per um

    return(adata_result, result_dict)

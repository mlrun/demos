import os
import zipfile
import json
from tempfile import mktemp
import pandas as pd


def open_archive(context, 
                 target_dir='content',
                 archive_url=''):
    """Open a file/object archive into a target directory"""
        
    os.makedirs(target_dir, exist_ok=True)
    context.logger.info('Verified directories')
    
    context.logger.info('Extracting zip')
    zip_ref = zipfile.ZipFile(archive_url, 'r')
    zip_ref.extractall(target_dir)
    zip_ref.close()
    
    context.logger.info(f'extracted archive to {target_dir}')
    context.log_artifact('content', target_path=target_dir)

    
from mlrun.artifacts import TableArtifact

def categories_map_builder(context,
                           source_dir,
                           df_filename='file_categories_df.csv',
                           map_filename='categories_map.json'):
    """Read labeled images from a directory and create category map + df
    
    filename format: <category>.NN.jpg"""
    
    filenames = [file for file in os.listdir(source_dir) if file.endswith('.jpg')]
    categories = []
        
    for filename in filenames:
        category = filename.split('.')[0]
        categories.append(category)

    df = pd.DataFrame({
        'filename': filenames,
        'category': categories
    })
    df['category'] = df['category'].astype('str')
    
    categories = df.category.unique()
    categories = {i: category for i, category in enumerate(categories)}
    with open(os.path.join(context.out_path, map_filename), 'w') as f:
        f.write(json.dumps(categories))
        
    context.logger.info(categories)
    context.log_artifact('categories_map', src_path=map_filename)
    context.log_artifact(TableArtifact('file_categories', df=df, src_path=df_filename))


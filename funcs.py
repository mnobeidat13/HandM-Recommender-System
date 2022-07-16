import os 
from PIL import Image, ImageOps
import numpy as np
import pandas as pd


def get_item_image(item_id, resize=True, width=100, height = 150):
    
    path = 'results/images/'+item_id+'.jpeg'
    image = Image.open(path)
    
    if resize:
        basewidth = width
        wpercent = (basewidth / float(image.size[0]))
        hsize = int((float(image.size[1]) * float(wpercent)))
        image = image.resize((width, height), Image.ANTIALIAS)
    image = ImageOps.expand(image, 2)
        
    return image    


def get_rcmnds(customer_data):
    
    combined_rcmnds = customer_data.combined_rcmnds.values
    tfrs_rcmnds = customer_data.tfrs_rcmnds.values
    image_rcmnds = customer_data.image_rcmnds.values
    text_rcmnds = customer_data.text_rcmnds.values
    feature_rcmnds = customer_data.feature_rcmnds.values

    return image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds

def get_rcmnds_scores(customer_data):
    
    combined_rcmnds = customer_data.combined_scores.values
    tfrs_rcmnds = customer_data.tfrs_scores.values
    image_rcmnds = customer_data.image_scores.values
    text_rcmnds = customer_data.text_scores.values
    feature_rcmnds = customer_data.feature_scores.values

    return image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds

def get_rcmnds_images(image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds):
    
    combined_rcmnds = [get_item_image(str(i)) for i in combined_rcmnds]
    tfrs_rcmnds = [get_item_image(str(i)) for i in tfrs_rcmnds]
    image_rcmnds = [get_item_image(str(i)) for i in image_rcmnds]
    text_rcmnds = [get_item_image(str(i)) for i in text_rcmnds]
    feature_rcmnds = [get_item_image(str(i)) for i in feature_rcmnds]
    
    return image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds


def get_rcmnds_features(df, image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds):
    
    combined_rcmnds = df[df.article_id.isin(combined_rcmnds)][['product_type_name', 'colour_group_name', 'department_name']]
    tfrs_rcmnds = df[df.article_id.isin(tfrs_rcmnds)][['product_type_name', 'colour_group_name', 'department_name']]
    image_rcmnds = df[df.article_id.isin(image_rcmnds)][['product_type_name', 'colour_group_name', 'department_name']]
    text_rcmnds = df[df.article_id.isin(text_rcmnds)][['product_type_name', 'colour_group_name', 'department_name']]
    feature_rcmnds = df[df.article_id.isin(feature_rcmnds)][['product_type_name', 'colour_group_name', 'department_name']]
    
    return image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds
    

def get_rcmnds_desc(df, image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds):
    
    combined_rcmnds = df[df.article_id.isin(combined_rcmnds)].detail_desc
    tfrs_rcmnds = df[df.article_id.isin(tfrs_rcmnds)].detail_desc
    image_rcmnds = df[df.article_id.isin(image_rcmnds)].detail_desc
    text_rcmnds = df[df.article_id.isin(text_rcmnds)].detail_desc
    feature_rcmnds = df[df.article_id.isin(feature_rcmnds)].detail_desc
    
    return image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds
    
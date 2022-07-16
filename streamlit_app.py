import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from funcs import *
import streamlit.components.v1 as components

def main():

    st.set_page_config(layout="wide", initial_sidebar_state='expanded')
    github = """
    <a href="https://github.com/mnobeidat13", target="_blank">
      <img src="https://pbs.twimg.com/profile_images/1414990564408262661/r6YemvF9_400x400.jpg" alt="HTML tutorial" style="width:50px;height:50px;">

    </a>
    """

    kaggle = """
    <a href="https://www.kaggle.com/mohammedobeidat", target="_blank">
      <img src="https://miro.medium.com/max/3200/1*K5NPQiLmq30qmkySiVb5JQ.jpeg" alt="HTML tutorial" style="width:100px;height:50px;">

    </a>
    """

    linkedin = """
    <a href="https://www.linkedin.com/in/mnobeidat/", target="_blank">
      <img src="https://play-lh.googleusercontent.com/kMofEFLjobZy_bCuaiDogzBcUT-dz3BBbOrIEjJ-hqOabjK8ieuevGe6wlTD15QzOqw" alt="HTML tutorial" style="width:50px;height:50px;">

    </a>
    """
    
        
    sidebar_header = '''This is a demo to illustrate a recommender system that finds similar items to a given clothing article or recommend items for a customer using 4 different approaches:'''
    
    page_options = ["Find similar items",
                    "Customer Recommendations",
                    "Product Captioning",
                    'Documentation']
    
    st.sidebar.info(sidebar_header)


    
    page_selection = st.sidebar.radio("Try", page_options)
    articles_df = pd.read_csv('articles.csv')
    
    models = ['Similar items based on image embeddings', 
              'Similar items based on text embeddings', 
              'Similar items based discriptive features', 
              'Similar items based on embeddings from TensorFlow Recommendrs model',
              'Similar items based on a combination of all embeddings']
    
    model_descs = ['Image embeddings are calculated using VGG16 CNN from Keras', 
                  'Text description embeddings are calculated using "universal-sentence-encoder" from TensorFlow Hub',
                  'Features embeddings are calculated by one-hot encoding the descriptive features provided by H&M',
                  'TFRS model performes a collaborative filtering based ranking using a neural network', 
                  'A concatenation of all embeddings above is used to find similar items']

#########################################################################################
#########################################################################################

    if page_selection == "Find similar items":

        articles_rcmnds = pd.read_csv('results/articles_rcmnds.csv')

        articles = articles_rcmnds.article_id.unique()
        get_item = st.sidebar.button('Get Random Item')
        
        if get_item:
            
            rand_article = np.random.choice(articles)
            article_data = articles_rcmnds[articles_rcmnds.article_id == rand_article]
            rand_article_desc = articles_df[articles_df.article_id == rand_article].detail_desc.iloc[0]
            image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds = get_rcmnds(article_data)
            
            rcmnds = (image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds)
            
            scores = get_rcmnds_scores(article_data)
            features = get_rcmnds_features(articles_df, image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds)
            images = get_rcmnds_images(image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds)
            detail_descs  = get_rcmnds_desc(articles_df, image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds)
            
            st.sidebar.image(get_item_image(str(rand_article), width=200, height=300))
            st.sidebar.write('Article description')
            st.sidebar.caption(rand_article_desc)

            with st.container():     
                for i, model, image_set, score_set, model_desc, detail_desc_set, features_set, rcmnds_set in zip(range(5), models, images, scores, model_descs, detail_descs, features, rcmnds):
                    container = st.expander(model, expanded = model == 'Similar items based on image embeddings' or model == 'Similar items based on text embeddings')
                    with container:
                        cols = st.columns(7)
                        cols[0].write('###### Similarity Score')
                        cols[0].caption(model_desc)
                        for img, col, score, detail_desc, rcmnd in zip(image_set[1:], cols[1:], score_set[1:], detail_desc_set[1:],  rcmnds_set[1:]):
                            with col:
                                st.caption('{}'.format(score))
                                st.image(img, use_column_width=True)
                                if model == 'Similar items based on text embeddings':
                                    st.caption(detail_desc)
                                    
#########################################################################################
#########################################################################################

    if page_selection == "Product Captioning": 
        captions = pd.read_csv('caption_desc_embeds.csv', dtype={'id':str}).drop('Unnamed: 0', axis=1)
        
        
        get_item = st.sidebar.button('Get Random Item')      
        
        st.sidebar.warning('In this section you get try a transformer based model that generates product captions given its image')
        
            
        if get_item:
            
            
            
            rand_article = np.random.choice(captions.id)
            desc = captions[captions.id == rand_article].desc.iloc[0]
            caption = captions[captions.id == rand_article].caption.iloc[0].capitalize()
            
            cols = st.columns(2)
            with cols[0]:
                st.image(get_item_image(str(rand_article[1:]), resize=True, width=300, height=400))
            with cols[1]:
                with st.expander('Actual Product Description', expanded=True):
                    components.html(f"""
           <header>
            <h4 style="color: #253f4e;">{desc}</h4>
           </header>
            """)
                
                with st.expander('Generated Product Description', expanded=True):
                     components.html(f"""
           <header>
            <h4 style="color: #253f4e;">{caption}</h4>
           </header>
            """)
            
            
            
#########################################################################################
#########################################################################################
    if page_selection == "Customer Recommendations":
        
        customers_rcmnds = pd.read_csv('results/customers_rcmnds.csv')
        customers = customers_rcmnds.customer.unique()        
        
        get_item = st.sidebar.button('Get Random Customer')
        if get_item:
            st.sidebar.write('#### Customer history')

            rand_customer = np.random.choice(customers)
            customer_data = customers_rcmnds[customers_rcmnds.customer == rand_customer]
            customer_history = np.array(eval(customer_data.history.iloc[0]))

            image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds = get_rcmnds(customer_data)
            
            scores = get_rcmnds_scores(customer_data)
            features = get_rcmnds_features(articles_df, combined_rcmnds, tfrs_rcmnds, image_rcmnds, text_rcmnds, feature_rcmnds)
            images = get_rcmnds_images(combined_rcmnds, tfrs_rcmnds, image_rcmnds, text_rcmnds, feature_rcmnds)
            detail_descs  = get_rcmnds_desc(articles_df, image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds)
            
            rcmnds = (image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds)

            splits = [customer_history[i:i+3] for i in range(0, len(customer_history), 3)]
                            
            for split in splits:
                with st.sidebar.container():
                    cols = st.columns(3)
                    for item, col in zip(split, cols):
                        col.image(get_item_image(str(item), 100))
                    

            with st.container():            
                for i, model, image_set, score_set, model_desc, detail_desc_set, features_set, rcmnds_set in zip(range(5), models, images, scores, model_descs, detail_descs, features, rcmnds):
                    container = st.expander(model, expanded=True)
                    with container:
                        cols = st.columns(7)
                        cols[0].write('###### Similarity Score')
                        cols[0].caption(model_desc)
                        for img, col, score, detail_desc, rcmnd in zip(image_set[1:], cols[1:], score_set[1:], detail_desc_set[1:],  rcmnds_set[1:]):
                            with col:
                                st.caption('{}'.format(score))
                                st.image(img, use_column_width=True)
                                

#########################################################################################  
#########################################################################################

    if page_selection == "Documentation":

        with st.container():
            cols = st.columns(3)
            with cols[0]:
                components.html(github)
            with cols[1]:
                    components.html(linkedin)
            with cols[2]:
                components.html(kaggle)
                
                
        components.html(
            """
           <header>
           

        <h2>H&M Personalized Fashion Recommendations</h2>
        
        This is a demonstration of my work on recommender systems in a competition hosted by H&M on 
        <a href="https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations", target="_blank">Competition Page</a>.
        
        <br>
        
        The goal of this competition is to build a recommender system to generate recoomendations for customers based on their purchase history.
        <br>
        <br>
        In this project, I used two different approaches:
        <ul>
        <li>Content Based Filtering</li>
        <li>Collaborative Filtering</li>
        </ul>
        
        <br>
        The idea in recommendation systems is to build representations of customers and items. These representations are called embeddings which are a projection of customers and items into an N-dimensional space.
        <br>
        <br>
        Content-based filtering uses item features to recommend other items similar to what the user likes, based on their previous actions or explicit feedback. These hand-engineered features are used to build items embeddings. You also represent the user in the same feature space. In this project I used the features provided by H&M that describe each item like color, texture, what piece of clothing it is and so on.
        
        <br>
        <br>
        Besides hand-engineered features, I built two other sets of embeddings from images and text description of each item. I fed the images through a pre-trained "VGG16" convolutional neural network from 
        
        <a href="https://keras.io/api/applications/", target="_blank">Keras Applications API</a> to get image embeddings.
        
        
        For text embeddings I used "universal-sentence-encoder", a pre-trained model from 
        
        <a href="https://tfhub.dev/google/universal-sentence-encoder/4", target="_blank">TensorFlow Hub.</a> 
        
        <br>
        <br>
        Collaborative filtering uses item-user interaction matrix to adress some of the limitations of content-based filtering which is the need to hand engineer items features. The interaction matrix is then factorized into two sets of N-dimensional embeddings, one for users and one for items.
        
        <br>
        <br>
        To read more on recommendation systems head to 
        <a href="https://developers.google.com/machine-learning/recommendation", target="_blank">Google Developers.</a> 
        <br>

        <br>
        <br>
        In product captioning section, I used a transformer based model that generates captions given product images as input. Captions are capped             at 10 words.
        <br>
        Follow the tutorial on Keras Code Examples website:
        <a href="https://keras.io/examples/vision/image_captioning/", target="_blank">Image Captioning.</a> 

        <br>
        
        <h2>A list of notebooks and datasets used in this project</h2>
        <ul>
        
        <li>Product similarity with text embeddings

        <a href="https://www.kaggle.com/code/mohammedobeidat/product-similarity-with-text-embeddings/notebook", target="_blank">
        https://www.kaggle.com/code/mohammedobeidat/product-similarity-with-text-embeddings</a> 
        
        </li>
        <li>Customer and item embeddings from text

        <a href="https://www.kaggle.com/code/mohammedobeidat/h-m-article-and-customer-embeddings-from-text-desc", target="_blank">
        https://www.kaggle.com/code/mohammedobeidat/h-m-article-and-customer-embeddings-from-text-desc</a> 
        
        </li>
        <br>
        <li>
        Product Similarity from images

        <a href="https://www.kaggle.com/code/mohammedobeidat/finding-similar-items-with-image-embeddings-knn", target="_blank">
        https://www.kaggle.com/code/mohammedobeidat/finding-similar-items-with-image-embeddings-knn</a> 

        </li>
        <li>Item embeddings from images

        <a href="https://www.kaggle.com/code/mohammedobeidat/product-embeddings-from-images-keras-vgg16", target="_blank">
        https://www.kaggle.com/code/mohammedobeidat/product-embeddings-from-images-keras-vgg16</a> 
        </li>
        
        <li>Customer embeddings from images

        <a href="https://www.kaggle.com/datasets/mohammedobeidat/hm-product-image-embeddings", target="_blank">
        https://www.kaggle.com/datasets/mohammedobeidat/hm-product-image-embeddings</a> 
        </li>
        <br>
        <li>
        Product Similarity from features

        <a href="https://www.kaggle.com/code/mohammedobeidat/content-based-filtering-with-pca", target="_blank">
        https://www.kaggle.com/code/mohammedobeidat/content-based-filtering-with-pca</a> 

        </li>
        <li>Customer Embeddings from Features
        <a href="https://www.kaggle.com/code/mohammedobeidat/customer-embeddings-from-features", target="_blank">
        https://www.kaggle.com/code/mohammedobeidat/customer-embeddings-from-features</a> 
        </li>
        
        <li>Item Embeddings from Features
        <a href="https://www.kaggle.com/code/mohammedobeidat/article-embeddings-from-features", target="_blank">
        https://www.kaggle.com/code/mohammedobeidat/article-embeddings-from-features</a> 
        
        </li>
        <br>
        <li>
        Product Similarity with TFRS
        <a href="https://www.kaggle.com/code/viji1609/h-m-basic-retrieval-model-tf-recommender/notebook", target="_blank">
        https://www.kaggle.com/code/viji1609/h-m-basic-retrieval-model-tf-recommender/notebook</a> 
        <br>
        This notebook is credited to 
        <a href="https://www.kaggle.com/viji1609", target="_blank">
        https://www.kaggle.com/viji1609</a> 
        </li>
        <br>
        <li>
        
        Comparing all 4 Different Approaches

        <a href="https://www.kaggle.com/code/mohammedobeidat/comparing-4-different-approaches", target="_blank">
        https://www.kaggle.com/code/mohammedobeidat/comparing-4-different-approaches</a> 
        </li>
        
        <br>
        <li>
        
        Prodcuct Captioning

        <a href="https://www.kaggle.com/code/mohammedobeidat/h-m-product-captioning", target="_blank">
        https://www.kaggle.com/code/mohammedobeidat/h-m-product-captioning</a> 
        </li>
        
        </ul>
        </header>
        
        
        
            """,
            height=1000,
)

if __name__ == '__main__':
    main()

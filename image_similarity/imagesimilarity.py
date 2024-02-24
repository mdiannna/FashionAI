from PIL import Image
from keras.applications import vgg16
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
from keras.applications.imagenet_utils import preprocess_input
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class ModelNotFoundError(Exception):
    pass

class ImageSimilarity:
    def __init__(self, model_path=None, imgs_model_width=224, imgs_model_height=224):
        """Initialize ImageSimilarity class.
        
        Args:
            model_path (str, optional): The path for Image recognition/feature extraction model.
            imgs_model_width (int): The width of images for model.
            imgs_model_height (int): The height of images for model.
        """
        self.__model_path = model_path
        self.__imgs_model_width = imgs_model_width
        self.__imgs_model_height = imgs_model_height
        self.model = self.load_model(model_name='vgg16', from_keras=True)
        print(self.model.summary())

    def compute_similarity_btw_features(self, img1_features, img2_features):
        """Compute similarity score between the features of two images already extracted.
        
        Args:
            img1_features (numpy.ndarray): Features extracted from image 1.
            img2_features (numpy.ndarray): Features extracted from image 2.
        
        Returns:
            float: The similarity score.
        """
        arr_features = np.array([img1_features, img2_features])
        arr_features = arr_features.reshape(2, arr_features.shape[2])
        cosSimilarities = cosine_similarity(arr_features)
        score = cosSimilarities[0][1]
        return score

    def compute_similarity(self, img_path1, img_path2):
        """Compute similarity score between two images.
        
        Args:
            img_path1 (str): Path where first image is stored.
            img_path2 (str): Path where second image is stored.
        
        Returns:
            float: The similarity score.
        """
        img1_features = self.extract_features(self.model, img_path1)
        img2_features = self.extract_features(self.model, img_path2)
        return self.compute_similarity_btw_features(img1_features, img2_features)

    def get_most_similar(self, base_img_path, imgs_path, nr_similar_imgs=3, max_compare_imgs=10):
        """Get the most similar images to the image specified in img_path.
        
        Args:
            base_img_path (str): Path where the image is stored.
            imgs_path (str): Path where the other images are stored.
            nr_similar_imgs (int): Number of top similar images to return (default is 3).
        
        Returns:
            list: A list of similar images with their similarity score.
        """
        imgs_to_compare = [os.path.join(imgs_path, x) for x in os.listdir(imgs_path) if 
                           (not os.path.join(imgs_path, x) == base_img_path) and 
                           (("png" in x) or ("jpg" in x) or ("jpeg" in x))]
        
        
        base_img_features = self.extract_features(self.model, base_img_path)
        all_similarities = [self.compute_similarity(base_img_path, img_to_compare) for img_to_compare in imgs_to_compare]
        
        #if the similarity is close to 1, it means it's the same image, avoid this case:
        all_similarities = list(filter(lambda a: a <=0.99, all_similarities))

        print("all similarities:", all_similarities)
        most_similar_idx = np.argmax(all_similarities)

        # sort decreasing by indices:
        sorted_similarities = np.argsort(-np.array(all_similarities))

        
        print("sorted similarities:", sorted_similarities)
        
        
        # #filter only the top 3 similar images:
        # sorted_similarities = sorted_similarities[:nr_similar_imgs]

        imgs_to_compare = np.array(imgs_to_compare)

        return most_similar_idx, imgs_to_compare[most_similar_idx], imgs_to_compare[sorted_similarities]

    def load_model(self, model_name='vgg16', from_keras=True):
        """Load model for image similarity.
        
        Args:
            model_name (str): The name of the model to load.
            from_keras (bool): Specifies if the model loads from keras or not.
        
        Returns:
            keras.models.Model: The loaded model.
        
        Raises:
            ModelNotFoundError: If the specified model is not found.
        """
        if from_keras == True and model_name.lower() == 'vgg16':
            vgg_model = vgg16.VGG16(weights='imagenet')
            feat_extractor = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer("fc2").output)
            return feat_extractor
        raise ModelNotFoundError

    def extract_features(self, model, img_path):
        """Extract features using model from one image path.
        
        Args:
            model (keras.models.Model): The model for feature extraction from images.
            img_path (str): The path where the image is stored.
        
        Returns:
            numpy.ndarray: Array of numeric features extracted from the image.
        """
        original = load_img(img_path, target_size=(self.imgs_model_width, self.imgs_model_height))
        numpy_image = img_to_array(original)
        image_batch = np.expand_dims(numpy_image, axis=0)
        img_processed = preprocess_input(image_batch.copy())
        img_features = model.predict(img_processed)
        return img_features

    def extract_features_batch_imgs(self, model, imgs_paths_lst):
        """Extract features for all the image paths from a list.
        
        Args:
            model (keras.models.Model): The model for feature extraction from images.
            imgs_paths_lst (list): The list of paths where images are stored.
        
        Returns:
            numpy.ndarray: Array of numeric features extracted from the images from list.
        """
        imported_images = [np.expand_dims(img_to_array(load_img(img_path, target_size=(self.imgs_model_width, self.imgs_model_height))), axis=0) 
                           for img_path in imgs_paths_lst]
        imgs = np.vstack(imported_images)
        processed_imgs = preprocess_input(imgs.copy())
        imgs_features = model.predict(processed_imgs)
        return imgs_features

    @property
    def model_path(self):
        return self.__model_path

    @property
    def imgs_model_width(self):
        return self.__imgs_model_width

    @property
    def imgs_model_height(self):
        return self.__imgs_model_height

    @model_path.setter
    def model_path(self, model_path):
        self.__model_path = model_path

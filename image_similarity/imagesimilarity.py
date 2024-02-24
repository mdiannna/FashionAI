from PIL import Image
from keras.applications import vgg16
from keras.preprocessing.image import load_img,img_to_array
from keras.models import Model
from keras.applications.imagenet_utils import preprocess_input

from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd



class ModelNotFoundError(Exception):
    pass

class ImageSimilarity:    
    def __init__(self, model_path=None, imgs_model_width = 224, imgs_model_height = 224):
        """ 
        Init class

            Parameters:
            ----------
                imgs_path(str) - the path to folder where images are stored
                model_path (str) [optional] - the path for Image recognition/feature extraction model
                imgs_model_width(int) - the width of images for model
                imgs_model_height(int) - the heights of images for model
        """
        self.__model_path = model_path
        # parameters setup

        self.__imgs_model_width = imgs_model_width
        self.__imgs_model_height = imgs_model_height

        self.model = self.load_model(model_name='vgg16', from_keras=True)

        print(self.model.summary())
    
    
    def compute_similarity_btw_features(self, img1_features, img2_features):
        """ 
        Compute similarity score between the features of 2 images already extracted
            Parameters:
            ----------
                img1_features (str) - features extracted from img1
                img2_features (str) - features extracted from img2
            Returns:
            ----------
                score (float) - the similarity score (TODO: decide unit - percentages or other - standardizat cumva ???)
        
        TODO: decis algoritm (cosine similarity sau jaccard similarity sau Jensen-Shannon distance sau Earth Mover distance etc)
        """

        score = 0

        print("img1 features:", img1_features, "shape: ", img1_features.shape)
        print("img2 features:", img2_features, "shape: ", img2_features.shape)

        arr_features = np.array([img1_features, img2_features])
        arr_features = arr_features.reshape(2, arr_features.shape[2])
        print("arr features:", arr_features)
        
        print("arr features shape:", arr_features.shape)

        # TODO: try catch or return 0 if error
        cosSimilarities = cosine_similarity(arr_features)
        score = cosSimilarities[0][1]
       
        return score     


    # TODO: parametru metric sa poata fi setat!
    # TODO: poate de pus si db path, sau la init class sa fie, sau ca parametri cumva
    def compute_similarity(self, img_path1, img_path2):
        """ 
        Compute similarity score between 2 images
            Parameters:
            ----------
                img_path1 (str) - path where first image is stored
                img_path2 (str) - path were second image is stored
            Returns:
            ----------
                score (float) - the similarity score (TODO: decide unit - percentages or other)
        
        TODO: decis algoritm (cosine similarity sau jaccard similarity sau Jensen-Shannon distance sau Earth Mover distance etc)
        """

        score = 0

        #TODO: check if it is in db, if not, extract features
        img1_features = self.extract_features(self.model, img_path1)
        img2_features = self.extract_features(self.model, img_path2)

        return self.compute_similarity_btw_features(img1_features, img2_features)
    
    
    
    # TODO: test this function!
    # TODO: add more metrics if needed, not just cos similarity
    def compute_similarity_batch(self, imgs_features, imgs_paths_lst, verbose=False):
        """ 
        Compute the similarity between multiple images 
        -------
        parameters:
            - imgs_features(np array) - the extracted features from images (obtained with the function extract_features_batch_imgs)
            - imgs_paths_lst(list of str) - paths to the image files
            - verbose(bool) - if True prints more output
        -------
        returns:
            - cos_similarities_df (pd dataframe) - the similarities computed between each pairs of images in form of dataframe table
        """
        # compute cosine similarities between images
        cosSimilarities = cosine_similarity(imgs_features)

        # store the results into a pandas dataframe
        cos_similarities_df = pd.DataFrame(cosSimilarities, columns=imgs_paths_lst, index=imgs_paths_lst)
        
        if verbose:
            print(cos_similarities_df.head())

        return cos_similarities_df


    def get_most_similar(self, base_img_path, imgs_path, nr_similar_imgs=3):
        """ 
        Get the most similar images to the image specified in img_path
            Parameters:
            ----------
                base_img_path (str) - path where the image is stored
                imgs_path(str) - path where are the other images (TODO: decis if needed)
                nr_similar_imgs (int) - how many top similar images to return (by default 3) 
            Returns:
            ----------
                results (list of dictionaries) - a list of similar images with their similarity score, in the form
                    [{"similar_image": "path"(str), "score": float},  .... ]        
        """
        
        score = 0
       
        #IDEA: later, additionally to img_path add a list of potential images, not to search through all of them!!
        imgs_to_compare = [imgs_path + x for x in os.listdir(imgs_path) if (not (imgs_path + x)==base_img_path) and (("png" in x) or ("jpg" in x)or ("jpeg" in x))]
        print("imgs_to_compare[0]:", imgs_to_compare[0])
        print("number of images:",len(imgs_to_compare))

        # features = self.extract_features(self.model, imgs_to_compare[0])
        base_img_features = self.extract_features(self.model, base_img_path)
        print("base_img features:", base_img_features)

        all_similarities = []
        for img_to_compare in imgs_to_compare:
            similarity = self.compute_similarity(base_img_path, img_to_compare)
            all_similarities.append(similarity)

        print("all similarities:", all_similarities)
        print("base image path:", base_img_path)

        # raise NotImplementedError
        most_similar_idx = np.argmax(all_similarities)
        return most_similar_idx, imgs_to_compare[most_similar_idx]
        

    def load_model(self, model_name='vgg16', from_keras=True):
        """ 
        Load model for image similarity
            Parameters:
            ----------
                model_name(str) - the name of the model to load
                from_keras(bool) - specifies if the model loads from keras or not
        """
        if from_keras==True and model_name.lower() =='vgg16':
            # load the model
            vgg_model = vgg16.VGG16(weights='imagenet')

            # remove the last layers in order to get features instead of predictions
            feat_extractor = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer("fc2").output)
        
            return feat_extractor
        #else
        # TODO: implement loading from custom model_path!!
        # self.model_path(str) [optional] - specifies the model_path if from_keras is false TODO: implement

        raise ModelNotFoundError


    # TODO: set model as internal class parameter???
    def extract_features(self, model, img_path, verbose=False):
        """ 
        Extract features using model from one image path
        ----------
        parameters:
            TODO: set model as internal class parameter???
            - model- the model for feature extraction from images
            - img_path(str) - the path where image is stored
            - verbose(bool) -  if True prints more output
        ----------
        returns:
            - img_features(np array) - array of numeric features extracted from image
        """

        # load an image in PIL format
        original = load_img(img_path, target_size=(self.imgs_model_width, self.imgs_model_height))
        
        # show img
        # plt.imshow(original)
        # plt.show()
        print("image loaded successfully!")

        # convert the PIL image to a numpy array
        numpy_image = img_to_array(original)

        # convert the image / images into batch format
        # expand_dims will add an extra dimension to the data at a particular axis
        # we want the input matrix to the network to be of the form (batchsize, height, width, channels)
        # thus we add the extra dimension to the axis 0.
        image_batch = np.expand_dims(numpy_image, axis=0)

        if verbose:
            print('image batch size', image_batch.shape)

        # prepare the image for the VGG model
        img_processed = preprocess_input(image_batch.copy())

        # get the extracted features
        img_features = model.predict(img_processed)

        if verbose:
            print("features successfully extracted!")
            print("number of image features:",img_features.size)

        return img_features


    # TODO: acest proces se face doar la inceput, sau pentru fiecare imagine individual si se salveaza/updateaza in db,
    # in asa mod nu va fi complexitate mare si timp mare
    #TODO: se poate de facut functie import_images cu mai multe imagini, dar nush daca trebuie
    #TODO!!! fiecare user trebuie sa aiba separat folderul de fisiere/ bd 
    def extract_features_batch_imgs(self, model, imgs_paths_lst):
        """ 
        Extract features for all the image paths from a list 

        ----------
        parameters:
            TODO: set model as internal class parameter???
            - model- the model for feature extraction from images
            - imgs_paths_lst (list of str) - the list of paths where images are stored
        ----------
        returns:
            - img_features(np array) - array of numeric features extracted from the images from list
        """
        # load all the images and prepare them for feeding into the CNN

        imported_images = []

        for img_path in imgs_paths_lst:
            original = load_img(img_path, target_size=(self.imgs_model_width, self.imgs_model_height))
            np_image = img_to_array(original)
            image_batch = np.expand_dims(np_image, axis=0)
            
            imported_images.append(image_batch)
            
        imgs = np.vstack(imported_images)

        processed_imgs = preprocess_input(imgs.copy())

        # extract the images features
        imgs_features = model.predict(processed_imgs)

        print("features successfully extracted!")
        imgs_features.shape

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
    
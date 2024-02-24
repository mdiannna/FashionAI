from imagesimilarity import ImageSimilarity

# imgSim = ImageSimilarity()
imgSim = ImageSimilarity() 
 



img_folder = 'uploads/'
similarity_score1 = imgSim.compute_similarity(img_folder + "red_shirt1.jpg", img_folder + "red_shirt2.jpg")
print("similarity score 1:", similarity_score1)


most_similar = imgSim.get_most_similar(img_folder + "red_shirt1.jpg", 'uploads/')
print("most similar img:", most_similar)


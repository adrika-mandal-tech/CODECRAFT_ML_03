import os
# Make sure OpenCV is installed: pip install opencv-python
import cv2   # type: ignore # OpenCV for image processing
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split # type: ignore # For splitting the dataset into training and testing sets
from sklearn.svm import SVC # type: ignore # Support Vector Classifier for machine learning

# This code reads images from a directory, processes them, and saves the data into a pickle file.
# It also loads the data from the pickle file and prepares it for training a machine learning model

'''
dir='G:\ML&DL Projects\SVM_Dog&Cat_ImgClassifier\Data_Pet_Images'

categories= ['Cat','Dog']
data=[]
# Loop through each category and read the images
# Display the images using OpenCV

for category in categories:
    path=os.path.join(dir, category)
    label= categories.index(category) # Assign a label to each category
    
    for img in os.listdir(path):
        img_path = os.path.join(path,img)
        pet_img= cv2.imread(img_path,0) # Read the image in grayscale
        try:       
            pet_img= cv2.resize(pet_img, (50, 50)) # Resize the image to 100x100 pixels
            image=np.array(pet_img).flatten() # Convert the image to a numpy array
            data.append([image, label]) # Append the image and label to the data list
        except Exception as e:
            pass
            
#print(f"Total number of images: {len(data)}")

pick_in = open('data1.pickle', 'wb') # Open a file to save the data
pickle.dump(data, pick_in) # Save the data to the file
pick_in.close() # Close the file

'''

pick_in = open('data1.pickle','rb') # Open a file to read the data
data=pickle.load(pick_in) # Load the data from the file
pick_in.close() # Close the file

random.shuffle(data) # Shuffle the data to ensure randomness
features = [] 
labels = [] 

for feature, label in data:
    features.append(feature) 
    labels.append(label) 
    
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2) # 80% training and 20% testing


'''
C=1.0:
This is the regularization parameter. A smaller value makes the decision surface smoother, while a larger value tries to classify all training examples correctly (risking overfitting).
kernel='poly':
This tells the SVC to use a polynomial kernel. Kernels transform the input data into a higher-dimensional space, allowing the model to find non-linear decision boundaries.
gamma='auto':
This sets the kernel coefficient for 'rbf', 'poly', and 'sigmoid' kernels. With 'auto', gamma is set to 1 / n_features, where n_features is the number of features in your data
'''
'''
model=SVC(C=1.0, kernel='poly', gamma='auto') # Create a Support Vector Classifier model
model.fit(X_train, y_train) 
pick = open('model.sav', 'wb') # Open a file to save the trained model
pickle.dump(model, pick) # Save the trained model to a file
pick.close() # Close the file
'''
# model.sav is the file where the trained model is saved is created

pick= open('model.sav', 'rb') # Open the file to read the trained model
model = pickle.load(pick) # Load the trained model from the file
pick.close() # Close the file

prediction=model.predict(X_test) # Make predictions on the test set
accuracy=model.score(X_test, y_test) # Calculate the accuracy of the model
categories= ['Cat','Dog']
print(f"Accuracy: {accuracy*100:.2f}%") # Print the accuracy of the model
print(f"Prediction: {categories[prediction[0]]}") # Print the predicted category for the first test image
mypet=X_test[0].reshape(50, 50) # Reshape the first test image for display
plt.imshow(mypet, cmap='gray') # Display the image in grayscale
plt.show()



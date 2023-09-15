### Macaw Project
##### A Machine Learning project with a focus on Computer Vision.

#### Author: Sarah E. Yack
#### Date: 09/15/2023

#### Version: 0.0.1

#### Summary:
1. This is my first project in the Macaw Project.
2. This is being built using techniques learned in Coursera's Deep Learning Specialization.
3. The project is focused on Computer Vision and will be using basic forward and backward propagation techniques.

#### Description:
This is my first project in the Macaw Project. This is being built using techniques learned in Coursera's Deep Learning Specialization.
The project is focused on Computer Vision and will be using basic forward and backward propagation techniques. It was originally built using Conv2D and Dense layers, but I'm reworking it to use more basic functions for educational purposes.  
The idea for this project was originally inspired by an assignment in the Coursera Machine Learning For All course, where you uploaded your own images and had them classified by a trained model.

#### Implementation:
The first thing I did was I translated the images I had collected into data I could use for the model, however I wasn't and if I'm honest still don't know exactly how this worked, so I plan to study this process more in the future.
The second thing I did was creating a model using the TensorFlow library, I then used the model to classify the images I had collected. However, I found that the model wasn't able to classify the images I had collected properly and so decided to rework the whole process and use it to teach myself the more basic functions. So I reworked the file structure and added modules and helper functions so I could have a bit more control over the process. However, I'm still working on this step and haven't yet moved on to training and predicting with the model.  

#### Things to Know:
The image-data folder holds four files: Imagedata.py, image_data.pkl, image_data_test.pkl, and image_data_test.pkl. These are currently using web images from my folders.  
All relevant folder paths are used and stored using the config.py file in the helper folder under src.  
utility_functions.py holds the basic model building functions used in the project and is stored in the helper folder under src. This is things like load_data(), load_model(), save_model(), etc.  
my_model is the folder used to save my own model, and I'm unsure if it can be used for a different setup with the model.  
Currently the two files, imageml.py and nn-model.py are redundant and need to be combined into one file. However, I'm still in the process of doing this. 
All relevant model functions in regards to forward and backward propagation and computing cost for example are in the model_functions.py file.
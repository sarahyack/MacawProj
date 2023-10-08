### Macaw Project
### A Machine Learning project with a focus on Computer Vision.

#### Author: Sarah E. Yack
#### Date: 10/08/2023

#### Version: 1.0.0

#### Summary:
1. This is my first machine learning project that deals with images.
2. I'm building this using techniques learned in Coursera's Deep Learning Specialization.
3. The project is focused on Computer Vision and will be using basic forward and backward propagation techniques.

#### Description:
This is my first machine learning project that deals with images. I'm building this using techniques learned in Coursera's Deep Learning Specialization.
The project is focused on Computer Vision and will be using basic forward and backward propagation techniques. It was originally built using TensorFlow's Conv2D and Dense layers, but I'm reworking it to use more basic functions for educational purposes.  
The idea for this project was originally inspired by an assignment in the Coursera Machine Learning For All course, where you uploaded your own images and had them classified by a trained model.

#### Implementation:
The first thing I did was I translated the images I had collected into data I could use for the model, and stored them in a folder called image-data.
The second thing I did was creating a model using the TensorFlow library, I then used the model to classify the images I had collected. However, I found that the model wasn't able to classify the images I had collected properly and so decided to rework the whole process and use it to teach myself the more basic functions. So I reworked the file structure and added modules and helper functions so I could have a bit more control over the process, and little less clutter in my main file.

Now the structure of the model is 5-layer (including Input and Output layers) model, with the final layer being a softmax layer. Initially I had the wrong architecture, as I was trying to use naive softmax for the last layer. However, I misunderstood the softmax function and ended up changing around the cost function and various parts of the forward propogation and y-encoding process to make the softmax function more effective. The issue was that I didn't know the softmax function needed mutually exclusive classes.

Currently, I've gotten the model to run, however, now I need to properly optimize it.  

#### Things to Know:
The image-data folder holds four files: Imagedata.py, image_data.pkl, image_data_test.pkl, and image_data_test.pkl. These are currently using web images from my folders. It's best to run the .py files from the terminal location of image-data.  
All relevant folder paths are used and stored using the config.py file in the helper folder under src.  
utility_functions.py holds the basic model building functions used in the project and is stored in the helper folder under src. This is things like load_data(), load_model(), save_model(), etc.  
In the model folder, there are three files: model_functions.py, imageml.py, and nn_model.py. imageml.py is the main file with the function calls for the model. nn_model.py is the file that contains the Model Function. model_functions.py is the file that contains the forward and backward propagation functions, as well as various others such as the cost function.  
#### Current Hyperparamters:
Learning Rate: 0.001
Beta1: 0.9
Beta2: 0.99
Epochs: 1000
Epsilon: 1e-8


[![built with Codeium](https://codeium.com/badges/main)](https://codeium.com)
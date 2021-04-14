# For Step 1
import onnxruntime                                                  # For running the onnx model
from keras.preprocessing.image import img_to_array, load_img        # For preprocessing the image input
from skimage.transform import resize                                # For resizing the image input
import numpy as np                                                  # For manipulating the image input

# For Step 2
from prefect import task, Flow                                      # For workflow management
from pathlib import Path                                            # For moving processed image to another directory
import os                                                           # For manipulating file and directory path
import json                                                         # For storing in JSON format
from datetime import datetime, timedelta                            # For naming JSON file & scheduling task
from prefect.schedules import Schedule                              # For scheduling task
from prefect.schedules.clocks import CronClock                      # For scheduling task

# For Step 1
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Step 1: Create a Function
def predictHelmetProb(image_path):

    """ Read model path """
    model_path = 'helmetfromperson_yesno_mobilenetv2_140_128x128_11-3-2021.onnx'

    """ Load image"""
    img = load_img(image_path)

    """ Preprocess the image """
    ximg = img_to_array(img)                                        # Convert image to array
    ximg128 = resize(ximg / 255, (128, 128, 3), anti_aliasing=True) # Resize the image to 128x128
    ximg = ximg128[np.newaxis, :, :, :]                             # Add a new axis
    ximg = ximg.astype(np.float32)                                  # Convert it to np float
    ximg = np.swapaxes(ximg, 3,1)                                   # Adjusting the axes to meet the input requirement

    """ Create inference session """
    session = onnxruntime.InferenceSession(model_path)

    """ Get the prediction (without softmax) for the image from the model """
    input_name = session.get_inputs()[0].name
    prediction_before_softmax = session.run(None, {input_name: ximg})

    #print(prediction_before_softmax)

    """ Get the probabilities by passing through a softmax function """
    wear_helmet_prob, not_wear_helmet_prob, undetermined_prob = softmax(prediction_before_softmax)[0][0]
    
    return wear_helmet_prob, not_wear_helmet_prob, undetermined_prob

""" Test for Step 1 Function """
#wear_helmet_prob, not_wear_helmet_prob, undetermined_prob = predictHelmetProb('test.jpg')
#print("Probability of 'Wearing Helmet' is ", wear_helmet_prob)
#print("Probability of 'Not Wearing Helmet' is ", not_wear_helmet_prob)
#print("Probability of 'Undetermined' is ", undetermined_prob)



# Step 2: Create a Scheduled Task
""" Create a Task Definition"""
@task                       
def inferenceDirectory(): 

    """ Get user-specified directory path for unprocessed images """
    directory = r'C:\Users\user\Desktop\viAct Internship\Task 1\image_data'

    """ Get user-specified directory path for processed images """
    new_directory = r'C:\Users\user\Desktop\viAct Internship\Task 1\processed_image_data'

    """ Create a dictionary to store the predictions during runtime """
    store_prob = {"predictions": []}

    """ Create a list to store all filenames in the directory for unprocessed images """
    file_name_list = []

    """ Get all image filenames in the filename list """
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            file_name_list.append(filename)
        else:
            continue

    """ Loop through all the images in the directory"""
    for filename in file_name_list:
        wear_helmet_prob, not_wear_helmet_prob, undetermined_prob = predictHelmetProb(os.path.join(directory, filename)) # Get the predictions from the model
        
        wear_helmet_prob = str(wear_helmet_prob)            # Convert float to string in order to store as json later
        not_wear_helmet_prob = str(not_wear_helmet_prob)
        undetermined_prob = str(undetermined_prob)
        
        store_prob["predictions"].append({filename: [wear_helmet_prob, not_wear_helmet_prob, undetermined_prob]}) # Store prediction of image in dictionary
        
        Path(os.path.join(directory, filename)).rename(os.path.join(new_directory, filename))       # Move the processed image to new directory

    """ Convert the dictionary to json format """   
    json_data = json.dumps(store_prob)

    """ Get the datetime to name the json file for storing prediction every day """
    name = datetime.utcnow().strftime('%Y-%m-%d %H_%M_%S.%f')[:-3]
    json_file_name = "json_prediction_data/%s.json"% name

    """ Create a new json file for the prediction result """
    with open(json_file_name, 'x') as json_file:
        json.dump(json_data, json_file)

""" Schedule the flow so that the task will run at 2am every day """
schedule = Schedule(clocks=[CronClock("0 2 * * *")])

""" Create a flow that run the task """
with Flow('ModelInferenceWorkflow', schedule) as flow:
    inferenceDirectory()

""" Run the flow """
flow.run()

import onnxruntime
from keras.preprocessing.image import img_to_array, load_img
from skimage.transform import resize
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def predictHelmetProb(image_path):

    """Read model path"""
    model_path = 'helmetfromperson_yesno_mobilenetv2_140_128x128_11-3-2021.onnx'

    """ Load image"""
    img = load_img(image_path)

    """Preprocess the image"""
    ximg = img_to_array(img)                                        # Convert image to array
    ximg128 = resize(ximg / 255, (128, 128, 3), anti_aliasing=True) # Resize the image to 128x128
    ximg = ximg128[np.newaxis, :, :, :]                             # Not sure what this does
    ximg = ximg.astype(np.float32)                                  # Not sure what this does
    ximg = np.swapaxes(ximg, 3,1)                                   # Adjusting the axes to meet the input requirement

    """Create inference session"""
    session = onnxruntime.InferenceSession(model_path)

    """Get the prediction (without softmax) for the image from the model"""
    input_name = session.get_inputs()[0].name
    prediction_before_softmax = session.run(None, {input_name: ximg})

    #print(prediction_before_softmax)

    """Get the probabilities by passing through a softmax function"""
    wear_helmet_prob, not_wear_helmet_prob, undetermined_prob = softmax(prediction_before_softmax)[0][0]
    
    return wear_helmet_prob, not_wear_helmet_prob, undetermined_prob

wear_helmet_prob, not_wear_helmet_prob, undetermined_prob = predictHelmetProb('test.jpg')
print("Probability of 'Wearing Helmet' is ", wear_helmet_prob)
print("Probability of 'Not Wearing Helmet' is ", not_wear_helmet_prob)
print("Probability of 'Undetermined' is ", undetermined_prob)


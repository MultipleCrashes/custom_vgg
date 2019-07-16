from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.applications.vgg16 import decode_predictions

model = VGG16()
img = load_img('photo.jpeg', target_size=(224, 224))
loaded_img = img_to_array(img)
reshaped_img = loaded_img.reshape(1, loaded_img.shape[0], loaded_img.shape[1],loaded_img.shape[2])
processed_img_for_vgg = preprocess_input(reshaped_img)
predicted_obj = model.predict(processed_img_for_vgg)
# Get predicted value
output_result = decode_predictions(predicted_obj)
print('final results with probability ', output_result[0][0])

import tensorflow as tf
import pandas as pd
from keras.applications.inception_resnet_v2 import preprocess_input

model = tf.keras.models.load_model('./dog_pose_v3')

filepaths =  'data/train_data_set/not_pose/n02099712_6997.jpg'

IMAGE_SIZE  = (224, 224)
test_image = tf.keras.utils.load_img(filepaths
                            ,target_size =IMAGE_SIZE )
test_image = tf.keras.utils.img_to_array(test_image)
test_image = test_image.reshape((1, test_image.shape[0], test_image.shape[1], test_image.shape[2]))
test_image = preprocess_input(test_image)
prediction = model.predict(test_image)

if prediction[0][0] > 0.5:
    print(1)
else:
    print(0)

#0일때 not_pose
#1일때 sigor_pose
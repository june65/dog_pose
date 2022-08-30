import numpy as np
import tensorflow as tf
import keras
import glob

model = keras.models.load_model('./dog_pose')
dog_images_up = glob.glob('.\\data\\val_data_set\\sigor_pose\\*')
dog_images_down = glob.glob('.\\data\\val_data_set\\not_pose\\*')
score = [0,0]
i = 0
for image in dog_images_up:

    img=tf.keras.preprocessing.image.load_img(image,target_size=(300, 300))
    x=tf.keras.preprocessing.image.img_to_array(img)
    x=np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images)
    if classes[0] > 0.5 :
        score[0] += 1    
    else :
        score[1] += 1
    i += 1

for image in dog_images_down:

    img=tf.keras.preprocessing.image.load_img(image,target_size=(300, 300))
    x=tf.keras.preprocessing.image.img_to_array(img)
    x=np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images)
    if classes[0] <= 0.5 :
        score[0] += 1    
    else :
        score[1] += 1
    i += 1

print(score) 
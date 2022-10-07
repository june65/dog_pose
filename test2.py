import glob
import tensorflow as tf
import pandas as pd
from keras.applications.inception_resnet_v2 import preprocess_input

model = tf.keras.models.load_model('./dog_pose_v3')
#filepaths =  'data/val_data_set/not_pose/n02085620_575.jpg'


dog_images= glob.glob('.\\data\\val_data_set\\sigor_pose\\*')

for dog_image in dog_images:
    print(dog_image)
    img_path = dog_image.replace('\\','/').replace('./','')
    filepaths =  img_path
    IMAGE_SIZE  = (224, 224)
    test_image = tf.keras.utils.load_img(filepaths
                                ,target_size =IMAGE_SIZE )
    test_image = tf.keras.utils.img_to_array(test_image)
    test_image = test_image.reshape((1, test_image.shape[0], test_image.shape[1], test_image.shape[2]))
    test_image = preprocess_input(test_image)
    prediction = model.predict(test_image)

    df = pd.DataFrame({'pred':prediction[0]})
    df = df.sort_values(by='pred', ascending=False, na_position='first')
    print(f"## 예측률 : {(df.iloc[0]['pred'])* 100:.2f}%")
    print([df[df == df.iloc[0]].index[0]])
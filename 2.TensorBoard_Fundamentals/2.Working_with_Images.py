import cv2
import numpy as np
import tensorflow  as tf
sess = tf.Session()

imagename_list = ["Images/flower1.jpg", "Images/flower2.jpg", "Images/flower3.jpg"]
image_list = []


for i in range(len(imagename_list)):
    image = cv2.imread(imagename_list[i])
    resized_image = cv2.resize(image, (125, 125))
    image_list.append(resized_image)

image_list = np.array(image_list)
print(image_list.shape)

writer = tf.summary.FileWriter('./logs', graph=sess.graph)

index = 0
for image_tensor in image_list:
    summary_str = sess.run(tf.summary.image("image-" + str(index), image_tensor))
    writer.add_summary(summary_str)
    index += 1


writer.close()
sess.close()
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cnnmodel
import cv2
from subprocess import call
import os
import logging
logging.basicConfig(filename = 'my_logs.log', level = logging.DEBUG)
logging.info("Lab recording.")
from smmodel import PSPNet50
import imageio

#check if on windows OS
windows = False
if os.name == 'nt':
    windows = True

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "save/model.ckpt")

img = cv2.imread('steering_wheel_image.jpg',0)
rows,cols = img.shape

smoothed_angle = 0

smmodel = PSPNet50() # or another smmodel
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

smmodel.load("./smmodel/pspnet50.npy", sess)  # load pretrained smmodel

i = 0
while(cv2.waitKey(10) != ord('q')):
    
    smmodel.read_input("driving_dataset/" + str(i) + ".jpg")  # read image data from path
    preds = smmodel.forward(sess) # Get prediction 

    full_image = cv2.imread("driving_dataset/" + str(i) + ".jpg")
    cv2.imshow("Origin", full_image)
    full_image = preds[0]
    image = cv2.resize(full_image[-150:], (200, 66)) / 255.0
    degrees = cnnmodel.y.eval(feed_dict={cnnmodel.x: [image], cnnmodel.keep_prob: 1.0})[0][0] * 180.0 / 3.14159265
    if not windows:
        call("clear")
    print("Predicted steering angle: " + str(degrees) + " degrees")
    deg_str=format(degrees, '.6f')
    logging.info(str(i)+".jpg"+deg_str)
    cv2.imshow("Seg", full_image)

    #make smooth angle transitions by turning the steering wheel based on the difference of the current angle
    #and the predicted angle
    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imshow("steering wheel", dst)
    i += 1

cv2.destroyAllWindows()

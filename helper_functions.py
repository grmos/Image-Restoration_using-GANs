import os
import tensorflow as tf
import numpy as np
from numpy.linalg import norm
from skimage import transform
import imageio
import matplotlib.pyplot as plt
import math

fig = plt.figure(figsize=(10, 10))
ax1 = plt.subplot2grid(shape=(4, 2), loc=(0, 0), colspan=1)
ax2 = plt.subplot2grid(shape=(4, 2), loc=(0, 1), colspan=1)
ax3 = plt.subplot2grid(shape=(4, 2), loc=(1, 0), colspan=1)
ax4 = plt.subplot2grid(shape=(4, 2), loc=(1, 1), colspan=1)
ax5 = plt.subplot2grid(shape=(4, 2), loc=(2, 0), colspan=2)

def nearest_power_of_two(n):
    return int(math.pow(2, round(math.log(n, 2))))

def Transformation_1(LR,HR):
    term =int(HR/LR)
    T=np.zeros((LR*LR,HR*HR))
    for j in range(LR):
        for i in range(LR):
            for k in range(j*term,j*term+term):
                for m in range(i*term,i*term+term):
                    T[j*LR+i][k*HR+m]=1/(term**2)
    T = T.astype('float32')
    T=tf.convert_to_tensor(T)
    return T

def Transformation_2(LR,HR):
    term =int(HR/LR)
    avg_pool_2d = tf.keras.layers.AveragePooling2D(pool_size=(term,term),strides=(term, term), padding='valid')
    return avg_pool_2d

def get_module_space_image(T,LR,HR,bool_T,latent_dim,progan):
  #Create a face image using GAN
  vector = tf.random.normal([1, latent_dim])
  images = progan(vector)['default'][0]
  ax1.set_title('realimage')
  ax1.imshow(images)
  ax1.axis("off")
  #Use a transformation to reduce the image size
  if(bool_T):
      image = tf.reshape(images, [1, HR, HR, 3])
      image=T(image)
      image=tf.reshape(image, [1,LR,LR,3])
  else:
      image=tf.reshape(images,(HR**2,3))
      image=tf.linalg.matmul(T,image)
      image=tf.reshape(image, [LR*LR,3])
  images = tf.reshape(image, [ LR, LR, 3])
  bicubic=tf.image.resize(images, [HR,HR])
  ax2.set_title('lowres')
  ax2.imshow(images)
  ax2.axis("off")
  ax3.set_title('bicubic')
  ax3.imshow(bicubic)
  ax3.axis("off")
  return image

def upload_image(T,LR,HR,bool_T,imagepath):
  #Upload an image from disc
  image = imageio.imread(os.path.abspath(imagepath))
  realimage=transform.resize(image, [HR, HR])
  realimage=realimage[:,:,:3]
  realimage=realimage.astype('float32')
  realimage = tf.convert_to_tensor(realimage,dtype=tf.float32)
  ax1.set_title('realimage')
  ax1.imshow(realimage)
  ax1.axis("off")
  #Use a transformation to reduce the image size
  if(bool_T):
      image = tf.reshape(realimage, [1, HR, HR, 3])
      image=T(image)
      image=tf.reshape(image, [1,LR,LR,3])
  else:
      image=tf.reshape(realimage,(HR**2,3))
      image=tf.linalg.matmul(T,image)
      image=tf.reshape(image, [LR*LR,3])
  images = tf.reshape(image, [ LR, LR, 3])
  bicubic=tf.image.resize(images, [HR,HR])
  ax2.set_title('lowres')
  ax2.imshow(images)
  ax2.axis("off")
  ax3.set_title('bicubic')
  ax3.imshow(bicubic)
  ax3.axis("off")
  return image

def upload_LRimage(T,HR,bool_T,imagepath):
  #Upload an image from disc
  image = imageio.imread(os.path.abspath(imagepath))
  print(image.shape)
  h,w=image.shape[0:2]
  LR=np.min(np.array([h,w]))
  LR=nearest_power_of_two(LR)
  realimage=transform.resize(image, [LR, LR])
  realimage=realimage[:,:,:3]
  print(realimage.shape)
  realimage=realimage.astype('float32')
  realimage = tf.convert_to_tensor(realimage,dtype=tf.float32)
  ax1.set_title('realimage')
  ax1.imshow(realimage)
  ax1.axis("off")
  #Use a transformation to reduce the image size
  if(bool_T):
      image=tf.reshape(realimage, [1,LR,LR,3])
  else:
      image=tf.reshape(realimage, [LR*LR,3])
  images = tf.reshape(realimage, [ LR, LR, 3])
  bicubic=tf.image.resize(images, [HR,HR])
  ax2.set_title('lowres')
  ax2.imshow(images)
  ax2.axis("off")
  ax3.set_title('bicubic')
  ax3.imshow(bicubic)
  ax3.axis("off")
  return image,LR


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer._decayed_lr(tf.float32) # I use ._decayed_lr method instead of .lr
    return lr


def find_closest_latent_vector(progan,initial_vector, num_optimization_steps,
                               steps_per_image,T,bool_T,M,target_image):

  images = []
  losses = []
  vector = tf.Variable(initial_vector)  
  optimizer_1 = tf.optimizers.Adam(learning_rate=0.05)
  optimizer_2 = tf.optimizers.Adam(learning_rate=0.01)
  optimizer_3 = tf.optimizers.Adam(learning_rate=0.001)
  optimizer_4 = tf.optimizers.Adam(learning_rate=0.0001)
  optimizer_5 = tf.optimizers.Adam(learning_rate=0.00005)

  for step in range(num_optimization_steps):
    if (step % 100)==0:
      print("_______________step:{}___________________".format(step))
    with tf.GradientTape() as tape:
      image = progan(vector.read_value())['default'][0]
      if(bool_T):
          image = tf.reshape(image, [1, 128, 128, 3])
      else:
          image=tf.reshape(image,(HR**2,3))
      if(step<200):
          if (step % steps_per_image) == 0:
              images.append(image.numpy())
      elif(step<1000):
          if (step % (50*steps_per_image)) == 0:
              images.append(image.numpy())
      else:
          if (step % (500*steps_per_image)) == 0:
              images.append(image.numpy())
      if(bool_T):
          target_image_difference=M* tf.experimental.numpy.log(tf.square(tf.norm(T(image)- target_image)))
      else:
          target_image_difference=M* tf.experimental.numpy.log10(tf.square(tf.norm(tf.linalg.matmul(T,image)- target_image)))

      regularizer=tf.square(tf.norm(vector) )
      loss = target_image_difference + regularizer
      losses.append(loss.numpy())
      print(loss.numpy())
    grads = tape.gradient(loss, [vector])
    if(step<500 ):
      optimizer_1.apply_gradients(zip(grads, [vector]))
    elif(step<1500):
      optimizer_2.apply_gradients(zip(grads, [vector]))
    elif(step<10000):
      optimizer_3.apply_gradients(zip(grads, [vector]))
    elif(step<40000):
      optimizer_4.apply_gradients(zip(grads, [vector]))
    else:
      optimizer_5.apply_gradients(zip(grads, [vector]))


    
  return images, losses,vector

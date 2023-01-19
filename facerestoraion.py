from helper_functions import *
import random
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-if", "--ImageFrom", type=int, default=1,
	help="1->get_module_space_image,0->upload_image")
ap.add_argument("-resi", "--reducesize", type=int, default=0,
	help="IF you want to reduce image size->0, If the image has resolution <128->1")
ap.add_argument("-LR", "--LowResolution", type=int, default=32,
	help="IFyou pass an image and  you want to reduce the size choose a power of two")
ap.add_argument("-ip", "--imagepath", type=str, default="images/8.jpg",
	help="The path of the image")
ap.add_argument("-mp", "--modelpath", type=str, default="output/facemodel",
	help="The path of the generator model")
ap.add_argument("-ns", "--numsteps", type=int, default=300,
	help="The number of optimization steps")
ap.add_argument("-Tr", "--Transform", type=int, default=1,
	help="1-> Conv2D, 0-> array T")
args = vars(ap.parse_args())


num = random.random()
tf.random.set_seed(int(num*100))

latent_dim = 512
bool_T=args["Transform"] 
image_from_module_space = args["ImageFrom"]  
HR=128
if(args["reducesize"]):
	image = imageio.imread(os.path.abspath(args["imagepath"]))
	h,w=image.shape[0:2]
	LR=np.min(np.array([h,w]))
	LR=nearest_power_of_two(LR)
else:
	LR=args["LowResolution"]
M=LR**2*3 #length of LR image vector
path=os.path.abspath(args["modelpath"])
progan =tf.saved_model.load(path).signatures['default']
if(bool_T):
    T=Transformation_2(LR,HR)
else:
    T=Transformation_1(LR,HR)

if image_from_module_space:
  target_image = get_module_space_image(T,LR,HR,bool_T,latent_dim,progan)
else:
	if(args["reducesize"]):
		target_image,LR = upload_LRimage(T,HR,bool_T,args["imagepath"])
	else:
		target_image = upload_image(T,LR,HR,bool_T,args["imagepath"])

##Create an initial vector to start the proccess
initial_vector = tf.random.normal([1, latent_dim])
##Generate a random Face from initial vector
img=progan(initial_vector)['default'][0]
##Number of steps for  optimization
num_optimization_steps=args["numsteps"]
steps_per_image=1
images, loss,vector = find_closest_latent_vector(progan,initial_vector, num_optimization_steps, steps_per_image,T,bool_T,M,target_image)

##Create Gif
imagesan=tf.reshape(images,[len(images),HR,HR,3])
imagesan = np.array(imagesan)
converted_images = np.clip(imagesan * 255, 0, 255).astype(np.uint8)
imageio.mimsave('./animation.gif', converted_images)

##Plot the Loss Curve 
ax5.set_title('loss')
ax5.plot(loss)

##Show the image is generated from GAN
image = tf.convert_to_tensor(images[-1],dtype=tf.float32)
image=tf.reshape(image,(HR,HR,3))
ax4.set_title('GAN')
ax4.imshow(image)
ax4.axis("off")

plt.show()
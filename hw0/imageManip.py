import math
import cv2

import numpy as np

def load(image_path):
    """Loads an image from a file path.

    HINT: Look up `cv2.imread()` function - 
          whatch out  for the returned color format ! Check the following link for some fun : 
          https://www.learnopencv.com/why-does-opencv-use-bgr-color-format/

    Args:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### VOTRE CODE ICI - DEBUT
    # Utilisez cv2.imread - le format RGB doit être retourné
    pass
    ### VOTRE CODE ICI - FIN

    # Let's convert the image to be between the correct range.
    out = out.astype(np.float32) / 255
    return out


def dim_image(image):
    """Change the value of every pixel by following

                        x_n = 0.5*x_p^2

    where x_n is the new value and x_p is the original value.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None

    ### VOTRE CODE ICI - DEBUT
    out = np.array(image)
    out = 0.5 * np.power(image, 2)
    ### VOTRE CODE ICI - FIN

    return out


def convert_to_grey_scale(image):
    """Change image to gray scale.

    HINT: see if you can use  the opencv function `cv2.cvtColor()` 
    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width).
    """
    out = None

    ### VOTRE CODE ICI - DEBUT    
    out = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY )

    ### VOTRE CODE ICI - FIN

    return out


def rgb_exclusion(image, channel):
    """Return image **excluding** the rgb channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "R", "G" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None

    ### VOTRE CODE ICI - DEBUT
    out = np.array(image)
    if channel == 'R' :
        out[:, :, 0] = 0
                

    elif channel == 'G' :
        out[:, :, 1] = 0
    
    elif channel == 'B' :
        out[:, :, 2] = 0
    ### VOTRE CODE ICI - FIN

    return out


def lab_decomposition(image, channel):
    """Decomposes the image into LAB and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "L", "A" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """

    out = None

    ### VOTRE CODE ICI - DEBUT
    img = cv2.cvtColor(image, cv2.COLOR_RGB2LAB )
    L,A,B = cv2.split(img)
    
    
    if channel == 'L':
        
        out = L
    if channel == "A":
        
        out = A
        
    if channel == 'B':
        
        out = B
    ### VOTRE CODE ICI - FIN

    return out


def hsv_decomposition(image, channel='H'):
    """Decomposes the image into HSV and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "H", "S" or "V".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """
    out = None

    ### VOTRE CODE ICI - DEBUT
    img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV )
    H,S,V = cv2.split(img)
    
    if channel == 'H':
        
        out = H
    if channel == "S":
        
        out = S
        
    if channel == 'V':
        
        out = V
    ### VOTRE CODE ICI - FIN

    return out


def mix_images(image1, image2, channel1, channel2):
    """Combines image1 and image2 by taking the left half of image1
    and the right half of image2. The final combination also excludes
    channel1 from image1 and channel2 from image2 for each image.

    HINTS: Use `rgb_exclusion()` you implemented earlier as a helper
    function. Also look up `np.concatenate()` to help you combine images.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).
        image2: numpy array of shape(image_height, image_width, 3).
        channel1: str specifying channel used for image1.
        channel2: str specifying channel used for image2.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None
    ### VOTRE CODE ICI - DEBUT
    img1 = np.array(image1)
    img2 = np.array(image2)
    
    x = img1.shape[1]
    y = img1.shape[0]
    
    
    
    out1 = rgb_exclusion(img1[:, :int(x/2)], channel1)
    out2 = rgb_exclusion(img2[:, int(x/2):], channel2)
    
    
    
    out = np.concatenate((out1, out2), axis=1)
    ### VOTRE CODE ICI - FIN

    return out


def mix_quadrants(image):
    """
    This function takes an image, and performs a different operation
    to each of the 4 quadrants of the image. Then it combines the 4
    quadrants back together.

    Here are the 4 operations you should perform on the 4 quadrants:
        Top left quadrant: Remove the 'R' channel using `rgb_exclusion()`.
        Top right quadrant: Dim the quadrant using `dim_image()`.
        Bottom left quadrant: Brighthen the quadrant using the function:
            x_n = x_p^0.5
        Bottom right quadrant: Remove the 'R' channel using `rgb_exclusion()`.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### VOTRE CODE ICI - DEBUT
    
    out = np.array(image)
    x = out.shape[1]
    y = out.shape[0]
    out[:int(y/2), :int(x/2)] = rgb_exclusion(out[:int(y/2), :int(x/2)], 'R')
    out[:int(y/2), int(x/2):] = dim_image(out[:int(y/2), int(x/2):])
    out[int(y/2):, :int(x/2)] = np.power(out[int(y/2):, :int(x/2)], 0.5)
    out[int(y/2):, int(x/2):] = rgb_exclusion(out[int(y/2):, int(x/2):], 'R')

    

    ### VOTRE CODE ICI - FIN

    return out

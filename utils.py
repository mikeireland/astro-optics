"""Useful methods to assist with various functions.

Author: Adam Rains 
"""
import pylab as pl
import cv2
import glob

def save_plot(data, title, directory, imagename, format='.jpg'):
    """
    
    Parameters
    ----------
    
    data: 2D numpy.array
        The data to be plotted and saved. Will not accept complex numbers.
    title: string
        The title of the plot.
    directory: string
        The directory to save the plot to, ending with a "/"
    imagename: string
        The name of the plot to be saved.
    format: string  
        The file formate of the saved image.
    """
    
    pl.clf()
    pl.imshow(data) 
    pl.title(title)
    pl.savefig( (directory + (imagename) + format ) )

def create_movie(directory, image_format='jpg', fps=5, video_name='video.avi'):
    """Create a movie using a sequence of images

    Parameters
    ----------
    directory: string
        The directory to load the images from and save the video to.
    image_format: string
        The file format of the image files.
    fps: integer
        Frames Per Second of the created video
    video_name: string  
        The filename of the resulting video (including format)
    """
    
    # Get a list of the file paths of all images matching the file format and sort them
    image_paths = glob.glob( (directory + "*." +image_format) )
    image_paths.sort()
    
    # Create the video writer
    # cv2.VideoWriter(filename, fourcc, fps, frame_size, is_color)
    height, width, layers = cv2.imread(image_paths[0]).shape
    video = cv2.VideoWriter( (directory + video_name), -1, fps, (width, height), 1)
    
    # Read each image and write to the video
    for i in image_paths:
        image = cv2.imread(i)
        
        video.write(image)
    
    # Clean up
    cv2.destroyAllWindows()
    video.release()
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision
from scipy.optimize import linear_sum_assignment
from skimage import io, color
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from IPython.display import Video
from IPython.display import HTML


def get_frames(start_frame, end_frame, show = False):
    '''
    Retrieves a list of frame filenames from the movie "10 Things I Hate About You".
    
    This function generates paths to frame image files based on the specified range.
    Optionally displays the first and last frames of the selected range using matplotlib.
    
    Args:
        start_frame (int): Integer from [0,999] specifying the start frame.
        end_frame (int): Integer from [1,1000] specifying frame to stop before.
        show (bool): Whether or not to show the start and end frame. Default is False.
        
    Returns:
        list: List of frame file paths from "10 Things I Hate About You" movie.
    '''
    
    output = []
    for frame in range(start_frame, end_frame): 
        output.append(f"kaggle/input/movie-identification-dataset-800-movies/resized_frames/10 Things I Hate About You (1999)/frame_{'0'*(4-len(str(frame))) + str(frame)}.jpg")
        
    if show:
        plt.figure(figsize=(12, 6))
    
        # Display the original image
        plt.subplot(1, 2, 1)
        plt.imshow(io.imread(output[0]))
        plt.title('First Frame')
        plt.axis('off')
        
        # Display the reference map
        plt.subplot(1, 2, 2)
        plt.imshow(io.imread(output[-1]))
        plt.title('Last Frame')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
    return output
    
# Function to generate video
# Modified from: https://www.geeksforgeeks.org/python-create-video-using-multiple-images-using-opencv/
def generate_video(images, name, path = "output_videos/", create_playable = True):
    '''
    Creates a video from a sequence of images.
    
    This function converts a list of RGB images to a video file in AVI format.
    Optionally creates an additional MP4 version of the video for better playback compatibility.
    
    Args:
        images (list): List of numpy arrays representing images to be converted to video frames.
        name (str): Base name for the output video file (without extension).
        path (str): Directory path where the video will be saved. Default is "output_videos/".
        create_playable (bool): Whether to create an additional MP4 version of the video. Default is True.
        
    Returns:
        str: Path to the generated video file.
    '''
    
    # images = [cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2BGR) for img in images] # Convert to BGR
    images = [np.uint8(cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2BGR)) for img in images]
    # plt.imshow(images[0])
    video_name = path + name + '.avi'
    # Set frame from the first image
    frame = images[0]
    height, width, layers = frame.shape
    # Video writer to create .avi file
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 1, (width, height))
    # Appending images to video
    for image in images:
        video.write(image)
    
    # Release the video file
    video.release()
    cv2.destroyAllWindows()
    print("Video generated successfully!")
    if create_playable:
        # creates an mp4 version of video
        mp4_video_name = video_name[:-4]
        convert_avi_to_mp4(video_name, mp4_video_name)
        video_name = mp4_video_name + ".mp4"
        
    return video_name

def convert_avi_to_mp4(avi_file_path, output_name):
    '''
    Converts an AVI video file to MP4 format using ffmpeg.
    
    This function uses the ffmpeg command-line tool to convert AVI videos to the more
    widely compatible MP4 format with specific encoding parameters for quality.
    
    Args:
        avi_file_path (str): Path to the input AVI file.
        output_name (str): Name for the output MP4 file (without extension).
        
    Returns:
        bool: True if the conversion process was initiated successfully.
    '''
    os.popen("ffmpeg -i '{input}' -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 '{output}.mp4'".format(input = avi_file_path, output = output_name))
    return True

def show_color_histogram(data, img):
    """
    Display color histograms for RGB and HSV color spaces of an image.
    
    This function creates a visualization with multiple subplots showing the original image
    alongside RGB and HSV color histograms for comparison and analysis. It uses matplotlib
    to create a figure with separate plots for RGB channels and HSV channels.
    
    Args:
        data (tuple): A tuple containing two elements:
            - First element: Tuple of (r_hist, g_hist, b_hist) containing histogram data for RGB channels
            - Second element: Tuple of (h_hist, s_hist, v_hist) containing histogram data for HSV channels
    
    Returns:
        None: This function displays the plots but does not return any values.
    
    Notes:
        - RGB histograms are plotted with corresponding channel colors (red, green, blue)
        - HSV histograms use purple for hue, orange for saturation, and gray for value
        - All histograms are displayed with slight transparency (alpha=0.7) for better overlap visibility
    """
    r_hist, g_hist, b_hist = data[0]
    h_hist, s_hist, v_hist = data[1]

    # Create a figure with subplots
    plt.figure(figsize=(15, 10))
    
    # Plot the original image
    plt.subplot(3, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
    
    # Plot RGB histogram
    plt.subplot(3, 2, 3)
    plt.plot(r_hist, color='red', alpha=0.7)
    plt.plot(g_hist, color='green', alpha=0.7)
    plt.plot(b_hist, color='blue', alpha=0.7)
    plt.title('RGB Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.xlim([0, 256])
    plt.legend(['Red', 'Green', 'Blue'])

    # Plot HSV histogram combined
    plt.subplot(3, 2, 4)
    plt.plot(h_hist, color='purple', alpha=0.7)
    plt.plot(s_hist, color='orange', alpha=0.7)
    plt.plot(v_hist, color='gray', alpha=0.7)
    plt.title('HSV Channels Comparison')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.xlim([0, 256])
    plt.legend(['Hue', 'Saturation', 'Value'])
    
    plt.tight_layout()
    plt.show()

    
def color_histogram(img, show = False):
    """
    Calculate and optionally display color histograms for RGB and HSV color spaces of an image.
    
    This function computes histograms for all channels in both RGB and HSV color spaces 
    of the input image. It can optionally display these histograms using the 
    show_color_histogram() function.
    
    Args:
        img (numpy.ndarray): Input image in RGB color space.
        show (bool, optional): Whether to display the histograms using show_color_histogram().
            Defaults to False.
    
    Returns:
        tuple: A tuple containing two elements:
            - First element: Tuple of (r_hist, g_hist, b_hist) containing histogram data for RGB channels
            - Second element: Tuple of (h_hist, s_hist, v_hist) containing histogram data for HSV channels
    
    Notes:
        - RGB histograms use 256 bins covering the range [0, 256]
        - Hue histogram uses 180 bins covering the range [0, 180]
    """
    
    img_rgb = img
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Split the image into its three color channels
    r, g, b = cv2.split(img_rgb)
    h, s, v = cv2.split(img_hsv)

    # Set up the histogram parameters
    rgb_bins = 256  # Number of bins
    h_bins = 180 
    s_v_bins = 256
    
    rgb_range = (0, 256)  # Range of values
    h_range = (0, 180)
    s_v_range = (0, 256)

    # Calculate histograms for each channel
    r_hist = cv2.calcHist([r], [0], None, [rgb_bins], rgb_range)
    g_hist = cv2.calcHist([g], [0], None, [rgb_bins], rgb_range)
    b_hist = cv2.calcHist([b], [0], None, [rgb_bins], rgb_range)

    h_hist = cv2.calcHist([h], [0], None, [h_bins], h_range)
    s_hist = cv2.calcHist([s], [0], None, [s_v_bins], s_v_range)
    v_hist = cv2.calcHist([v], [0], None, [s_v_bins], s_v_range)

    data = [(r_hist, g_hist, b_hist), (h_hist, s_hist, v_hist)]
    
    if show:
        show_color_histogram(data, img)
    
    return data 

def compare_histograms(hist1, hist2):
    """
    Compare two histograms using the Bhattacharyya distance after normalization.
    
    This function normalizes two input histograms to the range [0,1] and then
    calculates the Bhattacharyya distance between them using OpenCV's compareHist 
    function. The Bhattacharyya distance measures the similarity of two probability
    distributions, with values closer to 0 indicating more similar distributions.
    
    Args:
        hist1 (numpy.ndarray): First histogram to compare.
        hist2 (numpy.ndarray): Second histogram to compare.
    
    Returns:
        float: Bhattacharyya distance between the normalized histograms, where:
            - 0 indicates perfectly matching histograms
            - 1 indicates completely non-overlapping histograms
    
    Notes:
        - Lower return values indicate more similar histograms
        - Uses a more numerically stable approach with separate output arrays
        - Clips negative values to zero to ensure valid Bhattacharyya calculation
    """
    # Check if either histogram is empty (all zeros)
    if np.sum(hist1) == 0 or np.sum(hist2) == 0:
        return 1.0  # Maximum distance for empty histograms
    
    # Use None as output parameter to avoid in-place modification
    hist1_normalized = cv2.normalize(hist1, None, 0, 1, cv2.NORM_MINMAX)
    hist2_normalized = cv2.normalize(hist2, None, 0, 1, cv2.NORM_MINMAX)
    
    # Clip any tiny negative values to zero (from floating-point precision issues)
    hist1_normalized = np.maximum(hist1_normalized, 0)
    hist2_normalized = np.maximum(hist2_normalized, 0)
    
    # Calculate Bhattacharyya distance between normalized histograms
    bhattacharyya = cv2.compareHist(hist1_normalized, hist2_normalized, cv2.HISTCMP_BHATTACHARYYA)
    
    # Handle any NaN values from calculation
    if np.isnan(bhattacharyya):
        return 1.0  # Return maximum distance if calculation fails
        
    return bhattacharyya
    

def compare_histograms_average(hist_set1, hist_set2):
    """
    Calculate the average Bhattacharyya distance across RGB color channels.
    
    This function compares two sets of RGB histograms by computing the Bhattacharyya
    distance for each individual color channel (R, G, B) and then averaging these
    distances to provide a single similarity metric between the two histogram sets.
    
    Args:
        hist_set1 (tuple): Tuple containing (histR1, histG1, histB1) - histograms for
            the first image's red, green, and blue channels respectively.
        hist_set2 (tuple): Tuple containing (histR2, histG2, histB2) - histograms for
            the second image's red, green, and blue channels respectively.
    
    Returns:
        float: Average Bhattacharyya distance across the three color channels, where:
            - 0 indicates perfectly matching histogram sets
            - 1 indicates completely non-overlapping histogram sets
    
    Notes:
        - Each histogram is compared separately using the compare_histograms() function
        - Lower return values indicate more similar overall color distributions
    """ 
    histR1, histG1, histB1 = hist_set1
    histR2, histG2, histB2 = hist_set2
    
    bhattacharyyaR = compare_histograms(histR1, histR2)
    bhattacharyyaG = compare_histograms(histG1, histG2)
    bhattacharyyaB = compare_histograms(histB1, histB2)
    
    # Calculate the mean of the three distances
    average_bhattacharyya = (bhattacharyyaR + bhattacharyyaG + bhattacharyyaB) / 3.0
    
    return average_bhattacharyya

def test_scenes_metric(new_scene, source_img):
    """
    Calculate the average color similarity between a source image and a sequence of scene frames.
    
    This function measures how well a color transfer has been applied to multiple frames
    by computing the average Bhattacharyya distance between the RGB histograms of the 
    source image and each frame in the new scene. Lower values indicate better color 
    matching across the entire scene.
    
    Args:
        new_scene (list): List of RGB images (numpy.ndarray) where color transfer has been applied.
        source_img (numpy.ndarray): Source RGB image used as the color reference.
    
    Returns:
        float: Average Bhattacharyya distance across all frames in the scene, where:
            - Values closer to 0 indicate better color matching with the source image
            - Values closer to 1 indicate poor color matching with the source image

    Interpretation of return values:
        - < 0.3: Good color transfer
        - 0.3-0.5: Acceptable color transfer
        - > 0.5: Poor color transfer
    """

    # Calculate color histogram for the source image (reference)
    source_hist = color_histogram(source_img)
    source_hist_rgb = source_hist[0]  # Extract only RGB histograms, ignoring HSV

    # Initialize accumulator for total Bhattacharyya distance across all frames
    total_scene_bhattacharyya = 0

    # Loop through each frame in the new scene
    for i,frame in enumerate(new_scene): 
        # Calculate color histogram for the current frame
        frame_hist = color_histogram(new_scene[i])
        frame_hist_rgb = frame_hist[0]  # Extract only RGB histograms
        # Calculate similarity between current frame and source image histograms
        # Add to running total (lower values indicate better similarity)
        total_scene_bhattacharyya += compare_histograms_average(frame_hist_rgb, source_hist_rgb)

    # Calculate the average similarity across all frames in the scene
    average_scene_bhattacharyya = total_scene_bhattacharyya / len(new_scene)

    # Print the score and provide a qualitative assessment based on thresholds
    print("Scene Average Bhattacharyya Score: ", str(average_scene_bhattacharyya))
    if average_scene_bhattacharyya > 0.5:
        print("Poor Transfer")  # High distance indicates poor color matching
    elif average_scene_bhattacharyya < 0.3:
        print("Good Transfer")  # Low distance indicates good color matching
    else: 
        print("Acceptable Transfer")  # Middle range indicates acceptable results
        
    return average_scene_bhattacharyya
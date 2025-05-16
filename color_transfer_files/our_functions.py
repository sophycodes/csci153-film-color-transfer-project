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

# Define paths
base_dir = "/mnt/cs/cs153/projects/sophy_theo"
# Join path components together
depth_anything_dir = os.path.join(base_dir, "Depth-Anything-V2")

# Import from depth-anything
# Add the directory path stored in the variable depth_anything_dir to Python's module search path
sys.path.append(depth_anything_dir)
from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet

# Set up device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# Model configs for Depth Anything
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]} # giant(not yet out)
}

# Load the Depth Anything model
# small 24.8M params, base 97.5M params, large 335.3M params
# encoders = ['vits', 'vitb', 'vitl', 'vitg']
depth_encoder = 'vits'  # or 'vitl', 'vitb', 'vitg'
depth_model = DepthAnythingV2(**model_configs[depth_encoder])
depth_checkpoint = os.path.join(depth_anything_dir, f'checkpoints/depth_anything_v2_{depth_encoder}.pth')
depth_model.load_state_dict(torch.load(depth_checkpoint, map_location='cpu'))
depth_model = depth_model.to(device).eval()
print("Depth model loaded successfully")

# Define segment anything path 
segment_anything_dir = os.path.join(base_dir, "segment-anything")
# Add the parent directory to sys.path, not the segment_anything/segment_anything subdirectory
sys.path.append(segment_anything_dir)

from skimage import io, color

# Now import from segment_anything
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

print("our_functions version: " + "2")

def get_poster(poster_vals, km):
    """
    Reconstructs a posterized image by mapping each pixel's cluster index to its corresponding color center.
    
    Args:
        poster_vals (numpy.ndarray): 2D array with shape (height, width) containing cluster indices (0 to n_clusters-1) 
            for each pixel in the image.
        km (sklearn.cluster.KMeans): Fitted K-means model containing the cluster_centers_ attribute 
            that stores the representative colors for each cluster.
            
    Returns:
        numpy.ndarray: 3D array with shape (height, width, n_features) where each pixel's cluster index 
            has been replaced with the actual color values from the corresponding cluster center. 
            For CIELAB color transfers the shape is (height, width, 2) containing only a* and b* channels.
    """
    f = lambda x: km.cluster_centers_[x] # get color center
    poster = f(poster_vals)
    return poster

def active():
    return True

def get_num_clusters(data):
    '''
    Determines the optimal number of clusters for K-means by evaluating silhouette scores.
    
    This function runs K-means clustering multiple times with cluster counts ranging from 2 to 10,
    and evaluates each clustering using the silhouette score metric. The number of clusters
    that produces the highest silhouette score is returned as the optimal choice.
   
       Args:
           data (numpy.ndarray): A 2D array where each row represents a pixel's color values and each column represents feature/dimension.
               Expected shape is (n_samples, n_features).
               
       Returns:
           int: The optimal number of clusters (between 2 and 10) based on silhouette score evaluation.
               A higher silhouette score indicates better-defined, more separated clusters.
    '''
    sil_score_max = -1 #this is the minimum possible score
    best_n_clusters = 0
    print("Finding Best Cluster: \n")
    for n_clust in range(2,11):
        kmeans = KMeans(n_clusters=n_clust).fit(data)
        sil_score = silhouette_score(data, kmeans.labels_)
        print("The average silhouette score for %i clusters is %0.2f" %(n_clust,sil_score))
        if sil_score >= sil_score_max:
            sil_score_max = sil_score
            best_n_clusters = n_clust
    print(f"Using {best_n_clusters} clusters")
    return best_n_clusters

def posterize_v1(img, t = 5, nclust = None, resdrop = 200):
    """
    This function identifies the most representative colors in a CIELAB image by:
       1. Applying a brightness threshold to exclude very dark pixels
       2. Extracting and clustering only the a* and b* color channels
       3. Automatically determining the optimal number of clusters if not specified
       
       For efficiency, when automatically determining the optimal cluster count, the function
       operates on a downsampled version of the image before applying the final clustering
       to the full-resolution data.
       
       Args:
           img (numpy.ndarray): Input image in CIELAB color space with shape (height, width, 3)
           t (int, optional): Brightness threshold value. Pixels with L* value <= t will be excluded. Defaults to 5.
           nclust (int, optional): Number of color clusters to extract. If None, automatically 
                                  determined using silhouette scores. Defaults to None.
           resdrop (int, optional): Horizontal resolution for the downsampled image used when 
                                   automatically determining optimal cluster count. Defaults to 200.
       
       Returns:
           sklearn.cluster.KMeans: Fitted K-means model containing:
               - cluster_centers_: Representative colors (a*, b*) extracted from the image
               - labels_: Cluster assignments for each pixel in the flattened image
        
    """
    # Create a mask based on the L* channel threshold
    # This mask identifies which pixels are bright enough to keep
    cut = img[:,:,0] > t
    darkcut = np.stack([cut, cut, cut], axis = 2)
    # Apply mask to the full image and extract the a* and b* color channels
    # Sets all color values to zero for dark pixels
    refined = (darkcut * img)[:,:,1:]

    # Flatten the a* and b* channels into a 2D array for clustering
    color_data = refined.reshape((refined.shape[0]*refined.shape[1],2))
    
    if nclust == None:
        print("lowres img")
        # Resize the image to a lower resolution to make the optimal cluster determination process faster.
        lowres = cv2.resize(img, (resdrop, int(resdrop*img.shape[0]/img.shape[1])))
        plt.imshow(color.lab2rgb(lowres))
        plt.show()
        lowcut = lowres[:,:,0] > t
        lowdarkcut = np.stack([lowcut, lowcut, lowcut], axis = 2)
        lowrefined = (lowdarkcut * lowres)[:,:,1:]
        # Apply the same brightness thresholding and channel extraction 
        low_color_data = lowrefined.reshape((lowrefined.shape[0]*lowrefined.shape[1],2))
        # Determine the optimal number of clusters
        nclust = get_num_clusters(low_color_data)

    return KMeans(n_clusters=nclust, random_state = 0).fit(color_data), nclust

def add_in_lightness(img, poster):
    """
     Reintegrates the original lightness channel with posterized color channels.
     
     This function combines the L* channel from the original CIELAB image with 
     the posterized a* and b* color channels to create a complete CIELAB image.
     This approach preserves the original lighting details while using the simplified color palette.
     
     Args:
       img (numpy.ndarray): Original CIELAB image with shape (height, width, 3),
           from which the L* channel will be extracted.
       poster (numpy.ndarray): Posterized color channels (a* and b*) with shape
           (height, width, 2), typically obtained from a clustering algorithm.


     Returns:
       numpy.ndarray: Complete CIELAB image with shape (height, width, 3) combining
           the original lightness with the posterized colors. This image can be 
           converted to RGB for display using color.lab2rgb().
    """
    return np.stack([img[:,:,0],poster[:,:,0],poster[:,:,1]], axis = 2)
    
def color_by_number(img, kmeans):
    """
    Converts an image into a segmentation map based on K-means color clustering.
    
    This function takes an image and a fitted K-means model, and returns a 2D array
    where each pixel value represents the index of the color cluster to which that
    pixel belongs.
    
    Args:
        img (numpy.ndarray): The input CIELAB image array. The function only uses the shape
            of this array and does not process its content directly.
        kmeans (sklearn.cluster.KMeans): A fitted K-means model containing the cluster
            assignments for each pixel in the image. The 'labels_' attribute of this
            model should contain the cluster assignments in a flattened format.
    
    Returns:
        numpy.ndarray: A 2D array with the same spatial dimensions as the input image
            (height, width), where each element is an integer representing the cluster
            index (0 to n_clusters-1) of the corresponding pixel.
    
    Note:
        This function assumes that the K-means model was fitted on a flattened version
        of the image, where each pixel is represented as a feature vector.
    """
    return kmeans.labels_.copy().reshape(img.shape[0:2])

def visualize_poster(img, kmeans):
    """
   Creates and displays a posterized version of a CIELAB image using K-means clustering.
   
   This function handles the complete posterization visualization process:
   1. Gets cluster assignments for each pixel using color_by_number()
   2. Converts cluster indices to actual color values using get_poster()
   3. Reintegrates the original lightness channel using add_in_lightness()
   4. Converts the resulting CIELAB image to RGB and displays it
   
   Args:
       img (numpy.ndarray): CIELAB image with shape (height, width, 3) to be posterized.
       kmeans (sklearn.cluster.KMeans): Fitted K-means model containing cluster centers
           and labels for color posterization. This should have been trained on the
           a* and b* channels of the image.
   
   Returns:
       None: The function directly displays the posterized image using matplotlib.

    """
    poster_vals = color_by_number(img, kmeans)
    poster = get_poster(poster_vals, kmeans)
    final_img = add_in_lightness(img, poster)
    plt.imshow(color.lab2rgb(final_img))

def visualize_posterS(img, kmeans, ax=None):
    """
    Creates and displays a posterized version of a CIELAB image using K-means clustering.
    
    This function handles the complete posterization visualization process:
    1. Gets cluster assignments for each pixel using color_by_number()
    2. Converts cluster indices to actual color values using get_poster()
    3. Reintegrates the original lightness channel using add_in_lightness()
    4. Converts the resulting CIELAB image to RGB and displays it
    
    Args:
        img (numpy.ndarray): CIELAB image with shape (height, width, 3) to be posterized.
        kmeans (sklearn.cluster.KMeans): Fitted K-means model containing cluster centers
            and labels for color posterization. This should have been trained on the
            a* and b* channels of the image.
        ax (matplotlib.axes.Axes, optional): Matplotlib axes object to plot on.
            If None, a new figure and axes will be created.
    
    Returns:
        matplotlib.figure.Figure: The figure containing the posterized image.
    """
    poster_vals = color_by_number(img, kmeans)
    poster = get_poster(poster_vals, kmeans)
    final_img = add_in_lightness(img, poster)
    
    # Create new axes if none provided
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    
    # Display the image on the provided axes
    ax.imshow(color.lab2rgb(final_img))
    ax.axis('off')  # Hide axes for better visualization
    
    return fig

def dist(p1, p2):
    """
    Return Euclidean distance between two points.
    
    Args:
        p1 (numpy.ndarray): First point coordinates as a numpy array.
        p2 (numpy.ndarray): Second point coordinates as a numpy array.
    
    Returns:
        float: The Euclidean distance between the two points.
    """
    return np.sum((p1 - p2) ** 2) ** 0.5

def all_dist(set1, set2):
    """
    Takes in two sets of points and returns an adjacency matrix of distances.
    
    This function calculates the Euclidean distance between all pairs of points
    from the two input sets, creating a complete bipartite distance matrix.
    
    Args:
        set1 (numpy.ndarray): First set of points, with shape (n, d) where n is 
            the number of points and d is the dimensionality of each point.
        set2 (numpy.ndarray): Second set of points, with shape (m, d) where m is 
            the number of points and d is the dimensionality of each point.
    
    Returns:
        numpy.ndarray: A matrix of shape (n, m) where element [i,j] contains the 
            Euclidean distance between the i-th point in set1 and the j-th point 
            in set2.
    """
    all_dists = np.zeros((set1.shape[0], set2.shape[0]))
    
    for i in range(set1.shape[0]):
        for j in range(set2.shape[0]):
            all_dists[i,j] = dist(set1[i], set2[j])
            
    return all_dists

def construct_img(img, color_by_num, matched_colors):
    """
    Constructs a full image by applying matched colors to a color-indexed image.
    
    This function takes a base image, a color index array where each pixel is assigned 
    a cluster index, and a set of matched colors to apply to each cluster. It then 
    constructs the final image by applying the appropriate color to each pixel and 
    reintegrating the original lightness channel.
    
    Args:
        img (numpy.ndarray): Original CIELAB image with shape (height, width, 3).
        color_by_num (numpy.ndarray): Array of same shape as img (height, width) where
            each element is an integer cluster index.
        matched_colors (numpy.ndarray): Array of color values to assign to each cluster,
            where matched_colors[i] contains the color for cluster index i.
    
    Returns:
        numpy.ndarray: Reconstructed CIELAB image with the new color assignments and
            original lightness values.
    """
    f = lambda x: matched_colors[x] # get color center
    poster = f(color_by_num)
    return add_in_lightness(img, poster)

def simple_match(centers1, centers2):
    """
    Matches colors from one palette to another using minimum Euclidean distance.
    
    For each color in centers1, finds the closest matching color in centers2
    based on Euclidean distance in the color space.
    
    Args:
        centers1 (numpy.ndarray): First set of palette colors, with shape (n, d)
            where n is the number of colors and d is the color space dimensionality.
        centers2 (numpy.ndarray): Second set of palette colors, with shape (m, d).
    
    Returns:
        numpy.ndarray: Array of indices into centers2, where the i-th element is 
            the index of the color in centers2 that best matches the i-th color 
            in centers1.
    """
    all_dists = all_dist(centers1, centers2)
    return np.argmin(all_dists, axis = 1)

def shift_points(points, normalize= False):
    """
    Centers a set of points at the origin and optionally normalizes them.
    
    Calculates the centroid of the points and shifts all points so the centroid
    is at the origin. If normalize is True, also scales the points so their average
    distance from the origin is 1.
    
    Args:
        points (numpy.ndarray): Set of points with shape (n, d) where n is the
            number of points and d is the dimensionality.
        normalize (bool, optional): Whether to scale the points so their average
            distance from the origin is 1. Defaults to False.
    
    Returns:
        numpy.ndarray: The shifted (and optionally normalized) points.
    """
    # get average points
    points_center = np.apply_along_axis(np.mean, 0, points)
    # points shifted to both be at zero
    points_shift = np.apply_along_axis(lambda x: x - points_center, 1, points)
    if normalize:
        # get_magnitudes (used for normalization)
        points_magnitudes = np.apply_along_axis(lambda x: np.sqrt(np.sum(x)), 1, points_shift**2)
        points_shift /= np.mean(points_magnitudes)
        
    return points_shift

def show_before(target_path, reference_path):
    """
    Displays both target and reference images side by side.
    
    Creates a figure with two subplots, one showing the target image and the
    other showing the reference image.
    
    Args:
        target_path (str): File path to the target image.
        reference_path (str): File path to the reference image.
    
    Returns:
        None: The function directly displays the images using matplotlib.
    """
    plt.figure(figsize=(12, 6))

    # Display the original image
    plt.subplot(1, 2, 1)
    plt.imshow(io.imread(target_path))
    plt.title('Target Image')
    plt.axis('off')
    
    # Display the reference map
    plt.subplot(1, 2, 2)
    plt.imshow(io.imread(reference_path))
    plt.title('Source Image')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def relative_shift_pair(set1, set2, normalize = False, show = False):
    """
    Shifts two sets of points to their respective origins and optionally normalizes them.
    
    Applies the shift_points function to both sets independently, centering each at
    the origin. If normalize is True, each set is scaled so its average distance from
    the origin is 1. Can optionally visualize the shifted points.
    
    Args:
        set1 (numpy.ndarray): First set of 2D points with shape (n, 2).
        set2 (numpy.ndarray): Second set of 2D points with shape (m, 2).
        normalize (bool, optional): Whether to normalize point distances. Defaults to False.
        show (bool, optional): Whether to display a visualization of the shifted points. 
            Defaults to False.
    
    Returns:
        tuple: A pair of numpy.ndarray objects containing the shifted (and optionally
            normalized) point sets.
    """
    set1_shift = shift_points(set1, normalize = normalize)
    set2_shift = shift_points(set2, normalize = normalize)
    
    # see points post shift
    if show:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(set1_shift[:,0], set1_shift[:,1])
        ax.scatter(set2_shift[:,0], set2_shift[:,1])
        circ = plt.Circle((0, 0), radius=1, edgecolor='b', facecolor='None')
        if normalize:
            ax.add_patch(circ)
            ax.set_aspect('equal')
        plt.show()
    
    return set1_shift, set2_shift

def color_swap_by_relative_dist(target_scene_paths, reference_img_path, n_clust = None, n_clust_target = None, one_to_one = True, display=False):
    """
    Applies color palette from a reference image to target images based on relative color distances.
    
    This function extracts color palettes from both reference and target images, then maps
    colors between them based on their relative positions in the color space, after centering
    and normalizing. The target images are recolored using the matched colors.
    
    Args:
        target_scene_paths (list): List of file paths to target images to be color-shifted.
        reference_img_path (str): File path to the reference image to extract palette from.
        n_clust (int, optional): Number of palette colors to use. If None, determined
            automatically using silhouette score on the first target image. Defaults to None.
        n_clust_target (int, optional): Number of palette colors to use for the reference image.
            If None, uses the same value as n_clust. Defaults to None.
        one_to_one (bool, optional): If True, uses linear_sum_assignment for optimal one-to-one
            matching. If False, uses simple nearest-neighbor matching. Defaults to True.
    
    Returns:
        list: List of recolored images as numpy.ndarray in RGB color space (uint8).
    """
    
    target_scene = [color.rgb2lab(io.imread(target_path)) for target_path in target_scene_paths] # gets cielab imgs
    reference_img = color.rgb2lab(io.imread(reference_img_path))
    
    outputs = []

    if n_clust == None: # uses silloute score method to get empty n_cluster
        _ , n_clust = posterize_v1(target_scene[0])

    if n_clust_target == None:
        n_clust_target = n_clust
    
    kcolor_ref, _ = posterize_v1(reference_img, nclust = n_clust_target)
    kcolor_ref_shift = shift_points(kcolor_ref.cluster_centers_, normalize = True)

    
    for img_num in range(len(target_scene)):
        kcolor_target, n_clust = posterize_v1(target_scene[img_num], nclust = n_clust)
        kcolor_target_shift = shift_points(kcolor_target.cluster_centers_, normalize = True)
        
        if one_to_one:
            _, new_colors = linear_sum_assignment(all_dist(kcolor_target_shift, kcolor_ref_shift) * -1, maximize = True)
            
        else:
            new_colors = simple_match(kcolor_target_shift, kcolor_ref_shift) # color pairings
        
        new_colors = np.array([kcolor_ref.cluster_centers_[color] for color in new_colors])
        target_color_groups = color_by_number(target_scene[img_num], kcolor_target) # get imgs in color by number form
        
        
        outputs.append(construct_img(target_scene[img_num], target_color_groups, new_colors))

    # Display the outputs if requested
    if display:
        fig = display_color_transfer_results(reference_img_path, target_scene_paths, outputs)

    outputs = [np.uint8(np.float32(color.lab2rgb(img)) * 255) for img in outputs]

    return outputs

def color_swap_by_depth(target_scene_paths, reference_img_path, n_clust = None, display=False):
    """
    Applies color palette from a reference image to target images based on depth ordering.
    
    This function extracts color palettes from both reference and target images, then
    orders colors by their average depth in their respective images. It then maps colors
    from the reference to the target based on their depth rank order (e.g., the deepest
    color in the reference is mapped to the deepest color in the target).
    
    Args:
        target_scene_paths (list): List of file paths to target images to be color-shifted.
        reference_img_path (str): File path to the reference image to extract palette from.
        n_clust (int, optional): Number of palette colors to use. If None, determined 
            automatically using silhouette score on the first target image. Defaults to None.
    
    Returns:
        list: List of recolored images as numpy.ndarray in RGB color space (int).
    """
    
    target_scene = [color.rgb2lab(io.imread(target_path)) for target_path in target_scene_paths] # gets cielab imgs
    reference_img = color.rgb2lab(io.imread(reference_img_path))
    
    outputs = []

    if n_clust == None: # uses silloute score method to get empty n_cluster
        _ , n_clust = posterize_v1(target_scene[0])
    
    kcolor_ref, _ = posterize_v1(reference_img, nclust = n_clust)
    ref_depth = depth_model.infer_image(cv2.imread(reference_img_path)) # get depth maps
    ref_color_groups = color_by_number(reference_img, kcolor_ref) # get imgs in color by number form

    ref_color_depths = [] # will fill these with the average depth at every color in each pallete

    
    for clust in range(n_clust):
        # get mask of just the color groups
        ref_clust = ref_color_groups == clust
        # gets avg depth of each color
        ref_color_depths.append(np.sum(ref_clust * ref_depth) / np.sum(ref_clust))

    ref_keys = list(range(n_clust))
    ref_keys.sort(key = lambda x: ref_color_depths[x]) # pallete indeces soreted by depth

    
    for img_num in range(len(target_scene)):
        kcolor_target, n_clust = posterize_v1(target_scene[img_num], nclust = n_clust)
        target_depth = depth_model.infer_image(cv2.imread(target_scene_paths[img_num]))
        target_color_groups = color_by_number(target_scene[img_num], kcolor_target) # get imgs in color by number form

        target_color_depths = [] # will fill these with the average depth at every color in each pallete

        for clust in range(n_clust):
            target_clust = target_color_groups == clust # get mask of just the color groups
            target_color_depths.append(np.sum(target_clust * target_depth) / np.sum(target_clust)) # add average depths to associated lists

        target_keys = list(range(n_clust))
        target_keys.sort(key = lambda x: target_color_depths[x])
        
        new_colors = np.array([kcolor_ref.cluster_centers_[ref_keys[target_keys.index(i)]] for i in range(n_clust)]) # color pairings
        outputs.append(construct_img(target_scene[img_num], target_color_groups, new_colors))


    # Display the outputs if requested
    if display:
        fig = display_color_transfer_results(reference_img_path, target_scene_paths, outputs)

    # outputs = [((color.lab2rgb(img)) * 255).astype('int') for img in outputs]
    outputs = [np.uint8(np.float32(color.lab2rgb(img)) * 255) for img in outputs]
    
    return outputs


def color_frequency_sort(kmeans, img):
    """
    Sorts the color centers from a K-means model by their frequency in the image.
    
    Args:
        kmeans (sklearn.cluster.KMeans): Fitted K-means model containing cluster centers
            and labels. The cluster_centers_ attribute should contain colors in CIELAB space.
        img (numpy.ndarray): Original CIELAB image used to calculate the frequency of each color.
    
    Returns:
        tuple: (sorted_centers, percentages)
            - sorted_centers: numpy.ndarray of color centers sorted by frequency (highest to lowest)
            - percentages: numpy.ndarray of corresponding percentages for each color
    """
    centers = kmeans.cluster_centers_
    n_colors = centers.shape[0]
    
    # Get cluster assignments for each pixel
    poster_vals = color_by_number(img, kmeans)
    
    # Count occurrences of each cluster
    counts = np.bincount(poster_vals.flatten(), minlength=n_colors)
    
    # Convert to percentages
    percentages = 100 * (counts / counts.sum())
    
    # Get sorting indices (highest frequency first)
    sort_indices = np.argsort(-percentages)
    
    # Return sorted centers and their corresponding percentages
    sorted_centers = centers[sort_indices]
    sorted_percentages = percentages[sort_indices]
    
    return sorted_centers

def euclidean_distance_vectors(p1, points):
    """
    Calculate the Euclidean distance between a point and an array of points.
    
    Args:
        p1 (array): Reference point as an array [a, b]
        points (array): Array of points, shape (n, 2)
        
    Returns:
        array: Array of distances from p1 to each point in points
    """
    return np.linalg.norm(points - p1, axis=1)

    
def match_colors_many_to_one(color_set_A, color_set_B):
    """
    Match each color in set A to its closest color in set B (many-to-one matching).
    
    This function finds the closest color in set B for each color in set A using Euclidean distance.
    Multiple colors from set A can match to the same color in set B (many-to-one relationship).
    
    Args:
        color_set_A (numpy.ndarray): Array of colors with shape (n, 2), 
                                    where each row is a color point in a*b* space.
                                    Should be sorted by frequency (highest first).
        color_set_B (numpy.ndarray): Array of colors with shape (m, 2),
                                    where each row is a color point in a*b* space.
    
    Returns:
        list: A list of tuples, where each tuple contains a pair of matched colors
              in the form (color_A, color_B). Each color is a numpy array
              with shape (2,) representing the a*b* values.
    
    Notes:
        - Does NOT ensure one-to-one matching (colors in B can be reused)
        - Uses Euclidean distance as the color similarity metric
    """
    
    closest_pairs = []
    
    for s in color_set_A:
        # Calculate distances to all points in S_target
        distances = euclidean_distance_vectors(s, color_set_B)
        # Find index of minimum distance
        closest_idx = np.argmin(distances)
        closest_pairs.append((s, color_set_B[closest_idx]))

    return closest_pairs

def match_colors_kdtree(color_set_A, color_set_B):
   """
   Create a one-to-one mapping between two color sets using a K-d tree algorithm.
   
   This function matches each color from set A to its closest available color in set B,
   ensuring each color in set B is used only once. It uses a K-d tree spatial data 
   structure for efficient nearest-neighbor searches.
   
   Args:
       color_set_A (numpy.ndarray): Array of colors with shape (n, 2), 
                                   where each row is a color point in a*b* space ALREADY sorted on frequence highest frequency appears first in array A 
       color_set_B (numpy.ndarray): Array of colors with shape (n, 2),
                                   where each row is a color point in a*b* space.
   
   Returns:
       list: A list of tuples, where each tuple contains a pair of matched colors
             in the form (color_A, color_B). Each color is a numpy array
             with shape (2,) representing the a*b* values.
   
   Raises:
       AssertionError: If the color sets have different lengths.
   
   Notes:
       - This implementation assumes both arrays have the same size
       - Uses Euclidean distance as the color similarity metric
   """
   # Verify same size
   assert len(color_set_A) == len(color_set_B), "Arrays must have the same length"
   
   # Build the K-d tree
   tree = KDTree(color_set_B)
   # Empty set to track which colors have been assigned
   used_indices = set()
   # Empty list to store the matched color pairs
   pairs = []
   # process each color from set A
   for color_A in color_set_A:
       # Calculate how many points we need to query
       points_remaining = len(color_set_B) - len(used_indices)
       k = min(3 * points_remaining, len(color_set_B))
       
       # Get k nearest neighbors
       # (distances to each target)
       # (indices of targets sorted by distance)
       # The query() method internally calculates the Euclidean distance between the query point s and all points in the K-d tree (which contains the target colors).
       distances, indices = tree.query(color_A, k=k)
       
       # Find first unused color from set B
       # loops from nearest point 
       for idx in indices:
           # if color is not already used, make a match
           if idx not in used_indices:
               pairs.append((color_A, color_set_B[idx]))
               used_indices.add(idx)
               break
   
   return pairs


def transfer_colors_target(km_target, color_matches):
    """
    Transfer colors to target clusters based on frequency priority in the target.
    
    This function replaces each cluster center in the target K-means model with its
    corresponding matched color from the source, prioritizing by the target's
    frequency ordering (most frequent target colors get matched first).
    
    Args:
        km_target (sklearn.cluster.KMeans): Fitted K-means model from target containing 
                   the cluster_centers_ attribute that stores the representative colors.
        color_matches (list): A list of tuples, where each tuple contains a pair of matched 
                   colors in the form (target_color, source_color). Each color is a numpy 
                   array with shape (2,) representing the a*b* values.
    
    Returns:
        numpy.ndarray: New cluster centers where each original target cluster center 
                      has been replaced with its matched source color.
    """
    # Create a copy of the original cluster centers to modify
    new_cluster_centers = np.copy(km_target.cluster_centers_)
    
    # For each cluster center, find its match in the color_matches
    for i, cluster_color in enumerate(km_target.cluster_centers_):
        for target_color, source_color in color_matches:
            # If this cluster color matches target_color, replace it with source_color
            if np.array_equal(cluster_color, target_color):
                new_cluster_centers[i] = source_color
                break
                 
    return new_cluster_centers


def transfer_colors_source(km_target, color_matches):
    """
    Transfer colors to target clusters based on frequency priority in the source.
    
    This function replaces each cluster center in the target K-means model with its
    corresponding matched color from the source, prioritizing by the source's
    frequency ordering (most frequent source colors are matched first).
    
    Args:
        km_target (sklearn.cluster.KMeans): Fitted K-means model from target containing 
                   the cluster_centers_ attribute that stores the representative colors.
        color_matches (list): A list of tuples, where each tuple contains a pair of matched 
                   colors in the form (source_color, target_color). Each color is a numpy 
                   array with shape (2,) representing the a*b* values.
    
    Returns:
        numpy.ndarray: New cluster centers where each original target cluster center 
                      has been replaced with its matched source color.
    """
    # Create a copy of the original cluster centers to modify
    new_cluster_centers = np.copy(km_target.cluster_centers_)
    
    # For each cluster center, find its match in the color_matches
    for i, cluster_color in enumerate(km_target.cluster_centers_):
        for source_color, target_color in color_matches:
            # If this cluster color matches target_color, replace it with source_color
            if np.array_equal(cluster_color, target_color):
                new_cluster_centers[i] = source_color
                break
                 
    return new_cluster_centers

def print_color_matches(matches):
    """
    Prints a formatted list of color matches to the console.

    Args:
        matches (list): A list of tuples, where each tuple contains a pair of matched colors
                       in the form (color_A, color_B). Each color is expected to be a numpy array
                       that can be converted to a list.
    
    Returns:
        None: This function prints output to the console but does not return a value.
    """
    for i, (color_A, color_B) in enumerate(matches):
        print(f"Match {i+1}: {color_A.tolist()} → {color_B.tolist()}")


def display_color_palette(kmeans, img, sort_by='frequency', figsize=(12, 3), show_percentage=True, show_labels=False, ax=None):
    """
    Displays the color palette extracted from an image using K-means clustering.
    
    This function visualizes the color clusters found by K-means as a color palette,
    primarily sorting colors by their frequency in the image.
    
    Args:
        kmeans (sklearn.cluster.KMeans): Fitted K-means model containing cluster centers
            and labels. The cluster_centers_ attribute should contain colors in a*b* space.
        img (numpy.ndarray): Original CIELAB image, used to calculate the percentage 
            of each color in the image and extract lightness information.
        sort_by (str, optional): Method to sort colors in the palette:
            - 'frequency': Sort by frequency of appearance
            - None: Use the original order from K-means. Defaults to 'frequency'.
        figsize (tuple, optional): Figure size in inches (width, height). Larger values
            create a bigger palette. Defaults to (12, 3).
        show_percentage (bool, optional): Whether to display the percentage of each
            color in the image. Defaults to True.
        show_labels (bool, optional): Whether to display RGB values under each color.
            Defaults to False.
        ax (matplotlib.axes.Axes, optional): Matplotlib axes object to plot on.
            If None, a new figure and axes will be created.
    
    Returns:
        matplotlib.axes.Axes: The axes containing the color palette.
    """
    centers = kmeans.cluster_centers_  # These are a*b* values only
    n_colors = centers.shape[0] # number of clusters 
    
    # Get cluster assignments for each pixel
    poster_vals = color_by_number(img, kmeans)
    
    # Calculate percentages
    percentages = None
    if show_percentage:
        # Count occurrences of each cluster
        counts = np.bincount(poster_vals.flatten(), minlength=n_colors)
        # Convert to percentages
        percentages = 100 * (counts / counts.sum())
    
    # Sort colors by frequency
    sorted_indices = list(range(n_colors))
    if sort_by == 'frequency':
        # Sort by frequency (highest first)
        counts = np.bincount(poster_vals.flatten(), minlength=n_colors)
        sorted_indices = np.argsort(-counts)
    
    # Create a figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Store RGB colors for labels
    rgb_colors = []
    
    # For each cluster, calculate average lightness from original image
    avg_lightness = np.zeros(n_colors)
    for i in range(n_colors):
        mask = (poster_vals == i)
        if np.any(mask):
            avg_lightness[i] = np.mean(img[:,:,0][mask])
        else:
            avg_lightness[i] = 50  # Default L* if no pixels in this cluster
    
    # Draw the color palette
    for i, idx in enumerate(sorted_indices):
        # Construct complete LAB color with average lightness and cluster center a*b*
        lab_color = np.zeros((1, 1, 3))
        lab_color[0, 0, 0] = avg_lightness[idx]  # Use average L* from original image
        lab_color[0, 0, 1:] = centers[idx]       # Use a*b* from cluster center
        
        # Convert to RGB for display
        rgb_color = color.lab2rgb(lab_color)[0, 0]
        rgb_colors.append(rgb_color)
        
        # Draw a rectangle for this color
        width = 1.0 / n_colors
        rect = plt.Rectangle((i * width, 0), width, 1, color=rgb_color)
        ax.add_patch(rect)
        
        # Add percentage label if requested
        if percentages is not None and show_percentage:
            percentage = percentages[idx]
            ax.text(i * width + width/2, 0.5, f"{percentage:.1f}%", 
                   ha='center', va='center', 
                   color='white' if avg_lightness[idx] < 50 else 'black',
                   fontweight='bold')
    
    # Set the aspect and remove axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    
    # Add RGB value labels if requested
    if show_labels:
        for i, clr in enumerate(rgb_colors):
            r, g, b = [int(c * 255) for c in clr]
            width = 1.0 / n_colors
            ax.text(i * width + width/2, 0.1, f"#{r:02X}{g:02X}{b:02X}", 
                   ha='center', va='center', fontsize=8)
    
    return ax

def display_color_transfer_results(reference_img_path, target_scene_paths, outputs, max_cols=3, fig_size_per_cell=(5, 5), titles=None):
    """
    Display color transfer results in a grid layout.
    
    Args:
        reference_img_path (str): Path to the reference image
        target_scene_paths (list): List of paths to target images
        outputs (list): List of output images in LAB color space
        max_cols (int, optional): Maximum number of columns in the grid. Defaults to 3.
        fig_size_per_cell (tuple, optional): Figure size per cell in inches. Defaults to (5, 5).
        titles (dict, optional): Custom titles for different sections. Defaults to None.
    
    Returns:
        matplotlib.figure.Figure: The created figure object
    """
    # Use default titles if none provided
    if titles is None:
        titles = {
            'reference': 'Reference Image',
            'target': 'Original Target',
            'output': 'Recolored Output'
        }
    
    # Calculate number of rows and columns for the grid
    n_images = 1 + len(target_scene_paths) + len(outputs)
    n_cols = min(max_cols, n_images)  # Max columns
    n_rows = (n_images + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure and subplots
    width, height = fig_size_per_cell
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width*n_cols, height*n_rows))
    
    # Convert to array and flatten for easier indexing
    axes = np.array(axes).flatten() if n_rows * n_cols > 1 else np.array([axes])
    
    # Display reference image
    ref_rgb = io.imread(reference_img_path)
    axes[0].imshow(ref_rgb)
    axes[0].set_title(titles['reference'])
    axes[0].axis('off')
    
    # Display original target images
    for i, target_path in enumerate(target_scene_paths):
        idx = i + 1
        if idx < len(axes):  # Check bounds
            target_rgb = io.imread(target_path)
            axes[idx].imshow(target_rgb)
            axes[idx].set_title(f"{titles['target']} {i+1}")
            axes[idx].axis('off')
    
    # Display output images
    for i, output in enumerate(outputs):
        idx = i + len(target_scene_paths) + 1
        if idx < len(axes):  # Check bounds
            # Convert LAB to RGB for display
            output_rgb = color.lab2rgb(output)
            output_rgb = np.clip(output_rgb, 0, 1)  # Clip values
            
            axes[idx].imshow(output_rgb)
            axes[idx].set_title(f"{titles['output']} {i+1}")
            axes[idx].axis('off')
    
    # Hide any unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    return fig 

def color_swap_by_target_frequency_manyto1(target_scene_paths, reference_img_path, n_clust=None, display=False, video_save=False):
    """
    Apply color palette from a reference image to target images based on target color frequency.
    
    This function transfers colors from a reference image to target images using a many-to-one
    matching approach based on color frequency. It extracts color palettes from both reference
    and target images using K-means clustering, sorts target colors by frequency, and maps them
    to reference colors. Multiple target colors can map to the same reference color.
    
    Args:
        target_scene_paths (list): List of file paths to target images to be color-shifted.
        reference_img_path (str): File path to the reference image to extract palette from.
        n_clust (int, optional): Number of palette colors to use. If None, determined 
            automatically using silhouette score on the first target image. Defaults to None.
        display (bool, optional): Whether to display the results using 
            display_color_transfer_results(). Defaults to False.
        video_save (bool, optional): Whether to format output images for video saving. If True,
            converts images to 8-bit RGB with values in range [0, 255]. If False, keeps images
            as floating-point RGB with values in range [0, 1]. Defaults to False.
    
    Returns:
        list: List of recolored images as numpy.ndarray. Format depends on video_save parameter:
            - If video_save=False: RGB float arrays with values in range [0, 1]
            - If video_save=True: RGB integer arrays with values in range [0, 255]
    
    Notes:
        - Colors are processed in CIELAB color space for perceptual uniformity
        - Target colors are sorted by frequency before matching
        - Many-to-one matching allows multiple target colors to map to the same reference color
    """
    
    # Load and convert images to CIELAB color space
    target_scene = [color.rgb2lab(io.imread(target_path)) for target_path in target_scene_paths]
    reference_img = color.rgb2lab(io.imread(reference_img_path))
    
    outputs = []
    if n_clust == None:  # Determine optimal number of clusters if not provided
        _, n_clust = posterize_v1(target_scene[0])
    
    # Get K-means model for reference image
    kmeans_source, _ = posterize_v1(reference_img, nclust=n_clust)

    display_color_palette(kmeans_source, reference_img)
    
    for img_num in range(len(target_scene)):
        # Get K-means model for target image
        kmeans_target, _ = posterize_v1(target_scene[img_num], nclust=n_clust)
        
        # Sort target centers by frequency
        sorted_centers_target = color_frequency_sort(kmeans_target, target_scene[img_num])
        
        # Match target sorted centers to source centers (target = source)
        matches = match_colors_many_to_one(sorted_centers_target, kmeans_source.cluster_centers_)

        # Print matched color pairs for debugging
        # print_color_matches(matches)
        
        # Transfer colors based on target frequency
        new_centers = transfer_colors_target(kmeans_target, matches)
        
        # Get color assignments for each pixel in target image
        color_by_num = color_by_number(target_scene[img_num], kmeans_target)
        
        # Construct the output image with new colors
        output = construct_img(target_scene[img_num], color_by_num, new_centers)
        
        outputs.append(output)
        
    # Display the outputs if requested
    if display:
        fig = display_color_transfer_results(reference_img_path, target_scene_paths, outputs)
    
    if not video_save: 
        # Convert output images from CIELAB back to RGB
        outputs = [color.lab2rgb(output) for output in outputs]
    elif video_save: 
        outputs = [np.uint8(np.float32(color.lab2rgb(img)) * 255) for img in outputs]
        
    return outputs

def color_swap_by_target_frequency_oneto1(target_scene_paths, reference_img_path, n_clust=None, display=False, video_save=False):
    """
    Apply color palette from a reference image to target images based on one-to-one frequency matching.
    
    This function transfers colors from a reference image to target images using a one-to-one
    matching approach based on color frequency. It extracts color palettes from both reference
    and target images using K-means clustering, sorts target colors by frequency, and maps them
    to reference colors using KD-tree matching to ensure a one-to-one correspondence between colors.
    
    Args:
        target_scene_paths (list): List of file paths to target images to be color-shifted.
        reference_img_path (str): File path to the reference image to extract palette from.
        n_clust (int, optional): Number of palette colors to use. If None, determined 
            automatically using silhouette score on the first target image. Defaults to None.
        display (bool, optional): Whether to display the results using 
            display_color_transfer_results(). Defaults to False.
        video_save (bool, optional): Whether to format output images for video saving. If True,
            converts images to 8-bit RGB with values in range [0, 255]. If False, keeps images
            as floating-point RGB with values in range [0, 1]. Defaults to False.
    
    Returns:
        list: List of recolored images as numpy.ndarray. Format depends on video_save parameter:
            - If video_save=False: RGB float arrays with values in range [0, 1]
            - If video_save=True: RGB integer arrays with values in range [0, 255]
    
    Notes:
        - Colors are processed in CIELAB color space for perceptual uniformity
        - Target colors are sorted by frequency before matching
        - One-to-one matching ensures each reference color is used at most once
        - Uses KD-tree algorithm for efficient nearest-neighbor color matching
    """
    
    # Load and convert images to CIELAB color space
    target_scene = [color.rgb2lab(io.imread(target_path)) for target_path in target_scene_paths]
    reference_img = color.rgb2lab(io.imread(reference_img_path))
    
    outputs = []
    if n_clust == None:  # Determine optimal number of clusters if not provided
        _, n_clust = posterize_v1(target_scene[0])
    
    # Get K-means model for reference image
    kmeans_source, _ = posterize_v1(reference_img, nclust=n_clust)

    display_color_palette(kmeans_source, reference_img)
    
    for img_num in range(len(target_scene)):
        # Get K-means model for target image
        kmeans_target, _ = posterize_v1(target_scene[img_num], nclust=n_clust)
        
        # Sort target centers by frequency (most frequent colors first)
        sorted_centers_target = color_frequency_sort(kmeans_target, target_scene[img_num])
        
        # Match target sorted centers to source centers using KD-tree for one-to-one matching
        matches = match_colors_kdtree(sorted_centers_target, kmeans_source.cluster_centers_)
        
        # Print matched color pairs for debugging
        # print_color_matches(matches)
        
        # Transfer colors based on target frequency using the one-to-one matches
        new_centers = transfer_colors_target(kmeans_target, matches)
        
        # Get color assignments for each pixel in target image
        color_by_num = color_by_number(target_scene[img_num], kmeans_target)
        
        # Construct the output image with new colors
        output = construct_img(target_scene[img_num], color_by_num, new_centers)
        
        outputs.append(output)
        
    # Display the outputs if requested
    if display:
        fig = display_color_transfer_results(reference_img_path, target_scene_paths, outputs)
    
    if not video_save: 
        # Convert output images from CIELAB back to RGB
        outputs = [color.lab2rgb(output) for output in outputs]
    elif video_save: 
        outputs = [np.uint8(np.float32(color.lab2rgb(img)) * 255) for img in outputs]
        
    return outputs

def color_swap_by_source_frequency_oneto1(target_scene_paths, reference_img_path, n_clust=None, display=False, video_save=False):
    """
    Apply color palette from a reference image to target images based on source frequency matching.
    
    This function transfers colors from a reference image to target images using a one-to-one
    mapping based on the frequency of colors in the source (reference) image. It extracts color 
    palettes from both reference and target images using K-means clustering, sorts source colors 
    by frequency, and maps them to target colors using KD-tree matching to ensure a one-to-one
    correspondence.
    
    Args:
        target_scene_paths (list): List of file paths to target images to be color-shifted.
        reference_img_path (str): File path to the reference image to extract palette from.
        n_clust (int, optional): Number of palette colors to use. If None, determined 
            automatically using silhouette score on the first target image. Defaults to None.
        display (bool, optional): Whether to display the results using 
            display_color_transfer_results(). Defaults to False.
        video_save (bool, optional): Whether to format output images for video saving. If True,
            converts images to 8-bit RGB with values in range [0, 255]. If False, keeps images
            as floating-point RGB with values in range [0, 1]. Defaults to False.
    
    Returns:
        list: List of recolored images as numpy.ndarray in CIELAB color space.
    
    Notes:
        - Unlike target frequency matching, this method prioritizes the most frequent colors
          in the source/reference image rather than the target
        - Colors are processed in CIELAB color space for perceptual uniformity
        - Source colors are sorted by frequency before matching
        - One-to-one matching ensures each target color is used at most once
        - Uses KD-tree algorithm for efficient nearest-neighbor color matching
    """
    
    # Load and convert images to CIELAB color space
    target_scene = [color.rgb2lab(io.imread(target_path)) for target_path in target_scene_paths]
    reference_img = color.rgb2lab(io.imread(reference_img_path))
    
    outputs = []
    if n_clust is None:  # Determine optimal number of clusters if not provided
        _, n_clust = posterize_v1(target_scene[0])
    
    # Get K-means model for reference image
    kmeans_source, _ = posterize_v1(reference_img, nclust=n_clust)

    display_color_palette(kmeans_source, reference_img)
    
    # Sort source centers by frequency (most frequent colors first)
    sorted_centers_source = color_frequency_sort(kmeans_source, reference_img)
    
    for img_num in range(len(target_scene)):
        # Get K-means model for target image
        kmeans_target, _ = posterize_v1(target_scene[img_num], nclust=n_clust)
        
        # Match sorted source centers to target centers using KD-tree for one-to-one mapping
        # This creates pairs of (source_color, target_color)
        matches = match_colors_kdtree(sorted_centers_source, kmeans_target.cluster_centers_)
        
        # # Print matched color pairs for debugging
        # print_color_matches(matches)
        
        # Transfer colors based on source frequency 
        # (replaces target colors with matched source colors)
        new_centers = transfer_colors_source(kmeans_target, matches)
        
        # Get color assignments for each pixel in target image
        color_by_num = color_by_number(target_scene[img_num], kmeans_target)
        
        # Construct the output image with new colors
        output = construct_img(target_scene[img_num], color_by_num, new_centers)
        
        outputs.append(output)
        
    # Display the outputs if requested
    if display:
        fig = display_color_transfer_results(reference_img_path, target_scene_paths, outputs)
    
    if not video_save: 
        # Convert output images from CIELAB back to RGB
        outputs = [color.lab2rgb(output) for output in outputs]
    elif video_save: 
        outputs = [np.uint8(np.float32(color.lab2rgb(img)) * 255) for img in outputs]
        
    return outputs
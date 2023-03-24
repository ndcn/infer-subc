# centrosome routines
import numpy as np

from scipy.ndimage import distance_transform_edt, sum, minimum_filter, maximum_filter


def size_similarly(labels, secondary):
    """Size the secondary matrix similarly to the labels matrix

    labels - labels matrix
    secondary - a secondary image or labels matrix which might be of
                different size.
    Return the resized secondary matrix and a mask indicating what portion
    of the secondary matrix is bogus (manufactured values).

    Either the mask is all ones or the result is a copy, so you can
    modify the output within the unmasked region w/o destroying the original.
    """
    if labels.shape[:2] == secondary.shape[:2]:
        return secondary, np.ones(secondary.shape, bool)
    if labels.shape[0] <= secondary.shape[0] and labels.shape[1] <= secondary.shape[1]:
        if secondary.ndim == 2:
            return (
                secondary[: labels.shape[0], : labels.shape[1]],
                np.ones(labels.shape, bool),
            )
        else:
            return (
                secondary[: labels.shape[0], : labels.shape[1], :],
                np.ones(labels.shape, bool),
            )

    #
    # Some portion of the secondary matrix does not cover the labels
    #
    result = np.zeros(list(labels.shape) + list(secondary.shape[2:]), secondary.dtype)
    i_max = min(secondary.shape[0], labels.shape[0])
    j_max = min(secondary.shape[1], labels.shape[1])
    if secondary.ndim == 2:
        result[:i_max, :j_max] = secondary[:i_max, :j_max]
    else:
        result[:i_max, :j_max, :] = secondary[:i_max, :j_max, :]
    mask = np.zeros(labels.shape, bool)
    mask[:i_max, :j_max] = 1
    return result, mask


def distance_to_edge(labels):
    """Compute the distance of a pixel to the edge of its object

    labels - a labels matrix

    returns a matrix of distances
    """
    colors = color_labels(labels)
    max_color = np.max(colors)
    result = np.zeros(labels.shape)
    if max_color == 0:
        return result

    for i in range(1, max_color + 1):
        mask = colors == i
        result[mask] = distance_transform_edt(mask)[mask]
    return result


def color_labels(labels, distance_transform=False):
    """Color a labels matrix so that no adjacent labels have the same color

    distance_transform - if true, distance transform the labels to find out
         which objects are closest to each other.

    Create a label coloring matrix which assigns a color (1-n) to each pixel
    in the labels matrix such that all pixels similarly labeled are similarly
    colored and so that no similiarly colored, 8-connected pixels have
    different labels.

    You can use this function to partition the labels matrix into groups
    of objects that are not touching; you can then operate on masks
    and be assured that the pixels from one object won't interfere with
    pixels in another.

    returns the color matrix
    """
    if distance_transform:
        i, j = distance_transform_edt(labels == 0, return_distances=False, return_indices=True)
        dt_labels = labels[i, j]
    else:
        dt_labels = labels
    # Get the neighbors for each object
    v_count, v_index, v_neighbor = find_neighbors(dt_labels)
    # Quickly get rid of labels with no neighbors. Greedily assign
    # all of these a color of 1
    v_color = np.zeros(len(v_count) + 1, int)  # the color per object - zero is uncolored
    zero_count = v_count == 0
    if np.all(zero_count):
        # can assign all objects the same color
        return (labels != 0).astype(int)
    v_color[1:][zero_count] = 1
    v_count = v_count[~zero_count]
    v_index = v_index[~zero_count]
    v_label = np.argwhere(~zero_count).transpose()[0] + 1
    # If you process the most connected labels first and use a greedy
    # algorithm to preferentially assign a label to an existing color,
    # you'll get a coloring that uses 1+max(connections) at most.
    #
    # Welsh, "An upper bound for the chromatic number of a graph and
    # its application to timetabling problems", The Computer Journal, 10(1)
    # p 85 (1967)
    #
    sort_order = np.lexsort([-v_count])
    v_count = v_count[sort_order]
    v_index = v_index[sort_order]
    v_label = v_label[sort_order]
    for i in range(len(v_count)):
        neighbors = v_neighbor[v_index[i] : v_index[i] + v_count[i]]
        colors = np.unique(v_color[neighbors])
        if colors[0] == 0:
            if len(colors) == 1:
                # only one color and it's zero. All neighbors are unlabeled
                v_color[v_label[i]] = 1
                continue
            else:
                colors = colors[1:]
        # The colors of neighbors will be ordered, so there are two cases:
        # * all colors up to X appear - colors == np.arange(1,len(colors)+1)
        # * some color is missing - the color after the first missing will
        #   be mislabeled: colors[i] != np.arange(1, len(colors)+1)
        crange = np.arange(1, len(colors) + 1)
        misses = crange[colors != crange]
        if len(misses):
            color = misses[0]
        else:
            color = len(colors) + 1
        v_color[v_label[i]] = color
    return v_color[labels]


def find_neighbors(labels):
    """Find the set of objects that touch each object in a labels matrix

    Construct a "list", per-object, of the objects 8-connected adjacent
    to that object.
    Returns three 1-d arrays:
    * array of #'s of neighbors per object
    * array of indexes per object to that object's list of neighbors
    * array holding the neighbors.

    For instance, say 1 touches 2 and 3 and nobody touches 4. The arrays are:
    [ 2, 1, 1, 0], [ 0, 2, 3, 4], [ 2, 3, 1, 1]
    """
    max_label = np.max(labels)
    # Make a labels matrix with zeros around the edges so we can do index
    # offsets without worrying.
    #
    new_labels = np.zeros(np.array(labels.shape) + 2, labels.dtype)
    new_labels[1:-1, 1:-1] = labels
    labels = new_labels
    # Only consider the points that are next to others
    adjacent_mask = adjacent(labels)
    adjacent_i, adjacent_j = np.argwhere(adjacent_mask).transpose()
    # Get matching vectors of labels and neighbor labels for the 8
    # compass directions.
    count = len(adjacent_i)
    if count == 0:
        return (np.zeros(max_label, int), np.zeros(max_label, int), np.zeros(0, int))
    # The following bizarre construct does the following:
    # labels[adjacent_i, adjacent_j] looks up the label for each pixel
    # [...]*8 creates a list of 8 references to it
    # np.hstack concatenates, giving 8 repeats of the list
    v_label = np.hstack([labels[adjacent_i, adjacent_j]] * 8)
    v_neighbor = np.zeros(count * 8, int)
    index = 0
    for i, j in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
        v_neighbor[index : index + count] = labels[adjacent_i + i, adjacent_j + j]
        index += count
    #
    # sort by label and neighbor
    #
    sort_order = np.lexsort((v_neighbor, v_label))
    v_label = v_label[sort_order]
    v_neighbor = v_neighbor[sort_order]
    #
    # eliminate duplicates by comparing each element after the first one
    # to its previous
    #
    first_occurrence = np.ones(len(v_label), bool)
    first_occurrence[1:] = (v_label[1:] != v_label[:-1]) | (v_neighbor[1:] != v_neighbor[:-1])
    v_label = v_label[first_occurrence]
    v_neighbor = v_neighbor[first_occurrence]
    #
    # eliminate neighbor = self and neighbor = background
    #
    to_remove = (v_label == v_neighbor) | (v_neighbor == 0)
    v_label = v_label[~to_remove]
    v_neighbor = v_neighbor[~to_remove]
    #
    # The count of # of neighbors
    #
    v_count = fixup_scipy_ndimage_result(sum(np.ones(v_label.shape), v_label, np.arange(max_label, dtype=np.int32) + 1))
    v_count = v_count.astype(int)
    #
    # The index into v_neighbor
    #
    v_index = np.cumsum(v_count)
    v_index[1:] = v_index[:-1]
    v_index[0] = 0
    return (v_count, v_index, v_neighbor)


def fixup_scipy_ndimage_result(whatever_it_returned):
    """Convert a result from scipy.ndimage to a numpy array

    scipy.ndimage has the annoying habit of returning a single, bare
    value instead of an array if the indexes passed in are of length 1.
    For instance:
    scipy.ndimage.maximum(image, labels, [1]) returns a float
    but
    scipy.ndimage.maximum(image, labels, [1,2]) returns a list
    """
    if getattr(whatever_it_returned, "__getitem__", False):
        return np.array(whatever_it_returned)
    else:
        return np.array([whatever_it_returned])


def adjacent(labels):
    """Return a binary mask of all pixels which are adjacent to a pixel of
    a different label.

    """
    high = labels.max() + 1
    if high > np.iinfo(labels.dtype).max:
        labels = labels.astype(np.int32)
    image_with_high_background = labels.copy()
    image_with_high_background[labels == 0] = high
    min_label = minimum_filter(
        image_with_high_background,
        footprint=np.ones((3, 3), bool),
        mode="constant",
        cval=high,
    )
    max_label = maximum_filter(labels, footprint=np.ones((3, 3), bool), mode="constant", cval=0)
    return (min_label != max_label) & (labels > 0)

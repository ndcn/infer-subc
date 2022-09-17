"""
Objects represent segmentation.  They are sub-classed from the BioImContainer and operated on by 
Transforms and Pipelines (a sequence of Transforms).  When a Transform perfors a thresholding and yields a 
boolean image it makes an Object (vs. Image)

Raw -> BioImImage [Transform]-> BioImImage [Tranform]-> BioImObject  

"""

import numpy as np
import scipy.ndimage
import scipy.sparse
from numpy.random.mtrand import RandomState

from .base import BioImContainer, BioImLike
from .image import BioImImage

# called "BioImObject - "bio-image object" to disambiguate from other Object classes which are running around"
class BioImObject(BioImImage):
    """Represents a segmentation of an image.
    IdentityPrimAutomatic produces three variants of its segmentation
    result. This object contains all three.
    There are three formats for segmentation, two of which support
    overlapping objects:
    get/set_segmented - legacy, a single plane of labels that does not
                        support overlapping objects
    get/set_labels - supports overlapping objects, returns one or more planes
                     along with indices. A typical usage is to perform an
                     operation per-plane as if the objects did not overlap.
    get/set_ijv    - supports overlapping objects, returns a sparse
                     representation in which the first two columns are the
                     coordinates and the last is the object number. This
                     is efficient for doing things like calculating intensity
                     per-object.
    You can set one of the types and then get any of the types (except that
    get_segmented will raise an exception if objects overlap).
    """

    def __init__(self, child_obj: BioImLike, parent_img: BioImContainer = None):
        self.__segmented = None
        self.__unedited_segmented = None
        self.__small_removed_segmented = None
        self.__parent_image = None  # non-Object parent

    @property
    def ndim(self):
        if self.__parent_image:
            return self.__parent_image.ndim
        return len(self.shape)

    @property
    def has_parent_image(self):
        """True if this image has a defined parent"""
        return self.parent_image is not None

    @property
    def mask(self):
        """Return the mask (pixels to be considered) for the primary image"""
        if self._mask is not None:
            return self._mask

        if self.has_parent_image:
            return self.parent_image.mask

        # default to ones_like image... but no over channel.
        shape = self.image.shape
        if self.multichannel:
            shape = shape[-self.dimensions :]

        return np.ones(shape, dtype=bool)

    @property
    def file_name(self):
        """The name of the file holding this image
        If the image is derived, then return the file name of the first
        ancestor that has a file name. Return None if the image does not have
        an ancestor or if no ancestor has a file name.
        """
        if self._file_name is not None:
            return self._file_name
        elif self.has_parent_image:
            return self.parent_image.file_name
        else:
            return None

    @property
    def scale(self):
        """The scale at acquisition
        This is the intensity scale used by the acquisition device. For
        instance, a microscope might use a 12-bit a/d converter to acquire
        an image and store that information using the TIF MaxSampleValue
        tag = 4095.
        """
        if self._scale is None and self.has_parent_image:
            return self.parent_image.scale

        return self._scale

    @property
    def masked(self):
        mask = self.parent_image.mask

        return np.logical_and(self.segmented, mask)

    @property
    def shape(self):
        dense, _ = self.__segmented.get_dense()

        if dense.shape[3] == 1:
            return dense.shape[-2:]

        return dense.shape[-3:]

    @property
    def segmented(self):
        """Get the de-facto segmentation of the image into objects: a matrix
        of object numbers.
        """
        return self.__segmentation_to_labels(self.__segmented)

    @segmented.setter
    def segmented(self, labels):
        self.__segmented = self.__labels_to_segmentation(labels)

    @staticmethod
    def __labels_to_segmentation(labels):
        dense = downsample_labels(labels)

        if dense.ndim == 3:
            z, x, y = dense.shape
        else:
            x, y = dense.shape
            z = 1

        dense = dense.reshape((1, 1, 1, z, x, y))

        return Segmentation(dense=dense)

    @staticmethod
    def __segmentation_to_labels(segmentation):
        assert isinstance(segmentation, Segmentation), "Operation failed because objects were not initialized"

        dense, indices = segmentation.get_dense()

        assert len(dense) == 1, "Operation failed because objects overlapped. Please try with non-overlapping objects"

        if dense.shape[3] == 1:
            return dense.reshape(dense.shape[-2:])

        return dense.reshape(dense.shape[-3:])

    @property
    def indices(self):
        """Get the indices for a scipy.ndimage-style function from the segmented labels"""
        if len(self.ijv) == 0:
            return np.zeros(0, np.int32)
        max_label = np.max(self.ijv[:, 2])

        return np.arange(max_label).astype(np.int32) + 1

    @property
    def count(self):
        return len(self.indices)

    @property
    def areas(self):
        """The area of each object"""
        if len(self.indices) == 0:
            return np.zeros(0, int)

        return np.bincount(self.ijv[:, 2])[self.indices]

    def set_ijv(self, ijv, shape=None):
        """Set the segmentation to an IJV object format
        The ijv format is a list of i,j coordinates in slots 0 and 1
        and the label at the pixel in slot 2.
        """
        sparse = np.core.records.fromarrays(
            (ijv[:, 0], ijv[:, 1], ijv[:, 2]),
            [("y", ijv.dtype), ("x", ijv.dtype), ("label", ijv.dtype)],
        )
        if shape is not None:
            shape = (1, 1, 1, shape[0], shape[1])
        self.__segmented = Segmentation(sparse=sparse, shape=shape)

    def get_ijv(self):
        """Get the segmentation in IJV object format
        The ijv format is a list of i,j coordinates in slots 0 and 1
        and the label at the pixel in slot 2.
        """
        sparse = self.__segmented.sparse
        return np.column_stack([sparse[axis] for axis in ("y", "x", "label")])

    ijv = property(get_ijv, set_ijv)

    def get_labels(self):
        """Get a set of labels matrices consisting of non-overlapping labels
        In IJV format, a single pixel might have multiple labels. If you
        want to use a labels matrix, you have an ambiguous situation and the
        resolution is to process separate labels matrices consisting of
        non-overlapping labels.
        returns a list of label matrixes and the indexes in each
        """
        dense, indices = self.__segmented.get_dense()

        if dense.shape[3] == 1:
            return [(dense[i, 0, 0, 0], indices[i]) for i in range(dense.shape[0])]

        return [(dense[i, 0, 0], indices[i]) for i in range(dense.shape[0])]

    def has_unedited_segmented(self):
        """Return true if there is an unedited segmented matrix."""
        return self.__unedited_segmented is not None

    @property
    def unedited_segmented(self):
        """Get the segmentation of the image into objects, including junk that
        should be ignored: a matrix of object numbers.
        The default, if no unedited matrix is available, is the
        segmented labeling.
        """
        if self.__unedited_segmented is not None:
            return self.__segmentation_to_labels(self.__unedited_segmented)

        return self.segmented

    @unedited_segmented.setter
    def unedited_segmented(self, labels):
        self.__unedited_segmented = self.__labels_to_segmentation(labels)

    def has_small_removed_segmented(self):
        """Return true if there is a junk object matrix."""
        return self.__small_removed_segmented is not None

    @property
    def small_removed_segmented(self):
        """Get the matrix of segmented objects with the small objects removed
        This should be the same as the unedited_segmented label matrix with
        the small objects removed, but objects touching the sides of the image
        or the image mask still present.
        """
        if self.__small_removed_segmented is not None:
            return self.__segmentation_to_labels(self.__small_removed_segmented)

        return self.unedited_segmented

    @small_removed_segmented.setter
    def small_removed_segmented(self, labels):
        self.__small_removed_segmented = self.__labels_to_segmentation(labels)

    @property
    def parent_image(self):
        """The image that was analyzed to yield the objects.
        The image is an instance of BioImImage which means it has the mask
        """
        return self.__parent_image

    @parent_image.setter
    def parent_image(self, parent_image):
        self.__parent_image = parent_image
        for segmentation in (
            self.__segmented,
            self.__small_removed_segmented,
            self.__unedited_segmented,
        ):
            if segmentation is not None and not segmentation.has_shape():
                shape = (
                    1,
                    1,
                    1,
                    parent_image.pixel_data.shape[0],
                    parent_image.pixel_data.shape[1],
                )
                segmentation.shape = shape

    @property
    def has_parent_image(self):
        """True if the objects were derived from a parent image"""
        return self.__parent_image is not None

    def make_ijv_outlines(self, colors):
        """Make ijv-style color outlines
        Make outlines, coloring each object differently to distinguish between
        objects that might overlap.
        colors: a N x 3 color map to be used to color the outlines
        """
        #
        # Get planes of non-overlapping objects. The idea here is to use
        # the most similar colors in the color space for objects that
        # don't overlap.
        #
        all_labels = [(outline(label), indexes) for label, indexes in self.get_labels()]
        image = np.zeros(list(all_labels[0][0].shape) + [3], np.float32)
        #
        # Find out how many unique labels in each
        #
        counts = [np.sum(np.unique(l) != 0) for l, _ in all_labels]
        if len(counts) == 1 and counts[0] == 0:
            return image

        if len(colors) < len(all_labels):
            # Have to color 2 planes using the same color!
            # There's some chance that overlapping objects will get
            # the same color. Give me more colors to work with please.
            colors = np.vstack([colors] * (1 + len(all_labels) // len(colors)))
        r = RandomState()
        alpha = np.zeros(all_labels[0][0].shape, np.float32)
        order = np.lexsort([counts])
        label_colors = []
        for idx, i in enumerate(order):
            max_available = len(colors) / (len(all_labels) - idx)
            ncolors = min(counts[i], max_available)
            my_colors = colors[:ncolors]
            colors = colors[ncolors:]
            my_colors = my_colors[r.permutation(np.arange(ncolors))]
            my_labels, indexes = all_labels[i]
            color_idx = np.zeros(np.max(indexes) + 1, int)
            color_idx[indexes] = np.arange(len(indexes)) % ncolors
            image[my_labels != 0, :] += my_colors[color_idx[my_labels[my_labels != 0]], :]
            alpha[my_labels != 0] += 1
        image[alpha > 0, :] /= alpha[alpha > 0][:, np.newaxis]
        return image

    # def relate_children(self, children):
    #     """Relate the object numbers in one label to the object numbers in another
    #     children - another "objects" instance: the labels of children within
    #                the parent which is "self"
    #     Returns two 1-d arrays. The first gives the number of children within
    #     each parent. The second gives the mapping of each child to its parent's
    #     object number.
    #     """
    #     if self.volumetric:
    #         histogram = self.histogram_from_labels(self.segmented, children.segmented)
    #     else:
    #         histogram = self.histogram_from_ijv(self.ijv, children.ijv)

    #     return self.relate_histogram(histogram)

    # def relate_labels(self, parent_labels, child_labels):
    #     """relate the object numbers in one label to those in another
    #     parent_labels - 2d label matrix of parent labels
    #     child_labels - 2d label matrix of child labels
    #     Returns two 1-d arrays. The first gives the number of children within
    #     each parent. The second gives the mapping of each child to its parent's
    #     object number.
    #     """
    #     histogram = self.histogram_from_labels(parent_labels, child_labels)
    #     return self.relate_histogram(histogram)

    # @staticmethod
    # def relate_histogram(histogram):
    #     """Return child counts and parents of children given a histogram
    #     histogram - histogram from histogram_from_ijv or histogram_from_labels
    #     """
    #     parent_count = histogram.shape[0] - 1

    #     parents_of_children = np.argmax(histogram, axis=0)
    #     #
    #     # Create a histogram of # of children per parent
    #     children_per_parent = np.histogram(
    #         parents_of_children[1:], np.arange(parent_count + 2)
    #     )[0][1:]

    #     #
    #     # Make sure to remove the background elements at index 0
    #     #
    #     return children_per_parent, parents_of_children[1:]

    @staticmethod
    def histogram_from_labels(parent_labels, child_labels):
        """Find per pixel overlap of parent labels and child labels
        parent_labels - the parents which contain the children
        child_labels - the children to be mapped to a parent
        Returns a 2d array of overlap between each parent and child.
        Note that the first row and column are empty, as these
        correspond to parent and child labels of 0.
        """
        parent_count = np.max(parent_labels)
        child_count = np.max(child_labels)
        #
        # If the labels are different shapes, crop to shared shape.
        #
        common_shape = np.minimum(parent_labels.shape, child_labels.shape)

        if parent_labels.ndim == 3:
            parent_labels = parent_labels[0 : common_shape[0], 0 : common_shape[1], 0 : common_shape[2]]
            child_labels = child_labels[0 : common_shape[0], 0 : common_shape[1], 0 : common_shape[2]]
        else:
            parent_labels = parent_labels[0 : common_shape[0], 0 : common_shape[1]]
            child_labels = child_labels[0 : common_shape[0], 0 : common_shape[1]]

        #
        # Only look at points that are labeled in parent and child
        #
        not_zero = (parent_labels > 0) & (child_labels > 0)
        not_zero_count = np.sum(not_zero)

        #
        # each row (axis = 0) is a parent
        # each column (axis = 1) is a child
        #
        return scipy.sparse.coo_matrix(
            (
                np.ones((not_zero_count,)),
                (parent_labels[not_zero], child_labels[not_zero]),
            ),
            shape=(parent_count + 1, child_count + 1),
        ).toarray()

    @staticmethod
    def histogram_from_ijv(parent_ijv, child_ijv):
        """Find per pixel overlap of parent labels and child labels,
        stored in ijv format.
        parent_ijv - the parents which contain the children
        child_ijv - the children to be mapped to a parent
        Returns a 2d array of overlap between each parent and child.
        Note that the first row and column are empty, as these
        correspond to parent and child labels of 0.
        """
        parent_count = 0 if (parent_ijv.shape[0] == 0) else np.max(parent_ijv[:, 2])
        child_count = 0 if (child_ijv.shape[0] == 0) else np.max(child_ijv[:, 2])

        if parent_count == 0 or child_count == 0:
            return np.zeros((parent_count + 1, child_count + 1), int)

        dim_i = max(np.max(parent_ijv[:, 0]), np.max(child_ijv[:, 0])) + 1
        dim_j = max(np.max(parent_ijv[:, 1]), np.max(child_ijv[:, 1])) + 1
        parent_linear_ij = parent_ijv[:, 0] + dim_i * parent_ijv[:, 1].astype(np.uint64)
        child_linear_ij = child_ijv[:, 0] + dim_i * child_ijv[:, 1].astype(np.uint64)

        parent_matrix = scipy.sparse.coo_matrix(
            (np.ones((parent_ijv.shape[0],)), (parent_ijv[:, 2], parent_linear_ij)),
            shape=(parent_count + 1, dim_i * dim_j),
        )
        child_matrix = scipy.sparse.coo_matrix(
            (np.ones((child_ijv.shape[0],)), (child_linear_ij, child_ijv[:, 2])),
            shape=(dim_i * dim_j, child_count + 1),
        )
        # I surely do not understand the sparse code.  Converting both
        # arrays to csc gives the best peformance... Why not p.csr and
        # c.csc?
        return (parent_matrix.tocsc() * child_matrix.tocsc()).toarray()

    def fn_of_label_and_index(self, func):
        """Call a function taking a label matrix with the segmented labels
        function - should have signature like
                   labels - label matrix
                   index  - sequence of label indices documenting which
                            label indices are of interest
        """
        return func(self.segmented, self.indices)

    def fn_of_ones_label_and_index(self, func):
        """Call a function taking an image, a label matrix and an index with an image of all ones
        function - should have signature like
                   image  - image with same dimensions as labels
                   labels - label matrix
                   index  - sequence of label indices documenting which
                            label indices are of interest
        Pass this function an "image" of all ones, for instance to compute
        a center or an area
        """
        return func(np.ones(self.segmented.shape), self.segmented, self.indices)

    def center_of_mass(self):
        labels = self.segmented

        index = np.unique(labels)

        if index[0] == 0:
            index = index[1:]

        return np.array(scipy.ndimage.center_of_mass(np.ones_like(labels), labels, index))

    def overlapping(self):
        if not isinstance(self.__segmented, Segmentation):
            return False
        dense, indices = self.__segmented.get_dense()
        return len(dense) != 1


class Segmentation:
    """A segmentation of a space into labeled objects
    Supports overlapping objects and cacheing. Retrieval can be as a
    single plane (legacy), as multiple planes and as sparse ijv.
    """

    SEGMENTED = "segmented"
    UNEDITED_SEGMENTED = "unedited segmented"
    SMALL_REMOVED_SEGMENTED = "small removed segmented"

    def __init__(self, dense=None, sparse=None, shape=None):
        """Initialize the segmentation with either a dense or sparse labeling
        dense - a 6-D labeling with the first axis allowing for alternative
                labelings of the same hyper-voxel.
        sparse - the sparse labeling as a record array with axes from
                 cellprofiler_core.utilities.hdf_dict.HDF5ObjectSet
        shape - the 5-D shape of the imaging site if sparse.
        """

        self.__dense = dense
        self.__sparse = sparse
        if shape is not None:
            self.__shape = shape
            self.__explicit_shape = True
        else:
            self.__shape = None
            self.__explicit_shape = False

        if dense is not None:
            self.__indices = [np.unique(d) for d in dense]
            self.__indices = [idx[1:] if idx[0] == 0 else idx for idx in self.__indices]

    @property
    def shape(self):
        """Get or estimate the shape of the segmentation matrix
        Order of precedence:
        Shape supplied in the constructor
        Shape of the dense representation
        maximum extent of the sparse representation + 1
        """
        if self.__shape is not None:
            return self.__shape
        if self.has_dense():
            self.__shape = self.get_dense()[0].shape[1:]
        else:
            sparse = self.sparse
            if len(sparse) == 0:
                self.__shape = (1, 1, 1, 1, 1)
            else:
                self.__shape = tuple(
                    [
                        np.max(sparse[axis]) + 2 if axis in list(sparse.dtype.fields.keys()) else 1
                        for axis in ("c", "t", "z", "y", "x")
                    ]
                )
        return self.__shape

    @shape.setter
    def shape(self, shape):
        """Set the shape of the segmentation array
        shape - the 5D shape of the array
        This fixes the shape of the 5D array for sparse representations
        """
        self.__shape = shape
        self.__explicit_shape = True

    def has_dense(self):
        return self.__dense is not None

    def has_sparse(self):
        return self.__sparse is not None

    def has_shape(self):
        if self.__explicit_shape:
            return True

        return self.has_dense()

    @property
    def sparse(self):
        """Get the sparse representation of the segmentation
        returns a Numpy record array where every row represents
        the labeling of a pixel. The dtype record names are taken from
        HDF5ObjectSet.AXIS_[X,Y,Z,C,T] and AXIS_LABELS for the object
        numbers.
        """
        if self.__sparse is not None:
            return self.__sparse

        if not self.has_dense():
            raise ValueError("Can't find object dense segmentation.")

        return self.__convert_dense_to_sparse()

    def get_dense(self):
        """Get the dense representation of the segmentation
        return the segmentation as a 6-D array and a sequence of arrays of the
        object numbers in each 5-D hyperplane of the segmentation. The first
        axis of the segmentation allows us to assign multiple labels to
        individual pixels. Given a 5-D algorithm, the code typically iterates
        over the first axis:
        for labels in self.get_dense():
            # do something
        The remaining axes are in the order, C, T, Z, Y and X
        """
        if self.__dense is not None:
            return self.__dense, self.__indices

        if not self.has_sparse():
            raise ValueError("Can't find object sparse segmentation.")

        return self.__convert_sparse_to_dense()

    def __convert_dense_to_sparse(self):
        dense, indices = self.get_dense()
        axes = list(("c", "t", "z", "y", "x"))
        axes, shape = [[a for a, s in zip(aa, self.shape) if s > 1] for aa in (axes, self.shape)]
        #
        # dense.shape[0] is the overlap-axis - it's usually 1
        # except if there are multiply-labeled pixels and overlapping
        # objects. When collecting the coords, we can discard this axis.
        #
        dense = dense.reshape([dense.shape[0]] + shape)
        coords = np.where(dense != 0)
        plane, coords = coords[0], coords[1:]
        if np.max(shape) < 2**16:
            coords_dtype = np.uint16
        else:
            coords_dtype = np.uint32
        if len(plane) > 0:
            labels = dense[tuple([plane] + list(coords))]
            max_label = np.max(indices)
            if max_label < 2**8:
                labels_dtype = np.uint8
            elif max_label < 2**16:
                labels_dtype = np.uint16
            else:
                labels_dtype = np.uint32
        else:
            labels = np.zeros(0, dense.dtype)
            labels_dtype = np.uint8
        dtype = [(axis, coords_dtype) for axis in axes]
        dtype.append(("label", labels_dtype))
        sparse = np.core.records.fromarrays(list(coords) + [labels], dtype=dtype)
        self.__sparse = sparse
        return sparse

    def __set_dense(self, dense, indices=None):
        self.__dense = dense
        if indices is not None:
            self.__indices = indices
        else:
            self.__indices = [np.unique(d) for d in dense]
            self.__indices = [idx[1:] if idx[0] == 0 else idx for idx in self.__indices]
        return dense, self.__indices

    def __convert_sparse_to_dense(self):
        sparse = self.sparse
        if len(sparse) == 0:
            return self.__set_dense(np.zeros([1] + list(self.shape), np.uint16))

        #
        # The code below assigns a "color" to each label so that no
        # two labels have the same color
        #
        positional_columns = []
        available_columns = []
        lexsort_columns = []
        for axis in ("c", "t", "z", "y", "x"):
            if axis in list(sparse.dtype.fields.keys()):
                positional_columns.append(sparse[axis])
                available_columns.append(sparse[axis])
                lexsort_columns.insert(0, sparse[axis])
            else:
                positional_columns.append(0)
        labels = sparse["label"]
        lexsort_columns.insert(0, labels)

        sort_order = np.lexsort(lexsort_columns)
        n_labels = np.max(labels)
        #
        # Find the first of a run that's different from the rest
        #
        mask = available_columns[0][sort_order[:-1]] != available_columns[0][sort_order[1:]]
        for column in available_columns[1:]:
            mask = mask | (column[sort_order[:-1]] != column[sort_order[1:]])
        breaks = np.hstack(([0], np.where(mask)[0] + 1, [len(labels)]))
        firsts = breaks[:-1]
        counts = breaks[1:] - firsts
        #
        # Eliminate the locations that are singly labeled
        #
        mask = counts > 1
        firsts = firsts[mask]
        counts = counts[mask]
        if len(counts) == 0:
            dense = np.zeros([1] + list(self.shape), labels.dtype)
            dense[tuple([0] + positional_columns)] = labels
            return self.__set_dense(dense)
        #
        # There are n * n-1 pairs for each coordinate (n = # labels)
        # n = 1 -> 0 pairs, n = 2 -> 2 pairs, n = 3 -> 6 pairs
        #
        pairs = all_pairs(np.max(counts))
        pair_counts = counts * (counts - 1)
        #
        # Create an indexer for the inputs (indexes) and for the outputs
        # (first and second of the pairs)
        #
        # Remember idx points into sort_order which points into labels
        # to get the nth label, grouped into consecutive positions.
        #
        output_indexer = Indexes(pair_counts)
        #
        # The start of the run of overlaps and the offsets
        #
        run_starts = firsts[output_indexer.rev_idx]
        offs = pairs[output_indexer.idx[0], :]
        first = labels[sort_order[run_starts + offs[:, 0]]]
        second = labels[sort_order[run_starts + offs[:, 1]]]
        #
        # And sort these so that we get consecutive lists for each
        #
        pair_sort_order = np.lexsort((second, first))
        #
        # Eliminate dupes
        #
        to_keep = np.hstack(([True], (first[1:] != first[:-1]) | (second[1:] != second[:-1])))
        to_keep = to_keep & (first != second)
        pair_idx = pair_sort_order[to_keep]
        first = first[pair_idx]
        second = second[pair_idx]
        #
        # Bincount each label so we can find the ones that have the
        # most overlap. See cpmorphology.color_labels and
        # Welsh, "An upper bound for the chromatic number of a graph and
        # its application to timetabling problems", The Computer Journal, 10(1)
        # p 85 (1967)
        #
        overlap_counts = np.bincount(first.astype(np.int32))
        #
        # The index to the i'th label's stuff
        #
        indexes = np.cumsum(overlap_counts) - overlap_counts
        #
        # A vector of a current color per label. All non-overlapping
        # objects are assigned to plane 1
        #
        v_color = np.ones(n_labels + 1, int)
        v_color[0] = 0
        #
        # Clear all overlapping objects
        #
        v_color[np.unique(first)] = 0
        #
        # The processing order is from most overlapping to least
        #
        ol_labels = np.where(overlap_counts > 0)[0]
        processing_order = np.lexsort((ol_labels, overlap_counts[ol_labels]))

        for index in ol_labels[processing_order]:
            neighbors = second[indexes[index] : indexes[index] + overlap_counts[index]]
            colors = np.unique(v_color[neighbors])
            if colors[0] == 0:
                if len(colors) == 1:
                    # all unassigned - put self in group 1
                    v_color[index] = 1
                    continue
                else:
                    # otherwise, ignore the unprocessed group and continue
                    colors = colors[1:]
            # Match a range against the colors array - the first place
            # they don't match is the first color we can use
            crange = np.arange(1, len(colors) + 1)
            misses = crange[colors != crange]
            if len(misses):
                color = misses[0]
            else:
                max_color = len(colors) + 1
                color = max_color
            v_color[index] = color
        #
        # Create the dense matrix by using the color to address the
        # 5-d hyperplane into which we place each label
        #
        dense = np.zeros([np.max(v_color)] + list(self.shape), labels.dtype)
        slices = tuple([v_color[labels] - 1] + positional_columns)
        dense[slices] = labels
        indices = [np.where(v_color == i)[0] for i in range(1, dense.shape[0] + 1)]

        return self.__set_dense(dense, indices)


class Indexes(object):
    """The Indexes class stores indexes for manipulating subsets on behalf of a parent set

    The idea here is that you have a parent set of "things", for instance
    some pixels or objects. Each of these might have, conceptually, an N-d
    array of sub-objects where each array might have different dimensions.
    This class holds indexes that help out.

    For instance, create 300 random objects, each of which has
    an array of sub-objects of size 1x1 to 10x20. Create weights for each
    axis for the sub-objects, take the cross-product of the axis weights
    and then sum them (you'll do something more useful, I hope):

    i_count = np.random.randint(1,10, size=300)

    j_count = np.random.randint(1,20, size=300)

    i_indexes = Indexes([i_count])

    j_indexes = Indexes([j_count])

    indexes = Indexes([i_count, j_count])

    i_weights = np.random.uniform(size=i_indexes.length)

    j_weights = np.random.uniform(size=j_indexes.length)

    weights = (i_weights[i_indexes.fwd_idx[indexes.rev_idx] + indexes.idx[0]] *

               j_weights[j_indexes.fwd_idx[indexes.rev_idx] + indexes.idx[1]])

    sums_of_weights = np.bincount(indexes.rev_idx, weights)
    """

    def __init__(self, counts):
        """Constructor

        counts - an NxM array of dimensions of sub-arrays
                 N is the number of dimensions of the sub-object array
                 M is the number of objects.
        """
        counts = np.atleast_2d(counts).astype(int)
        self.__counts = counts.copy()
        if np.sum(np.prod(counts, 0)) == 0:
            self.__length = 0
            self.__fwd_idx = np.zeros(counts.shape[1], int)
            self.__rev_idx = np.zeros(0, int)
            self.__idx = np.zeros((len(counts), 0), int)
            return
        cs = np.cumsum(np.prod(counts, 0))
        self.__length = cs[-1]
        self.__fwd_idx = np.hstack(([0], cs[:-1]))
        self.__rev_idx = np.zeros(self.__length, int)
        non_empty_indices = np.arange(counts.shape[1]).astype(int)[np.prod(counts, 0) > 0]
        if len(non_empty_indices) > 0:
            self.__rev_idx[self.__fwd_idx[non_empty_indices[0]]] = non_empty_indices[0]
            if len(non_empty_indices) > 1:
                distance_to_next = non_empty_indices[1:] - non_empty_indices[:-1]
                self.__rev_idx[self.__fwd_idx[non_empty_indices[1:]]] = distance_to_next
            self.__rev_idx = np.cumsum(self.__rev_idx)
            self.__idx = []
            indexes = np.arange(self.length) - self.__fwd_idx[self.__rev_idx]
            for i, count in enumerate(counts[:-1]):
                modulos = np.prod(counts[(i + 1) :, :], 0)
                self.__idx.append((indexes / modulos[self.__rev_idx]).astype(int))
                indexes = indexes % modulos[self.__rev_idx]
            self.__idx.append(indexes)
            self.__idx = np.array(self.__idx)

    @property
    def length(self):
        """The number of elements in all sub-objects

        Use this number to create an array that holds a value for each
        sub-object.
        """
        return self.__length

    @property
    def fwd_idx(self):
        """The index to the first sub object per super-object

        Use the fwd_idx as part of the address of the sub-object.
        """
        return self.__fwd_idx

    @property
    def rev_idx(self):
        """The index of the super-object per sub-object"""
        return self.__rev_idx

    @property
    def idx(self):
        """For each sub-object, its indexes relative to the super-object array

        This lets you find the axis coordinates of any place in a sub-object
        array. For instance, if you have 2-d arrays of sub-objects,
        index.idx[0],index.idx[1] gives the coordinates of each sub-object
        in its array.
        """
        return self.__idx

    @property
    def counts(self):
        """The dimensions for each object along each of the axes

        The same values are stored here as are in the counts
        passed into the constructor.
        """
        return self.__counts


def all_pairs(n):
    """Return an (n*(n - 1)) x 2 array of all non-identity pairs of n things

    n - # of things

    The array is (cleverly) ordered so that the first m * (m - 1) elements
    can be used for m < n things:

    n = 3
    [[0, 1], # n = 2
     [1, 0], # n = 2
     [0, 2],
     [1, 2],
     [2, 0],
     [2, 1]]
    """
    # Get all against all
    i, j = [x.flatten() for x in np.mgrid[0:n, 0:n]]
    # Eliminate the diagonal of i == j
    i, j = [x[i != j] for x in (i, j)]
    # Order by smallest of i or j first, then j then i for neatness
    order = np.lexsort((j, i, np.maximum(i, j)))
    return np.column_stack((i[order], j[order]))


def outline(labels):
    """Given a label matrix, return a matrix of the outlines of the labeled objects

    If a pixel is not zero and has at least one neighbor with a different
    value, then it is part of the outline.
    """

    output = np.zeros(labels.shape, labels.dtype)
    lr_different = labels[1:, :] != labels[:-1, :]
    ud_different = labels[:, 1:] != labels[:, :-1]
    d1_different = labels[1:, 1:] != labels[:-1, :-1]
    d2_different = labels[1:, :-1] != labels[:-1, 1:]
    different = np.zeros(labels.shape, bool)
    different[1:, :][lr_different] = True
    different[:-1, :][lr_different] = True
    different[:, 1:][ud_different] = True
    different[:, :-1][ud_different] = True
    different[1:, 1:][d1_different] = True
    different[:-1, :-1][d1_different] = True
    different[1:, :-1][d2_different] = True
    different[:-1, 1:][d2_different] = True
    #
    # Labels on edges need outlines
    #
    different[0, :] = True
    different[:, 0] = True
    different[-1, :] = True
    different[:, -1] = True

    output[different] = labels[different]
    return output


def downsample_labels(labels):
    """Convert a labels matrix to the smallest possible integer format"""
    labels_max = np.max(labels)
    if labels_max < 128:
        return labels.astype(np.int8)
    elif labels_max < 32768:
        return labels.astype(np.int16)
    return labels.astype(np.int32)


def crop_labels_and_image(labels, image):
    """Crop a labels matrix and an image to the lowest common size
    labels - a n x m labels matrix
    image - a 2-d or 3-d image
    Assumes that points outside of the common boundary should be masked.
    """
    min_dim1 = min(labels.shape[0], image.shape[0])
    min_dim2 = min(labels.shape[1], image.shape[1])

    if labels.ndim == 3:  # volume
        min_dim3 = min(labels.shape[2], image.shape[2])

        if image.ndim == 4:  # multichannel volume
            return (
                labels[:min_dim1, :min_dim2, :min_dim3],
                image[:min_dim1, :min_dim2, :min_dim3, :],
            )

        return (
            labels[:min_dim1, :min_dim2, :min_dim3],
            image[:min_dim1, :min_dim2, :min_dim3],
        )

    if image.ndim == 3:  # multichannel image
        return labels[:min_dim1, :min_dim2], image[:min_dim1, :min_dim2, :]

    return labels[:min_dim1, :min_dim2], image[:min_dim1, :min_dim2]


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

import os
import operator
import string
import copy
import itertools
import numpy as np
import networkx as nx
import scipy.ndimage as nd
import image_ops
from from_fits import create_image_from_fits_file
from skimage.morphology import medial_axis
from scipy import nanmean
import matplotlib.pyplot as plt


# Create 4 to 8-connected elements to use with binary hit-or-miss
struct1 = np.array([[1, 0, 0],
                    [0, 1, 1],
                    [0, 0, 0]])

struct2 = np.array([[0, 0, 1],
                    [1, 1, 0],
                    [0, 0, 0]])

# Next check the three elements which will be double counted
check1 = np.array([[1, 1, 0, 0],
                   [0, 0, 1, 1]])

check2 = np.array([[0, 0, 1, 1],
                   [1, 1, 0, 0]])

check3 = np.array([[1, 1, 0],
                   [0, 0, 1],
                   [0, 0, 1]])


def eight_con():
    return np.ones((3, 3))


def _fix_small_holes(mask_array, rel_size=0.1):
    '''
    Helper function to remove only small holes within a masked region.

    Parameters
    ----------
    mask_array : numpy.ndarray
        Array containing the masked region.

    rel_size : float, optional
        If < 1.0, sets the minimum size a hole must be relative to the area
        of the mask. Otherwise, this is the maximum number of pixels the hole
        must have to be deleted.

    Returns
    -------
    mask_array : numpy.ndarray
        Altered array.
    '''

    if rel_size <= 0.0:
        raise ValueError("rel_size must be positive.")
    elif rel_size > 1.0:
        pixel_flag = True
    else:
        pixel_flag = False

    # Find the region area
    reg_area = len(np.where(mask_array == 1)[0])

    # Label the holes
    holes = np.logical_not(mask_array).astype(float)
    lab_holes, n_holes = nd.label(holes, eight_con())

    # If no holes, return
    if n_holes == 1:
        return mask_array

    # Ignore area outside of the region.
    out_label = lab_holes[0, 0]
    # Set size to be just larger than the region. Thus it can never be
    # deleted.
    holes[np.where(lab_holes == out_label)] = reg_area + 1.

    # Sum up the regions and find holes smaller than the threshold.
    sums = nd.sum(holes, lab_holes, range(1, n_holes + 1))
    if pixel_flag:  # Use number of pixels
        delete_holes = np.where(sums < rel_size)[0]
    else:  # Use relative size of holes.
        delete_holes = np.where(sums / reg_area < rel_size)[0]

    # Return if there is nothing to delete.
    if delete_holes == []:
        return mask_array

    # Add one to take into account 0 in list if object label 1.
    delete_holes += 1
    for label in delete_holes:
        mask_array[np.where(lab_holes == label)] = 1

    return mask_array


def isolateregions(binary_array, size_threshold=0, pad_size=5,
                   fill_hole=False, rel_size=0.1, morph_smooth=False):
    '''

    Labels regions in a boolean array and returns individual arrays for each
    region. Regions below a threshold can optionlly be removed. Small holes
    may also be filled in.

    Parameters
    ----------
    binary_array : numpy.ndarray
        A binary array of regions.
    size_threshold : int, optional
        Sets the pixel size on the size of regions.
    pad_size : int, optional
        Padding to be added to the individual arrays.
    fill_hole : int, optional
        Enables hole filling.
    rel_size : float or int, optional
        If < 1.0, sets the minimum size a hole must be relative to the area
        of the mask. Otherwise, this is the maximum number of pixels the hole
        must have to be deleted.
    morph_smooth : bool, optional
        Morphologically smooth the image using a binar opening and closing.

    Returns
    -------
    output_arrays : list
        Regions separated into individual arrays.
    num : int
        Number of filaments
    corners : list
        Contains the indices where each skeleton array was taken from
        the original.

    '''

    output_arrays = []
    corners = []

    # Label skeletons
    labels, num = nd.label(binary_array, eight_con())

    # Remove skeletons which have fewer pixels than the threshold.
    if size_threshold != 0:
        sums = nd.sum(binary_array, labels, range(1, num + 1))
        remove_fils = np.where(sums <= size_threshold)[0]
        for lab in remove_fils:
            binary_array[np.where(labels == lab + 1)] = 0

        # Relabel after deleting short skeletons.
        labels, num = nd.label(binary_array, eight_con())

    # Split each skeleton into its own array.
    for n in range(1, num + 1):
        x, y = np.where(labels == n)
        # Make an array shaped to the skeletons size and padded on each edge
        # the +1 is because, e.g., range(0, 5) only has 5 elements, but the
        # indices we're using are range(0, 6)
        shapes = (x.max() - x.min() + 2 * pad_size,
                  y.max() - y.min() + 2 * pad_size)
        eachfil = np.zeros(shapes)
        eachfil[x - x.min() + pad_size, y - y.min() + pad_size] = 1
        # Fill in small holes
        if fill_hole:
            eachfil = _fix_small_holes(eachfil, rel_size=rel_size)
        if morph_smooth:
            eachfil = nd.binary_opening(eachfil, np.ones((3, 3)))
            eachfil = nd.binary_closing(eachfil, np.ones((3, 3)))
        output_arrays.append(eachfil)
        # Keep the coordinates from the original image
        lower = (x.min() - pad_size, y.min() - pad_size)
        upper = (x.max() + pad_size + 1, y.max() + pad_size + 1)
        corners.append([lower, upper])

    return output_arrays, num, corners


def shifter(l, n):
    return l[n:] + l[:n]


def distance(x, x1, y, y1):
    return np.sqrt((x - x1) ** 2.0 + (y - y1) ** 2.0)


def find_filpix(branches, labelfil, final=True):
    '''

    Identifies the types of pixels in the given skeletons. Identification is
    based on the connectivity of the pixel.

    Parameters
    ----------
    branches : list
        Contains the number of branches in each skeleton.
    labelfil : list
        Contains the arrays of each skeleton.
    final : bool, optional
        If true, corner points, intersections, and body points are all
        labeled as a body point for use when the skeletons have already
        been cleaned.

    Returns
    -------
    fila_pts : list
        All points on the body of each skeleton.
    inters : list
        All points associated with an intersection in each skeleton.
    labelfil : list
       Contains the arrays of each skeleton where all intersections
       have been removed.
    endpts_return : list
        The end points of each branch of each skeleton.
  '''

    initslices = []
    initlist = []
    shiftlist = []
    sublist = []
    endpts = []
    blockpts = []
    bodypts = []
    slices = []
    vallist = []
    shiftvallist = []
    cornerpts = []
    subvallist = []
    subslist = []
    pix = []
    filpix = []
    intertemps = []
    fila_pts = []
    inters = []
    repeat = []
    temp_group = []
    all_pts = []
    pairs = []
    endpts_return = []

    for k in range(1, branches + 1):
        x, y = np.where(labelfil == k)
        # pixel_slices = np.empty((len(x)+1,8))
        for i in range(len(x)):
            if x[i] < labelfil.shape[0] - 1 and y[i] < labelfil.shape[1] - 1:
                pix.append((x[i], y[i]))
                initslices.append(np.array([[labelfil[x[i] - 1, y[i] + 1],
                                             labelfil[x[i], y[i] + 1],
                                             labelfil[x[i] + 1, y[i] + 1]],
                                            [labelfil[x[i] - 1, y[i]], 0,
                                             labelfil[x[i] + 1, y[i]]],
                                            [labelfil[x[i] - 1, y[i] - 1],
                                             labelfil[x[i], y[i] - 1],
                                             labelfil[x[i] + 1, y[i] - 1]]]))

        filpix.append(pix)
        slices.append(initslices)
        initslices = []
        pix = []

    for i in range(len(slices)):
        for k in range(len(slices[i])):
            initlist.append([slices[i][k][0, 0],
                             slices[i][k][0, 1],
                             slices[i][k][0, 2],
                             slices[i][k][1, 2],
                             slices[i][k][2, 2],
                             slices[i][k][2, 1],
                             slices[i][k][2, 0],
                             slices[i][k][1, 0]])
        vallist.append(initlist)
        initlist = []

    for i in range(len(slices)):
        for k in range(len(slices[i])):
            shiftlist.append(shifter(vallist[i][k], 1))
        shiftvallist.append(shiftlist)
        shiftlist = []

    for k in range(len(slices)):
        for i in range(len(vallist[k])):
            for j in range(8):
                sublist.append(
                    int(vallist[k][i][j]) - int(shiftvallist[k][i][j]))
            subslist.append(sublist)
            sublist = []
        subvallist.append(subslist)
        subslist = []

    # x represents the subtracted list (step-ups) and y is the values of the
    # surrounding pixels. The categories of pixels are ENDPTS (x<=1),
    # BODYPTS (x=2,y=2),CORNERPTS (x=2,y=3),BLOCKPTS (x=3,y>=4), and
    # INTERPTS (x>=3).
    # A cornerpt is [*,0,0] (*s) associated with an intersection,
    # but their exclusion from
    #   [1,*,0] the intersection keeps eight-connectivity, they are included
    #   [0,1,0] intersections for this reason.
    # A blockpt is  [1,0,1] They are typically found in a group of four,
    # where all four
    #   [0,*,*] constitute a single intersection.
    #   [1,*,*]
    # The "final" designation is used when finding the final branch lengths.
    # At this point, blockpts and cornerpts should be eliminated.
    for k in range(branches):
        for l in range(len(filpix[k])):
            x = [j for j, y in enumerate(subvallist[k][l]) if y == k + 1]
            y = [j for j, z in enumerate(vallist[k][l]) if z == k + 1]

            if len(x) <= 1:
                endpts.append(filpix[k][l])
                endpts_return.append(filpix[k][l])
            elif len(x) == 2:
                if final:
                    bodypts.append(filpix[k][l])
                else:
                    if len(y) == 2:
                        bodypts.append(filpix[k][l])
                    elif len(y) == 3:
                        cornerpts.append(filpix[k][l])
                    elif len(y) >= 4:
                        blockpts.append(filpix[k][l])
            elif len(x) >= 3:
                intertemps.append(filpix[k][l])
        endpts = list(set(endpts))
        bodypts = list(set(bodypts))
        dups = set(endpts) & set(bodypts)
        if len(dups) > 0:
            for i in dups:
                bodypts.remove(i)
        # Cornerpts without a partner diagonally attached can be included as a
        # bodypt.
        if len(cornerpts) > 0:
            deleted_cornerpts = []
            for i, j in zip(cornerpts, cornerpts):
                if i != j:
                    if distance(i[0], j[0], i[1], j[1]) == np.sqrt(2.0):
                        proximity = [(i[0], i[1] - 1),
                                     (i[0], i[1] + 1),
                                     (i[0] - 1, i[1]),
                                     (i[0] + 1, i[1]),
                                     (i[0] - 1, i[1] + 1),
                                     (i[0] + 1, i[1] + 1),
                                     (i[0] - 1, i[1] - 1),
                                     (i[0] + 1, i[1] - 1)]
                        match = set(intertemps) & set(proximity)
                        if len(match) == 1:
                            pairs.append([i, j])
                            deleted_cornerpts.append(i)
                            deleted_cornerpts.append(j)
            cornerpts = list(set(cornerpts).difference(set(deleted_cornerpts)))

        if len(cornerpts) > 0:
            for l in cornerpts:
                proximity = [(l[0], l[1] - 1),
                             (l[0], l[1] + 1),
                             (l[0] - 1, l[1]),
                             (l[0] + 1, l[1]),
                             (l[0] - 1, l[1] + 1),
                             (l[0] + 1, l[1] + 1),
                             (l[0] - 1, l[1] - 1),
                             (l[0] + 1, l[1] - 1)]
                match = set(intertemps) & set(proximity)
                if len(match) == 1:
                    intertemps.append(l)
                    fila_pts.append(endpts + bodypts)
                else:
                    fila_pts.append(endpts + bodypts + [l])
                    # cornerpts.remove(l)
        else:
            fila_pts.append(endpts + bodypts)

        # Reset lists
        cornerpts = []
        endpts = []
        bodypts = []

        if len(pairs) > 0:
            for i in range(len(pairs)):
                for j in pairs[i]:
                    all_pts.append(j)
        if len(blockpts) > 0:
            for i in blockpts:
                all_pts.append(i)
        if len(intertemps) > 0:
            for i in intertemps:
                all_pts.append(i)
        # Pairs of cornerpts, blockpts, and interpts are combined into an
        # array. If there is eight connectivity between them, they are labelled
        # as a single intersection.
        arr = np.zeros((labelfil.shape))
        for z in all_pts:
            labelfil[z[0], z[1]] = 0
            arr[z[0], z[1]] = 1
        lab, nums = nd.label(arr, eight_con())
        for k in range(1, nums + 1):
            objs_pix = np.where(lab == k)
            for l in range(len(objs_pix[0])):
                temp_group.append((objs_pix[0][l], objs_pix[1][l]))
            inters.append(temp_group)
            temp_group = []
    for i in range(len(inters) - 1):
        if inters[i] == inters[i + 1]:
            repeat.append(inters[i])
    for i in repeat:
        inters.remove(i)

    return fila_pts, inters, labelfil, endpts_return


def pix_identify(isolatefilarr, num):
    '''
    This function is essentially a wrapper on find_filpix. It returns the
    outputs of find_filpix in the form that are used during the analysis.

    Parameters
    ----------
    isolatefilarr : list
        Contains individual arrays of each skeleton.
    num  : int
        The number of skeletons.

    Returns
    -------
    interpts : list
        Contains lists of all intersections points in each skeleton.
    hubs : list
        Contains the number of intersections in each filament. This is
        useful for identifying those with no intersections as their analysis
        is straight-forward.
    ends : list
        Contains the positions of all end points in each skeleton.
    filbranches : list
        Contains the number of branches in each skeleton.
    labelisofil : list
        Contains individual arrays for each skeleton where the
        branches are labeled and the intersections have been removed.
    '''

    interpts = []
    hubs = []
    ends = []
    filbranches = []
    labelisofil = []

    for n in range(num):
        funcreturn = find_filpix(1, isolatefilarr[n], final=False)
        interpts.append(funcreturn[1])
        hubs.append(len(funcreturn[1]))
        isolatefilarr.pop(n)
        isolatefilarr.insert(n, funcreturn[2])
        ends.append(funcreturn[3])

        label_branch, num_branch = nd.label(isolatefilarr[n], eight_con())
        filbranches.append(num_branch)
        labelisofil.append(label_branch)

    return interpts, hubs, ends, filbranches, labelisofil


def skeleton_length(skeleton):
    '''
    Length finding via morphological operators. We use the differences in
    connectivity between 4 and 8-connected to split regions. Connections
    between 4 and 8-connected regions are found using a series of hit-miss
    operators.

    The inputted skeleton MUST have no intersections otherwise the returned
    length will not be correct!

    Parameters
    ----------
    skeleton : numpy.ndarray
        Array containing the skeleton.

    Returns
    -------
    length : float
        Length of the skeleton.

    '''

    # 4-connected labels
    four_labels = nd.label(skeleton)[0]

    four_sizes = nd.sum(skeleton, four_labels, range(np.max(four_labels) + 1))

    # Lengths is the number of pixels minus number of objects with more
    # than 1 pixel.
    four_length = np.sum(
        four_sizes[four_sizes > 1]) - len(four_sizes[four_sizes > 1])

    # Find pixels which a 4-connected and subtract them off the skeleton

    four_objects = np.where(four_sizes > 1)[0]

    skel_copy = copy.copy(skeleton)
    for val in four_objects:
        skel_copy[np.where(four_labels == val)] = 0

    # Remaining pixels are only 8-connected
    # Lengths is same as before, multiplied by sqrt(2)

    eight_labels = nd.label(skel_copy, eight_con())[0]

    eight_sizes = nd.sum(
        skel_copy, eight_labels, range(np.max(eight_labels) + 1))

    eight_length = (
                       np.sum(eight_sizes) - np.max(eight_labels)) * np.sqrt(2)

    # If there are no 4-connected pixels, we don't need the hit-miss portion.
    if four_length == 0.0:
        conn_length = 0.0

    else:

        store = np.zeros(skeleton.shape)

        # Loop through the 4 rotations of the structuring elements
        for k in range(0, 4):
            hm1 = nd.binary_hit_or_miss(
                skeleton, structure1=np.rot90(struct1, k=k))
            store += hm1

            hm2 = nd.binary_hit_or_miss(
                skeleton, structure1=np.rot90(struct2, k=k))
            store += hm2

            hm_check3 = nd.binary_hit_or_miss(
                skeleton, structure1=np.rot90(check3, k=k))
            store -= hm_check3

            if k <= 1:
                hm_check1 = nd.binary_hit_or_miss(
                    skeleton, structure1=np.rot90(check1, k=k))
                store -= hm_check1

                hm_check2 = nd.binary_hit_or_miss(
                    skeleton, structure1=np.rot90(check2, k=k))
                store -= hm_check2

        conn_length = np.sqrt(2) * \
                      np.sum(np.sum(store, axis=1), axis=0)  # hits

    return conn_length + eight_length + four_length


def init_lengths(labelisofil, filbranches, array_offsets, img):
    '''

    This is a wrapper on fil_length for running on the branches of the
    skeletons.

    Parameters
    ----------

    labelisofil : list
        Contains individual arrays for each skeleton where the
        branches are labeled and the intersections have been removed.

    filbranches : list
        Contains the number of branches in each skeleton.

    array_offsets : List
        The indices of where each filament array fits in the
        original image.

    img : numpy.ndarray
        Original image.

    Returns
    -------

    branch_properties: dict
        Contains the lengths and intensities of the branches.
        Keys are *length* and *intensity*.

    '''
    num = len(labelisofil)

    # Initialize Lists
    lengths = []
    av_branch_intensity = []

    for n in range(num):
        leng = []
        av_intensity = []

        label_copy = copy.copy(labelisofil[n])
        objects = nd.find_objects(label_copy)
        for obj in objects:
            # Scale the branch array to the branch size
            branch_array = label_copy[obj]

            # Find the skeleton points and set those to 1
            branch_pts = np.where(branch_array > 0)
            branch_array[branch_pts] = 1

            # Now find the length on the branch
            branch_length = skeleton_length(branch_array)
            if branch_length == 0.0:
                # For use in longest path algorithm, will be set to zero for
                # final analysis
                branch_length = 0.5

            leng.append(branch_length)

            # Now let's find the average intensity along each branch
            # Get the offsets from the original array and
            # add on the offset the branch array introduces.
            x_offset = obj[0].start + array_offsets[n][0][0]
            y_offset = obj[1].start + array_offsets[n][0][1]
            av_intensity.append(
                nanmean([img[x + x_offset, y + y_offset]
                         for x, y in zip(*branch_pts)
                         if np.isfinite(img[x + x_offset, y + y_offset]) and
                         not img[x + x_offset, y + y_offset] < 0.0]))

        lengths.append(leng)
        av_branch_intensity.append(av_intensity)

        branch_properties = {
            "length": lengths, "intensity": av_branch_intensity}

    return branch_properties


def product_gen(n):
    for r in itertools.count(1):
        for i in itertools.product(n, repeat=r):
            yield "".join(i)


def pre_graph(labelisofil, branch_properties, interpts, ends):
    '''

    This function converts the skeletons into a graph object compatible with
    networkx. The graphs have nodes corresponding to end and
    intersection points and edges defining the connectivity as the branches
    with the weights set to the branch length.

    Parameters
    ----------

    labelisofil : list
        Contains individual arrays for each skeleton where the
        branches are labeled and the intersections have been removed.

    branch_properties : dict
        Contains the lengths and intensities of all branches.

    interpts : list
        Contains the pixels which belong to each intersection.

    ends : list
        Contains the end pixels for each skeleton.

    Returns
    -------

    end_nodes : list
        Contains the nodes corresponding to end points.

    inter_nodes : list
        Contains the nodes corresponding to intersection points.

    edge_list : list
        Contains the connectivity information for the graphs.

    nodes : list
        A complete list of all of the nodes. The other nodes lists have
        been separated as they are labeled differently.

    '''

    num = len(labelisofil)

    end_nodes = []
    inter_nodes = []
    nodes = []
    edge_list = []

    def path_weighting(idx, length, intensity, w=0.5):
        '''

        Relative weighting for the shortest path algorithm using the branch
        lengths and the average intensity along the branch.

        '''
        if w > 1.0 or w < 0.0:
            raise ValueError(
                "Relative weighting w must be between 0.0 and 1.0.")
        return (1 - w) * (length[idx] / np.sum(length)) + \
               w * (intensity[idx] / np.sum(intensity))

    lengths = branch_properties["length"]
    branch_intensity = branch_properties["intensity"]

    for n in range(num):
        inter_nodes_temp = []
        # Create end_nodes, which contains lengths, and nodes, which we will
        # later add in the intersections
        end_nodes.append([(labelisofil[n][i[0], i[1]],
                           path_weighting(int(labelisofil[n][i[0], i[1]] - 1),
                                          lengths[n],
                                          branch_intensity[n]),
                           lengths[n][int(labelisofil[n][i[0], i[1]] - 1)],
                           branch_intensity[n][int(labelisofil[n][i[0], i[1]] - 1)])
                          for i in ends[n]])
        nodes.append([labelisofil[n][i[0], i[1]] for i in ends[n]])

        # Intersection nodes are given by the intersections points of the filament.
        # They are labeled alphabetically (if len(interpts[n])>26,
        # subsequent labels are AA,AB,...).
        # The branch labels attached to each intersection are included for future
        # use.
        for intersec in interpts[n]:
            uniqs = []
            for i in intersec:  # Intersections can contain multiple pixels
                int_arr = np.array([[labelisofil[n][i[0] - 1, i[1] + 1],
                                     labelisofil[n][i[0], i[1] + 1],
                                     labelisofil[n][i[0] + 1, i[1] + 1]],
                                    [labelisofil[n][i[0] - 1, i[1]], 0,
                                     labelisofil[n][i[0] + 1, i[1]]],
                                    [labelisofil[n][i[0] - 1, i[1] - 1],
                                     labelisofil[n][i[0], i[1] - 1],
                                     labelisofil[n][i[0] + 1, i[1] - 1]]]).astype(int)
                for x in np.unique(int_arr[np.nonzero(int_arr)]):
                    uniqs.append((x,
                                  path_weighting(x - 1, lengths[n],
                                                 branch_intensity[n]),
                                  lengths[n][x - 1],
                                  branch_intensity[n][x - 1]))
            # Intersections with multiple pixels can give the same branches.
            # Get rid of duplicates
            uniqs = list(set(uniqs))
            inter_nodes_temp.append(uniqs)

        # Add the intersection labels. Also append those to nodes
        inter_nodes.append(
            zip(product_gen(string.ascii_uppercase), inter_nodes_temp))
        for alpha, node in zip(product_gen(string.ascii_uppercase),
                               inter_nodes_temp):
            nodes[n].append(alpha)
        # Edges are created from the information contained in the nodes.
        edge_list_temp = []
        for i, inters in enumerate(inter_nodes[n]):
            end_match = list(set(inters[1]) & set(end_nodes[n]))
            for k in end_match:
                edge_list_temp.append((inters[0], k[0], k))

            for j, inters_2 in enumerate(inter_nodes[n]):
                if i != j:
                    match = list(set(inters[1]) & set(inters_2[1]))
                    new_edge = None
                    if len(match) == 1:
                        new_edge = (inters[0], inters_2[0], match[0])
                    elif len(match) > 1:
                        multi = [match[l][1] for l in range(len(match))]
                        keep = multi.index(min(multi))
                        new_edge = (inters[0], inters_2[0], match[keep])
                    if new_edge is not None:
                        if not (new_edge[1], new_edge[0], new_edge[2]) in edge_list_temp \
                                and new_edge not in edge_list_temp:
                            edge_list_temp.append(new_edge)

        # Remove duplicated edges between intersections

        edge_list.append(edge_list_temp)

    return edge_list, nodes


def try_mkdir(name):
    '''
    Checks if a folder exists, and makes it if it doesn't
    '''

    if not os.path.isdir(os.path.join(os.getcwd(), name)):
        os.mkdir(os.path.join(os.getcwd(), name))


def longest_path(edge_list, nodes, verbose=False,
                 skeleton_arrays=None, save_png=False, save_name=None):
    '''
    Takes the output of pre_graph and runs the shortest path algorithm.

    Parameters
    ----------

    edge_list : list
        Contains the connectivity information for the graphs.

    nodes : list
        A complete list of all of the nodes. The other nodes lists have
        been separated as they are labeled differently.

    verbose : bool, optional
        If True, enables the plotting of the graph.

    skeleton_arrays : list, optional
        List of the skeleton arrays. Required when verbose=True.

    save_png : bool, optional
        Saves the plot made in verbose mode. Disabled by default.

    save_name : str, optional
        For use when ``save_png`` is enabled.
        **MUST be specified when ``save_png`` is enabled.**

    Returns
    -------

    max_path : list
        Contains the paths corresponding to the longest lengths for
        each skeleton.

    extremum : list
        Contains the starting and ending points of max_path

    '''
    num = len(nodes)

    # Initialize lists
    max_path = []
    extremum = []
    graphs = []

    for n in range(num):
        G = nx.Graph()
        G.add_nodes_from(nodes[n])
        for i in edge_list[n]:
            G.add_edge(i[0], i[1], weight=i[2][1])
        paths = nx.shortest_path_length(G, weight='weight')
        values = []
        node_extrema = []
        for i in paths.iterkeys():
            j = max(paths[i].iteritems(), key=operator.itemgetter(1))
            node_extrema.append((j[0], i))
            values.append(j[1])
        start, finish = node_extrema[values.index(max(values))]
        extremum.append([start, finish])
        max_path.append(nx.shortest_path(G, start, finish))
        graphs.append(G)

        if verbose or save_png:
            if not skeleton_arrays:
                Warning("Must input skeleton arrays if verbose or save_png is"
                        " enabled. No plots will be created.")
            elif save_png and save_name is None:
                Warning("Must give a save_name when save_png is enabled. No"
                        " plots will be created.")
            else:
                # Check if skeleton_arrays is a list
                assert isinstance(skeleton_arrays, list)
                import matplotlib.pyplot as p
                if verbose:
                    print "Filament: %s / %s" % (n + 1, num)
                p.subplot(1, 2, 1)
                p.imshow(skeleton_arrays[n], interpolation="nearest",
                         origin="lower")

                p.subplot(1, 2, 2)
                elist = [(u, v) for (u, v, d) in G.edges(data=True)]
                pos = nx.spring_layout(G)
                nx.draw_networkx_nodes(G, pos, node_size=200)
                nx.draw_networkx_edges(G, pos, edgelist=elist, width=2)
                nx.draw_networkx_labels(
                    G, pos, font_size=10, font_family='sans-serif')
                p.axis('off')

                if save_png:
                    try_mkdir(save_name)
                    p.savefig(os.path.join(save_name,
                                           save_name + "_longest_path_" + str(n) + ".png"))
                if verbose:
                    p.show()
                p.clf()

    return max_path, extremum, graphs


def prune_graph(G, nodes, edge_list, max_path, labelisofil, branch_properties,
                length_thresh, relintens_thresh=0.2):
    '''
    Function to remove unnecessary branches, while maintaining connectivity
    in the graph. Also updates edge_list, nodes, branch_lengths and
    filbranches.

    Parameters
    ----------
    G : list
        Contains the networkx Graph objects.
    nodes : list
        A complete list of all of the nodes. The other nodes lists have
        been separated as they are labeled differently.
    edge_list : list
        Contains the connectivity information for the graphs.
    max_path : list
        Contains the paths corresponding to the longest lengths for
        each skeleton.
    labelisofil : list
        Contains individual arrays for each skeleton where the
        branches are labeled and the intersections have been removed.
    branch_properties : dict
        Contains the lengths and intensities of all branches.
    length_thresh : int or float
        Minimum length a branch must be to be kept. Can be overridden if the
        branch is bright relative to the entire skeleton.
    relintens_thresh : float between 0 and 1, optional.
        Threshold for how bright the branch must be relative to the entire
        skeleton. Can be overridden by length.

    Returns
    -------
    labelisofil : list
        Updated from input.
    edge_list : list
        Updated from input.
    nodes : list
        Updated from input.
    branch_properties : dict
        Updated from input.
    '''

    num = len(labelisofil)

    for n in range(num):
        degree = G[n].degree()
        single_connect = [key for key in degree.keys() if degree[key] == 1]

        delete_candidate = list(
            (set(nodes[n]) - set(max_path[n])) & set(single_connect))

        if not delete_candidate:  # Nothing to delete!
            continue

        edge_candidates = [edge for edge in edge_list[n] if edge[
            0] in delete_candidate or edge[1] in delete_candidate]
        intensities = [edge[2][3] for edge in edge_list[n]]
        for edge in edge_candidates:
            # In the odd case where a loop meets at the same intersection,
            # ensure that edge is kept.
            if isinstance(edge[0], str) & isinstance(edge[1], str):
                continue
            # If its too short and relatively not as intense, delete it
            length = edge[2][2]
            av_intensity = edge[2][3]
            if length < length_thresh \
                    and (av_intensity / np.sum(intensities)) < relintens_thresh:
                edge_pts = np.where(labelisofil[n] == edge[2][0])
                labelisofil[n][edge_pts] = 0
                edge_list[n].remove(edge)
                nodes[n].remove(edge[1])
                branch_properties["length"][n].remove(length)
                branch_properties["intensity"][n].remove(av_intensity)
                branch_properties["number"][n] -= 1

    return labelisofil, edge_list, nodes, branch_properties


def extremum_pts(labelisofil, extremum, ends):
    '''
    This function returns the the farthest extents of each filament. This
    is useful for determining how well the shortest path algorithm has worked.

    Parameters
    ----------
    labelisofil : list
        Contains individual arrays for each skeleton.
    extremum : list
       Contains the extents as determined by the shortest
       path algorithm.
    ends : list
        Contains the positions of each end point in eahch filament.

    Returns
    -------
    extrem_pts : list
        Contains the indices of the extremum points.
    '''

    num = len(labelisofil)
    extrem_pts = []

    for n in range(num):
        per_fil = []
        for i, j in ends[n]:
            if labelisofil[n][i, j] == extremum[n][0] or labelisofil[n][i, j] == extremum[n][1]:
                per_fil.append([i, j])
        extrem_pts.append(per_fil)

    return extrem_pts


def main_length(max_path, edge_list, labelisofil, interpts, branch_lengths,
                img_scale, verbose=False, save_png=False, save_name=None):
    '''
    Wraps previous functionality together for all of the skeletons in the
    image. To find the overall length for each skeleton, intersections are
    added back in, and any extraneous pixels they bring with them are deleted.

    Parameters
    ----------
    max_path : list
        Contains the paths corresponding to the longest lengths for
        each skeleton.
    edge_list : list
        Contains the connectivity information for the graphs.
    labelisofil : list
        Contains individual arrays for each skeleton where the
        branches are labeled and the intersections have been removed.
    interpts : list
        Contains the pixels which belong to each intersection.
    branch_lengths : list
        Lengths of individual branches in each skeleton.
    img_scale : float
        Conversion from pixel to physical units.
    verbose : bool, optional
        Returns plots of the longest path skeletons.
    save_png : bool, optional
        Saves the plot made in verbose mode. Disabled by default.
    save_name : str, optional
        For use when ``save_png`` is enabled.
        **MUST be specified when ``save_png`` is enabled.**

    Returns
    -------
    main_lengths : list
        Lengths of the skeletons.
    longpath_arrays : list
        Arrays of the longest paths in the skeletons.
    '''

    main_lengths = []
    longpath_arrays = []

    for num, (path, edges, inters, skel_arr, lengths) in \
            enumerate(zip(max_path, edge_list, interpts, labelisofil,
                          branch_lengths)):

        if len(path) == 1:
            main_lengths.append(lengths[0] * img_scale)
            skeleton = skel_arr  # for viewing purposes when verbose
        else:
            skeleton = np.zeros(skel_arr.shape)

            # Add edges along longest path
            good_edge_list = [(path[i], path[i + 1])
                              for i in range(len(path) - 1)]
            # Find the branches along the longest path.
            for i in good_edge_list:
                for j in edges:
                    if (i[0] == j[0] and i[1] == j[1]) or \
                            (i[0] == j[1] and i[1] == j[0]):
                        label = j[2][0]
                        skeleton[np.where(skel_arr == label)] = 1

            # Add intersections along longest path
            intersec_pts = []
            for label in path:
                try:
                    label = int(label)
                except ValueError:
                    pass
                if not isinstance(label, int):
                    k = 1
                    while zip(product_gen(string.ascii_uppercase),
                              [1] * k)[-1][0] != label:
                        k += 1
                    intersec_pts.extend(inters[k - 1])
                    skeleton[zip(*inters[k - 1])] = 2

            # Remove unnecessary pixels
            count = 0
            while True:
                for pt in intersec_pts:
                    # If we have already eliminated the point, continue
                    if skeleton[pt] == 0:
                        continue
                    skeleton[pt] = 0
                    lab_try, n = nd.label(skeleton, eight_con())
                    if n > 1:
                        skeleton[pt] = 1
                    else:
                        count += 1
                if count == 0:
                    break
                count = 0

            main_lengths.append(skeleton_length(skeleton) * img_scale)

        longpath_arrays.append(skeleton.astype(int))

        if verbose or save_png:
            if save_png and save_name is None:
                Warning("Must give a save_name when save_png is enabled. No"
                        " plots will be created.")
            import matplotlib.pyplot as p
            if verbose:
                print "Filament: %s / %s" % (num + 1, len(labelisofil))

            p.subplot(121)
            p.imshow(skeleton, origin='lower', interpolation="nearest")
            p.subplot(122)
            p.imshow(labelisofil[num],  origin='lower',
                     interpolation="nearest")

            if save_png:
                try_mkdir(save_name)
                p.savefig(os.path.join(save_name,
                                       save_name + "_main_length_" + str(num) + ".png"))
            if verbose:
                p.show()
            p.clf()

    return main_lengths, longpath_arrays


def find_extran(branches, labelfil):
    '''
    Identify pixels that are not necessary to keep the connectivity of the
    skeleton. It uses the same labeling process as find_filpix. Extraneous
    pixels tend to be those from former intersections, whose attached branch
    was eliminated in the cleaning process.

    Parameters
    ----------
    branches : list
        Contains the number of branches in each skeleton.
    labelfil : list
        Contains arrays of the labeled versions of each skeleton.

    Returns
    -------
    labelfil : list
       Contains the updated labeled arrays with extraneous pieces
       removed.
    '''

    initslices = []
    initlist = []
    shiftlist = []
    sublist = []
    extran = []
    slices = []
    vallist = []
    shiftvallist = []
    subvallist = []
    subslist = []
    pix = []
    filpix = []

    for k in range(1, branches + 1):
        x, y = np.where(labelfil == k)
        for i in range(len(x)):
            if x[i] < labelfil.shape[0] - 1 and y[i] < labelfil.shape[1] - 1:
                pix.append((x[i], y[i]))
                initslices.append(np.array([[labelfil[x[i] - 1, y[i] + 1],
                                             labelfil[x[i], y[i] + 1],
                                             labelfil[x[i] + 1, y[i] + 1]],
                                            [labelfil[x[i] - 1, y[i]], 0,
                                             labelfil[x[i] + 1, y[i]]],
                                            [labelfil[x[i] - 1, y[i] - 1],
                                             labelfil[x[i], y[i] - 1],
                                             labelfil[x[i] + 1, y[i] - 1]]]))

        filpix.append(pix)
        slices.append(initslices)
        initslices = []
        pix = []

    for i in range(len(slices)):
        for k in range(len(slices[i])):
            initlist.append([slices[i][k][0, 0],
                             slices[i][k][0, 1],
                             slices[i][k][0, 2],
                             slices[i][k][1, 2],
                             slices[i][k][2, 2],
                             slices[i][k][2, 1],
                             slices[i][k][2, 0],
                             slices[i][k][1, 0]])
        vallist.append(initlist)
        initlist = []

    for i in range(len(slices)):
        for k in range(len(slices[i])):
            shiftlist.append(shifter(vallist[i][k], 1))
        shiftvallist.append(shiftlist)
        shiftlist = []

    for k in range(len(slices)):
        for i in range(len(vallist[k])):
            for j in range(8):
                sublist.append(
                    int(vallist[k][i][j]) - int(shiftvallist[k][i][j]))
            subslist.append(sublist)
            sublist = []
        subvallist.append(subslist)
        subslist = []

    for k in range(len(slices)):
        for l in range(len(filpix[k])):
            x = [j for j, y in enumerate(subvallist[k][l]) if y == k + 1]
            y = [j for j, z in enumerate(vallist[k][l]) if z == k + 1]
            if len(x) == 0:
                labelfil[filpix[k][l][0], filpix[k][l][1]] = 0
            if len(x) == 1:
                if len(y) >= 2:
                    extran.append(filpix[k][l])
                    labelfil[filpix[k][l][0], filpix[k][l][1]] = 0
                    # if len(extran) >= 2:
                    #     for i in extran:
                    #         for j in extran:
                    #             if i != j:
                    #                 if distance(i[0], j[0], i[1], j[1]) == np.sqrt(2.0):
                    #                     proximity = [(i[0], i[1] - 1),
                    #                                  (i[0], i[1] + 1),
                    #                                  (i[0] - 1, i[1]),
                    #                                  (i[0] + 1, i[1]),
                    #                                  (i[0] - 1, i[1] + 1),
                    #                                  (i[0] + 1, i[1] + 1),
                    #                                  (i[0] - 1, i[1] - 1),
                    #                                  (i[0] + 1, i[1] - 1)]
                    #                     match = set(filpix[k]) & set(proximity)
                    #                     if len(match) > 0:
                    #                         for z in match:
                    #                             labelfil[z[0], z[1]] = 0
    return labelfil


def in_ipynb():
    try:
        cfg = get_ipython().config
        if cfg['IPKernelApp']['parent_appname'] == 'ipython-notebook':
            return True
        else:
            return False
    except NameError:
        return False


def make_final_skeletons(labelisofil, inters, verbose=False, save_png=False,
                         save_name=None):
    '''
    Creates the final skeletons outputted by the algorithm.

    Parameters
    ----------
    labelisofil : list
        List of labeled skeletons.
    inters : list
        Positions of the intersections in each skeleton.
    verbose : bool, optional
        Enables plotting of the final skeleton.
    save_png : bool, optional
        Saves the plot made in verbose mode. Disabled by default.
    save_name : str, optional
        For use when ``save_png`` is enabled.
        **MUST be specified when ``save_png`` is enabled.**

    Returns
    -------
    filament_arrays : list
        List of the final skeletons.
    '''

    filament_arrays = []

    for n, (skel_array, intersec) in enumerate(zip(labelisofil, inters)):
        copy_array = np.zeros(skel_array.shape, dtype=int)

        for inter in intersec:
            for pts in inter:
                x, y = pts
                copy_array[x, y] = 1

        copy_array[np.where(skel_array >= 1)] = 1

        cleaned_array = find_extran(1, copy_array)

        filament_arrays.append(cleaned_array)

        if verbose or save_png:
            if save_png and save_name is None:
                Warning("Must give a save_name when save_png is enabled. No"
                        " plots will be created.")

            plt.clf()
            plt.imshow(cleaned_array, origin='lower', interpolation='nearest')

            if save_png:
                try_mkdir(save_name)
                plt.savefig(os.path.join(save_name,
                                         save_name+"_final_skeleton_"+str(n)+".png"))
            if verbose:
                plt.show()
            if in_ipynb():
                plt.clf()

    return filament_arrays


def recombine_skeletons(skeletons, offsets, orig_size, pad_size,
                        verbose=False):
    '''
    Takes a list of skeleton arrays and combines them back into
    the original array.

    Parameters
    ----------
    skeletons : list
        Arrays of each skeleton.
    offsets : list
        Coordinates where the skeleton arrays have been sliced from the
        image.
    orig_size : tuple
        Size of the image.
    pad_size : int
        Size of the array padding.
    verbose : bool, optional
        Enables printing when a skeleton array needs to be resized to fit
        into the image.

    Returns
    -------
    master_array : numpy.ndarray
        Contains all skeletons placed in their original positions in the image.
    '''

    num = len(skeletons)

    master_array = np.zeros(orig_size)
    for n in range(num):
        x_off, y_off = offsets[n][0]  # These are the coordinates of the bottom
        # left in the master array.
        x_top, y_top = offsets[n][1]

        # Now check if padding will put the array outside of the original array
        # size
        excess_x_top = x_top - orig_size[0]

        excess_y_top = y_top - orig_size[1]

        copy_skeleton = copy.copy(skeletons[n])

        size_change_flag = False

        if excess_x_top > 0:
            copy_skeleton = copy_skeleton[:-excess_x_top, :]
            size_change_flag = True

        if excess_y_top > 0:
            copy_skeleton = copy_skeleton[:, :-excess_y_top]
            size_change_flag = True

        if x_off < 0:
            copy_skeleton = copy_skeleton[-x_off:, :]
            x_off = 0
            size_change_flag = True

        if y_off < 0:
            copy_skeleton = copy_skeleton[:, -y_off:]
            y_off = 0
            size_change_flag = True

        if verbose & size_change_flag:
            print "REDUCED FILAMENT %s/%s TO FIT IN ORIGINAL ARRAY" % (n, num)

        x, y = np.where(copy_skeleton >= 1)
        for i in range(len(x)):
            master_array[x[i] + x_off, y[i] + y_off] = 1

    return master_array

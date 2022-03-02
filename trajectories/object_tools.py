"""
object_tools module.

Created on Tue Feb 15 15:21:50 2022

@author: Peter Clark
"""
import numpy as np
from scipy import ndimage

#debug_unsplit = True
debug_unsplit = False
#debug_label = True
debug_label = False


def label_3D_cyclic(mask) :
    """
    Label 3D objects taking account of cyclic boundary in x and y.

    Uses ndimage(label) as primary engine.

    Args:
        mask: 3D logical array with object mask (i.e. objects are
            contiguous True).

    Returns
    -------
        Object identifiers::

            labs  : Integer array[nx,ny,nz] of labels. -1 denotes unlabelled.
            nobjs : number of distinct objects. Labels range from 0 to nobjs-1.

    @author: Peter Clark

    """
    (nx, ny, nz) = np.shape(mask)
    labels, nobjects = ndimage.label(mask)
    labels -=1
    def relabel(labs, nobjs, i,j) :
        lj = (labs == j)
        labs[lj] = i
        for k in range(j+1,nobjs) :
            lk = (labs == k)
            labs[lk] = k-1
        nobjs -= 1
        return labs, nobjs

    def find_objects_at_edge(minflag, dim, n, labs, nobjs) :
        i = 0
        while i < (nobjs-2) :
            # grid points corresponding to label i
            posi = np.where(labs == i)
            posid = posi[dim]
            if minflag :
                test1 = (np.min(posid) == 0)
                border = '0'
            else:
                test1 = (np.max(posid) == (n-1))
                border = f"n{['x','y'][dim]}-1"
            if test1 :
                if debug_label :
                    print('Object {:03d} on {}={} border?'.\
                          format(i,['x','y'][dim],border))
                j = i+1
                while j < nobjs :
                    # grid points corresponding to label j
                    posj = np.where(labs == j)
                    posjd = posj[dim]

                    if minflag :
                        test2 = (np.max(posjd) == (n-1))
                        border = f"n{['x','y'][dim]}-1"
                    else:
                        test2 = (np.min(posjd) == 0)
                        border = '0'

                    if test2 :
                        if debug_label :
                            print('Match Object {:03d} on {}={} border?'\
                                  .format(j,['x','y'][dim],border))

                        if minflag :
                            ilist = np.where(posid == 0)
                            jlist = np.where(posjd == (n-1))
                        else :
                            ilist = np.where(posid == (n-1))
                            jlist = np.where(posjd == 0)

                        int1 = np.intersect1d(posi[1-dim][ilist],
                                              posj[1-dim][jlist])
                        # z-intersection
                        int2 = np.intersect1d(posi[2][ilist],
                                              posj[2][jlist])
                        if np.size(int1)>0 and np.size(int2)>0 :
                            if debug_label :
                                print('Yes!',i,j)
                            labs, nobjs = relabel(labs, nobjs, i, j)
                    j += 1
            i += 1
        return labs, nobjs

    labels, nobjects = find_objects_at_edge(True,  0, nx, labels, nobjects)
    labels, nobjects = find_objects_at_edge(False, 0, nx, labels, nobjects)
    labels, nobjects = find_objects_at_edge(True,  1, ny, labels, nobjects)
    labels, nobjects = find_objects_at_edge(False, 1, ny, labels, nobjects)

    return labels, nobjects

def unsplit_object( pos, nx, ny ) :
    """
    Gather together points in object separated by cyclic boundaries.

        For example, if an object spans the 0/nx boundary, so some
        points are close to zero, some close to nx, they will be adjusted to
        either go from negative to positive, close to 0, or less than nx to
        greater than. The algorithm tries to group on the larges initial set.

    Args:
        pos      : grid positions of points in object.
        nx,ny    : number of grid points in x and y directions.

    Returns
    -------
        Adjusted grid positions of points in object.

    @author: Peter Clark

    """
    global debug_unsplit
    if debug_unsplit : print('pos:', pos)

    n = (nx, ny)

    for dim in range(2):
        q0 = pos[:, dim] < n[dim] * 0.25
        q3 = pos[:, dim] > n[dim] * 0.75
        if np.sum(q0) < np.sum(q3):
            pos[q0, dim] += n[dim]
        else:
            pos[q3, dim] -= n[dim]

    return pos

def unsplit_objects(trajectory, labels, nobjects, nx, ny) :
    """
    Unsplit a set of objects at a set of times using unsplit_object on each.

    Args:
        trajectory     : Array[nt, np, 3] of trajectory points, with nt \
                         times and np points.
        labels         : labels of trajectory points.
        nx,ny   : number of grid points in x and y directions.

    Returns
    -------
        Trajectory array with modified positions.

    @author: Peter Clark

    """
    global debug_unsplit
#    print np.shape(trajectory)
    if nobjects < 2:
        return trajectory

    print('Unsplitting Objects:')

    for iobj in range(0,nobjects):
        if debug_unsplit : print('Unsplitting Object: {:03d}'.format(iobj))

        for it in range(0,np.shape(trajectory)[0]) :
            if debug_unsplit : print('Time: {:03d}'.format(it))
            tr = trajectory[it,labels == (iobj),:]
            if ((np.max(tr[:,0])-np.min(tr[:,0])) > nx/2 ) or \
               ((np.max(tr[:,1])-np.min(tr[:,1])) > ny/2 ) :
                trajectory[it, labels == iobj,:] = \
                unsplit_object(trajectory[it,labels == iobj,:], \
                                               nx, ny)
                if debug_unsplit : print('New object:',\
                    trajectory[it,labels == iobj,:])

    return trajectory

def _test_indices_3d(i, j, k, k_limit, k_start, diagonal=False):
    """
    Find the indices of neighbouring grid boxes for a specified grid box.

    Args:
        i,j,k      : x-, y-, z- indices of a specified grid box
        k_limit    : the lower bound of z-direction 
        k_start    : the upper bound of z-direction
        diagonal   : whether to treat the diagonal grid cells as neighbours

    Returns
    -------
        indices of neighbouring grid boxes.

    @author: Jian-Feng Gu

    """

    # Standard, cells sharing a border are adjacent.
    if k > k_start and k < k_limit:
       indices = [(i-1, j, k), (i+1, j, k), (i, j-1, k), (i, j+1, k), (i, j, k-1), (i, j, k+1)]
       if diagonal:
           # Diagonal cells considered adjacent.
           indices += [(i-1, j-1, k), (i-1, j+1, k), (i+1, j-1, k), (i+1, j+1, k), (i-1, j, k-1), (i+1, j, k-1), (i, j-1, k-1), (i, j+1, k-1), (i-1, j, k+1), (i+1, j, k+1), (i, j-1, k+1), (i, j+1, k+1)]

    elif k <= k_start:
       indices = [(i-1, j, k), (i+1, j, k), (i, j-1, k), (i, j+1, k), (i, j, k+1)]
       if diagonal:
           # Diagonal cells considered adjacent.
           indices += [(i-1, j-1, k), (i-1, j+1, k), (i+1, j-1, k), (i+1, j+1, k), (i-1, j, k+1), (i+1, j, k+1), (i, j-1, k+1), (i, j+1, k+1)]

    elif k >= k_limit:
       indices = [(i-1, j, k), (i+1, j, k), (i, j-1, k), (i, j+1, k), (i, j, k-1)]
       if diagonal:
           # Diagonal cells considered adjacent.
           indices += [(i-1, j-1, k), (i-1, j+1, k), (i+1, j-1, k), (i+1, j+1, k), (i-1, j, k-1), (i+1, j, k-1), (i, j-1, k-1), (i, j+1, k-1)]

    return indices

def label_clds_3d(mask, diagonal=False, wrap=True, min_cells=0, k_start=1):
    """
    Label 3D contiguous objects.

    Args:
        mask      : 3D mask of True/False representing objects.
        diagonal  : Whether to treat diagonal grid boxes as contiguous.
        wrap      : Whether to wrap on edge.
        min_cells : Minimum number of grid-boxes to include in an object.

    Return:
    -------
        tuple(int, np.ndarray): max_label and 3D array of object labels.

    @author Jian-Feng Gu
    """

    labels = np.zeros_like(mask, dtype=np.int32)
    max_label = 0
    acceptable_blobs = []
    for k in range(k_start, mask.shape[2]-1):
        for j in range(mask.shape[1]):
            for i in range(mask.shape[0]):
                if labels[i, j, k]:
                    continue

                if mask[i, j, k]:
                    blob_count = 1
                    max_label += 1
                    labels[i, j, k] = max_label
                    outers = [(i, j, k)]
                    while outers:
                        new_outers = []
                        for ii, jj, kk in outers:
                            for it, jt, kt in _test_indices_3d(ii, jj, kk, mask.shape[2]-1, k_start, diagonal):
                                if not wrap:
                                    if it < 0 or it >= mask.shape[0] or \
                                                    jt < 0 or jt >= mask.shape[1]:
                                        continue
                                else:
                                    it %= mask.shape[0]
                                    jt %= mask.shape[1]

                                if kt >= mask.shape[2] or kt < k_start:
                                   print(kt, 'out of range of vertical domain')

                                if not labels[it, jt, kt] and mask[it, jt, kt]:
                                    blob_count += 1
                                    new_outers.append((it, jt, kt))
                                    labels[it, jt, kt] = max_label
                        outers = new_outers

                    if blob_count >= min_cells:
                        acceptable_blobs.append(max_label)

    if min_cells > 0:
        out_blobs = np.zeros_like(labels)
        num_acceptable_blobs = 1
        for blob_index in acceptable_blobs:
            out_blobs[labels == blob_index] = num_acceptable_blobs
            num_acceptable_blobs += 1

        return out_blobs, num_acceptable_blobs
    else:
        return labels, max_label

def label_halo(data, labels, nobjects, nlay=5, threshold=0.00001):
    """
    Function to set up the origin of halo region around cloud object for
    backward and forward trajectories.

    Args:
        data           : data for object indicator
        labels         : labels for cloud objects
        nobjects       : cloud object IDs
        nlay=5         : default number of layers from cloud edge for halo region
        thresh=0.00001 : Cloud liquid water threshold for clouds.

    Returns:
        Halo labels around cloud objects::

            halo_labels       : labels of original halo points at ref time,
                                lables are the same as the corresponding object id
            halo_dist_labels  : labels of original halo points, encoded with
                                the distance information away from the cloud object
            halo_logic_pos    : array of logical variable that marks the halo region

    @author: Jian-Feng Gu

    """

    def _test_indices(i, j, diagonal=False, extended=False):
        if extended:
            # Count any cells in a 5x5 area centred on the current i, j cell as being adjacent.
            indices = []
            for ii in range(i - 2, i + 3):
                for jj in range(j - 2, j + 3):
                    indices.append((ii, jj))
        else:
            # Standard, cells sharing a border are adjacent.
            indices = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
            if diagonal:
                # Diagonal cells considered adjacent.
                indices += [(i-1, j-1), (i-1, j+1), (i+1, j-1), (i+1, j+1)]
        return indices

    def grow(a, diagonal=False):
        """
        Grow original array by one cell.
        
        Args:
            a        : input array.
            diagonal : whether to grow in diagonal direction.
        Return: 
            new array that has been extended in each dir by one unit.
        """
        anew = a.copy()
        for i, j in _test_indices(0, 0, diagonal):
            anew |= np.roll(np.roll(a, i, axis=0), j, axis=1)
        return anew


    halo_labels = np.ones_like(labels)*(-1)
    halo_dist_labels = np.zeros_like(labels)*1.0
    halo_logic_pos = labels>-1

    # cloud logic field
    cld_logic = labels>-1
    # extended cloud logic field for cloud halo
    cld_extend_logic = cld_logic.copy()
    # temporary cloud logic field
    temp = cld_extend_logic.copy()
    # now start labelling halo region for object iobj
    for n in range(0,nlay):
        # extend the cloud boundary
        cld_extend_logic = grow(cld_extend_logic, diagonal=True)
        # label the halo region every grid box away from the cloud boundary
        layer = cld_extend_logic.copy()
        layer[temp] = False
        layer = np.logical_and(layer, data<threshold)
        # label the halo region of object with the number of layers
        halo_labels[layer] = n
        # encode the distance of halo away from the cloud boundary
        halo_dist_labels[layer] = np.exp(n+1)
        temp = cld_extend_logic

    halo_logic_pos = halo_labels>-1

    return halo_labels, halo_dist_labels, halo_logic_pos

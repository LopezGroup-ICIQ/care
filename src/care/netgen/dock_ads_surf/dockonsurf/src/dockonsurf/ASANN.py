#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#                    GNU LESSER GENERAL PUBLIC LICENSE
#                        Version 3, 29 June 2007
#
#  Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
#  Everyone is permitted to copy and distribute verbatim copies
#  of this license document, but changing it is not allowed.
#
#
#   This version of the GNU Lesser General Public License incorporates
# the terms and conditions of version 3 of the GNU General Public
# License, supplemented by the additional permissions listed below.
#
#   0. Additional Definitions.
#
#   As used herein, "this License" refers to version 3 of the GNU Lesser
# General Public License, and the "GNU GPL" refers to version 3 of the GNU
# General Public License.
#
#   "The Library" refers to a covered work governed by this License,
# other than an Application or a Combined Work as defined below.
#
#   An "Application" is any work that makes use of an interface provided
# by the Library, but which is not otherwise based on the Library.
# Defining a subclass of a class defined by the Library is deemed a mode
# of using an interface provided by the Library.
#
#   A "Combined Work" is a work produced by combining or linking an
# Application with the Library.  The particular version of the Library
# with which the Combined Work was made is also called the "Linked
# Version".
#
#   The "Minimal Corresponding Source" for a Combined Work means the
# Corresponding Source for the Combined Work, excluding any source code
# for portions of the Combined Work that, considered in isolation, are
# based on the Application, and not on the Linked Version.
#
#   The "Corresponding Application Code" for a Combined Work means the
# object code and/or source code for the Application, including any data
# and utility programs needed for reproducing the Combined Work from the
# Application, but excluding the System Libraries of the Combined Work.
#
#   1. Exception to Section 3 of the GNU GPL.
#
#   You may convey a covered work under sections 3 and 4 of this License
# without being bound by section 3 of the GNU GPL.
#
#   2. Conveying Modified Versions.
#
#   If you modify a copy of the Library, and, in your modifications, a
# facility refers to a function or data to be supplied by an Application
# that uses the facility (other than as an argument passed when the
# facility is invoked), then you may convey a copy of the modified
# version:
#
#    a) under this License, provided that you make a good faith effort to
#    ensure that, in the event an Application does not supply the
#    function or data, the facility still operates, and performs
#    whatever part of its purpose remains meaningful, or
#
#    b) under the GNU GPL, with none of the additional permissions of
#    this License applicable to that copy.
#
#   3. Object Code Incorporating Material from Library Header Files.
#
#   The object code form of an Application may incorporate material from
# a header file that is part of the Library.  You may convey such object
# code under terms of your choice, provided that, if the incorporated
# material is not limited to numerical parameters, data structure
# layouts and accessors, or small macros, inline functions and templates
# (ten or fewer lines in length), you do both of the following:
#
#    a) Give prominent notice with each copy of the object code that the
#    Library is used in it and that the Library and its use are
#    covered by this License.
#
#    b) Accompany the object code with a copy of the GNU GPL and this license
#    document.
#
#   4. Combined Works.
#
#   You may convey a Combined Work under terms of your choice that,
# taken together, effectively do not restrict modification of the
# portions of the Library contained in the Combined Work and reverse
# engineering for debugging such modifications, if you also do each of
# the following:
#
#    a) Give prominent notice with each copy of the Combined Work that
#    the Library is used in it and that the Library and its use are
#    covered by this License.
#
#    b) Accompany the Combined Work with a copy of the GNU GPL and this license
#    document.
#
#    c) For a Combined Work that displays copyright notices during
#    execution, include the copyright notice for the Library among
#    these notices, as well as a reference directing the user to the
#    copies of the GNU GPL and this license document.
#
#    d) Do one of the following:
#
#        0) Convey the Minimal Corresponding Source under the terms of this
#        License, and the Corresponding Application Code in a form
#        suitable for, and under terms that permit, the user to
#        recombine or relink the Application with a modified version of
#        the Linked Version to produce a modified Combined Work, in the
#        manner specified by section 6 of the GNU GPL for conveying
#        Corresponding Source.
#
#        1) Use a suitable shared library mechanism for linking with the
#        Library.  A suitable mechanism is one that (a) uses at run time
#        a copy of the Library already present on the user's computer
#        system, and (b) will operate properly with a modified version
#        of the Library that is interface-compatible with the Linked
#        Version.
#
#    e) Provide Installation Information, but only if you would otherwise
#    be required to provide such information under section 6 of the
#    GNU GPL, and only to the extent that such information is
#    necessary to install and execute a modified version of the
#    Combined Work produced by recombining or relinking the
#    Application with a modified version of the Linked Version. (If
#    you use option 4d0, the Installation Information must accompany
#    the Minimal Corresponding Source and Corresponding Application
#    Code. If you use option 4d1, you must provide the Installation
#    Information in the manner specified by section 6 of the GNU GPL
#    for conveying Corresponding Source.)
#
#   5. Combined Libraries.
#
#   You may place library facilities that are a work based on the
# Library side by side in a single library together with other library
# facilities that are not Applications and are not covered by this
# License, and convey such a combined library under terms of your
# choice, if you do both of the following:
#
#    a) Accompany the combined library with a copy of the same work based
#    on the Library, uncombined with any other library facilities,
#    conveyed under the terms of this License.
#
#    b) Give prominent notice with the combined library that part of it
#    is a work based on the Library, and explaining where to find the
#    accompanying uncombined form of the same work.
#
#   6. Revised Versions of the GNU Lesser General Public License.
#
#   The Free Software Foundation may publish revised and/or new versions
# of the GNU Lesser General Public License from time to time. Such new
# versions will be similar in spirit to the present version, but may
# differ in detail to address new problems or concerns.
#
#   Each version is given a distinguishing version number. If the
# Library as you received it specifies that a certain numbered version
# of the GNU Lesser General Public License "or any later version"
# applies to it, you have the option of following the terms and
# conditions either of that published version or of any later version
# published by the Free Software Foundation. If the Library as you
# received it does not specify a version number of the GNU Lesser
# General Public License, you may choose any version of the GNU Lesser
# General Public License ever published by the Free Software Foundation.
#
#   If the Library as you received it specifies that a proxy can decide
# whether future versions of the GNU Lesser General Public License shall
# apply, that proxy's public statement of acceptance of any version is
# permanent authorization for you to choose that version for the
# Library.

"""Python3 implementation of the ASANN algorithm (Anisotropically corrected
Solid-Angle based Nearest-Neighbors) """

# EXTERNAL MODULES IMPORTS ###
import numpy as np
import math


# FUNCTIONS DEFINITION ###

# PRE-PROCESSING FUNCTIONS ##
def add_periodic_images(coords, cell, mode):
    """Explicitely add adjacent or surrounding periodic images.

    Parameters: coords (2D array): List of atoms coordinates to consider.
    Important: direct coordinates are expected (torus topology of side 1) if
    `pbc` is set to True. Shape expected: (nb_atoms, nb_dim), where nb_atoms
    is the number of atoms considered, and nb_dim is the dimensionnality of
    the system.

        cell (numpy 2D array): List of cell vectors to consider for periodic
        boundaries conditions. Shape expected: (nb_dim, nb_dim), where nb_dim
        is the dimensionnality of the system considered. Exemple:
        cell_vectors=[[v1_x, v1_y, v1_z], [v2_x, v2_y, v2_z], [v3_x, v3_y,
        v3_z]]

        mode (str): Determines which periodic images should be included. If
        'adjacent', all adjacent periodic images are included (all periodic
        images sharing a face), increasing the number of coordinates 9-fold.
        If 'full', all surrounding periodic images are included (all periodic
        images sharing a face, an edge or a point), increasing the number of
        coordinates 27-fold.

    Returns: (nb_atoms, new_coords) nb_atoms: number of real atom coordinates
    new_coords: numpy 2D array containing updated coordinates (initial +
    periodic images) Shape: (nb_coords, nb_dim)
    """
    # Initialize new coordinates
    new_coords = np.mod(coords, 1)  # Project coords inside initial cell

    # Iterate over cell vectors
    for vect in np.eye(cell.shape[0]):
        # Add periodic images
        if mode == 'adjacent':
            new_coords = np.vstack((new_coords, coords + vect, coords - vect))
        elif mode == 'full':
            new_coords = np.vstack((new_coords, new_coords + vect,
                                    new_coords - vect))  # Recursive process
            # to include all dimension combinaisons
        else:
            raise NotImplementedError

    return coords.shape[0], new_coords


def get_pbc_vectors(coords, pbc, nb_atoms=None):
    """Compute pairwise vectors with or without periodic boundaries conditions.

    Parameters:
        coords (numpy 2D array): List of atoms coordinates to
        consider. Important: direct coordinates are expected (torus topology of
        side 1) if `pbc` is set to True. Shape expected: (nb_coords, nb_dim),
        where nb_coords is the number of coordinates considered, and nb_dim is
        the dimensionnality of the system.

        pbc (bool): Determines if periodic boundaries conditions should be
        applied. Default to True. If True, coordinates are interpreted as
        direct coordinates and the distance between points is computed as the
        minimum euclidian distance between all duplicates (due to periodic
        boundaries conditions) of these points. If false, coordinates are
        interpreted as cartesian coordinates and the metric used is simply
        the euclidian distance. Note: the minimum image convention is not
        applied if periodic images are already explicitely included.

        nb_atoms (int, optional): Number of real atoms coordinates (i.e. for
        which distances must be computed). This is particularly useful for
        excluding periodic images coordinates as central atoms. The real
        atoms coordinates are supposed to be the first coordinates in `coords`

    Returns: numpy 2D array containing pairwise vectors from `coords`.
    Important: if coords are direct coordinates (i.e. `pbc` is set to True),
    the vectors are in direct coordinates. If coords are in cartesian
    coordinates (i.e. `pbc` is set to False), the vectors are in cartesian
    coordinates. vectors[i,j] = (v_ij_x, v_ij_y, v_ij_z) = (r_j_x - r_i_x,
    r_j_y - r_i_y, r_j_z - r_i_z) Shape : (nb_atoms, nb_coords, nb_dim)
    """
    # Retrieve number of non-virtual atoms (from which distances are computed)
    if nb_atoms is None:
        nb_atoms = coords.shape[0]

    # Compute cartesian vectors
    vectors = coords[np.newaxis, :, :] - coords[:nb_atoms, np.newaxis, :]

    # Applying PBC (if minimum image convention is required)
    if pbc and pbc not in ('adjacent', 'full'):
        # vectors = np.mod(vectors + 0.5, 1) - 0.5 # modulo operation is a
        # bit slower than floor operation...
        vectors -= np.floor(vectors + 0.5)

    return vectors


def get_sorted_distances(coords, pbc, nb_atoms=None, cell=np.eye(3)):
    """Compute distances-sorted pairwise distances and vectors with or
    without periodic boundaries conditions.

    Parameters: coords (numpy 2D array): List of atoms coordinates to
    consider. Important: direct coordinates are expected (torus topology of
    side 1) if `pbc` is set to True. Shape expected: (nb_atoms, nb_dim),
    where nb_atoms is the number of atoms considered, and nb_dim is the
    dimensionnality of the system.

        pbc (bool): Determines if periodic boundaries conditions should be
        applied. Default to True. If True, coordinates are interpreted as
        direct coordinates and the distance between points is computed as the
        minimum euclidian distance between all duplicates (due to periodic
        boundaries conditions) of these points. If false, coordinates are
        interpreted as cartesian coordinates and the metric used is simply
        the euclidian distance.

        nb_atoms (int, optional): Number of real atoms coordinates (i.e. for
        which distances must be computed). This is particularly useful for
        excluding periodic images coordinates as central atoms. If None,
        all `coords` coordinates are considered. Default: None

        cell (2D array, optional): List of cell vectors to consider when
        periodic boundaries conditions are considered. Shape expected: (
        nb_dim, nb_dim), where nb_dim is the dimensionnality of the system
        considered. Important: For this parameter to be taken into account,
        `pbc` must be set to True. Exemple: cell_vectors=[[v1_x, v1_y, v1_z],
        [v2_x, v2_y, v2_z], [v3_x, v3_y, v3_z]] Default: numpy.eye(3) (i.e. a
        cubic cell of side 1).

    Returns: (sorted_distances, sorted_vectors, sorted_indexes)
    sorted_vectors: numpy 3D array containing pairwise vectors from `coords`,
    where each i-th row is sorted by increasing distance with respect to the
    i-th coordinates. Important: The vectors are in cartesian coordinates.
    sorted_vectors[i,j] = (v_ij_x, v_ij_y, v_ij_z) = (r_j_x - r_i_x, r_j_y -
    r_i_y, r_j_z - r_i_z) Shape : (nb_atoms, nb_coords, nb_dim)
    sorted_distances: numpy 2D array containing pairwise distances from
    `coords`, where each i-th row is sorted by increasing distance with
    respect to the i-th coordinates. Important: Cartesian euclidian distance
    is used here. sorted_distances[i,j-1] <= sorted_distances[i,
    j] <= sorted_distances[i,j+1] Shape : (nb_atoms, nb_coords)
    sorted_indexes: numpy 2D array containing for each row the
    distances-sorted indexes of neighbors. In other words, the atom indexed
    sorted_indexes[i,j] is the j-th nearest neighbor of atom i. Shape : (
    nb_atoms, nb_coords)
    """
    # Retrieve number of atoms if not given
    if nb_atoms is None:
        nb_atoms = coords.shape[0]

    # Computes pairwise vectors
    vectors = get_pbc_vectors(coords, pbc, nb_atoms=nb_atoms)  # vector.shape =
    # (nb_atoms, nb_coords, nb_dim)

    # Convert into cartesian coordinates if pbc
    if pbc:
        vectors = np.dot(vectors, cell)  # dot product with cell vectors to
        # have cartesian coordinates (for distance computation)

    # Computes pairwise distances
    distances = np.sqrt(np.sum(vectors ** 2, axis=-1))  # simply the square
    # root of the sum of squared components for each pairwise vector

    # Sorting vectors and distances (with respect to distance) for improved
    # performance of the (CA)SANN algorithm Getting sorted indexes to apply to
    # distances and vectors
    sorted_index_axis1 = np.argsort(distances, axis=-1)  # sorting columns
    sorted_index_axis0 = np.arange(nb_atoms)[:, None]  # keeping rows
    # Rearranging distances and vectors so that each row is sorted by
    # increasing distance. (i.e. distances[i, j-1] <= distances[i,
    # j] <= distances[i, j+1])
    distances = distances[sorted_index_axis0, sorted_index_axis1]
    vectors = vectors[sorted_index_axis0, sorted_index_axis1]

    return distances, vectors, sorted_index_axis1


# SANN IMPLEMENTATION ##
def get_SANN(all_distances):
    """Computes coordination numbers according to the SANN algorithm,
    from all pairwise distances.

    Parameters: all_distances: numpy 2D array containing pairwise distances
    from `coords`, where each i-th row is sorted by increasing distance with
    respect to the i-th coordinates. Important: Cartesian euclidian distance
    is used here. sorted_distances[i,j-1] <= sorted_distances[i,
    j] <= sorted_distances[i,j+1] Shape : (nb_atoms, nb_coords)

    Returns: list_sann_CN : Numpy 1D array containing SANN-based coordination
    numbers (i.e. number of neighbors).

        list_sann_radius : Numpy 1D array containing SANN-based coordination
        radius (i.e. coordination sphere radius).
    """

    # Retrieve number of atoms
    nb_coords = all_distances.shape[1]

    # Initialize coordination number vector (i-th element is the coordination
    # number of the i-th atom) and coordination radius
    list_sann_CN = list()
    list_sann_radius = list()
    # Initialize sum of distances vector (i-th element is meant to temporarly
    # store the sum of the i-th atom's 3 nearest neighbors distances)
    list_dist_sum = all_distances[:, 1:4].sum(axis=1)

    # Treat each atom separately (since CN can be different for each atom,
    # Numpy methods are unsuited here)
    for (dist_sum, atom_distances) in zip(list_dist_sum, all_distances):
        sann_CN = 3  # Set CN to 3 (i.e. the minimum CN value for the SANN
        # algorithm) SANN algorithm (i.e. while SANN radius sum(r_ij,1,
        # m)/(m-2) > r_i(m+1), increase m by 1)
        while (sann_CN + 1 < nb_coords) and (
                dist_sum / (sann_CN - 2) >= atom_distances[sann_CN + 1]):
            dist_sum += atom_distances[sann_CN + 1]
            sann_CN += 1
        # Store SANN CN found
        list_sann_CN.append(sann_CN)
        list_sann_radius.append(dist_sum / (sann_CN - 2))

    return (np.array(list_sann_CN), np.array(list_sann_radius))


## ANISOTROPY HANDLING ##
def dist_to_barycenter(nearest_neighbors, nearest_distances, radius):
    """Compute the distance from the central atom to the barycenter of
    nearest neighbors.

    Parameters: nearest_neighbors: numpy 2D array containing nearest
    neighbors vectors from the central atom, sorted by increasing distance
    with respect to the central atom. Important: The vectors must be in
    cartesian coordinates.

        nearest_distances: numpy 1D array containing nearest neighbors
        distances from the central atom, sorted by increasing distance with
        respect to the central atom.

        radius (float): radius R_i_m of the sphere of coordination.

    Returns: dist_bary (float): distance from the central atom to the
    barycenter of nearest neighbors, weighted by relative solid angles

    Computational details: The barycenter is computed using a solid angle
    weight (i.e. the solid angle associated with the corresponding neighbor).
    """

    # Compute solid angles (SA) of neighbors
    list_SA = 1 - nearest_distances / radius

    # Compute SA-based barycenter
    bary_vector = np.sum(nearest_neighbors * list_SA[:, np.newaxis],
                         axis=0) / np.sum(list_SA)

    # Returns distance from the central atom to the barycenter
    return (math.sqrt(np.sum(bary_vector ** 2)), bary_vector)


def angular_correction(nearest_neighbors, nearest_distances, radius):
    """Compute the angular correction `ang_corr`, such that R_i_m = sum(r_ij,
    j=1..m)/(m-2(1-ang_corr)).

    Parameters: nearest_neighbors: numpy 2D array containing the m nearest
    neighbors vectors from the central atom, sorted by increasing distance
    with respect to the central atom. Important: The vectors must be in
    cartesian coordinates.

        nearest_distances: numpy 1D array containing the m nearest neighbors
        distances from the central atom, sorted by increasing distance with
        respect to the central atom.

        radius (float): radius R_i_m of the sphere of coordination.

    Returns:
        ang_corr (float): angular correction.

    Computational details: The angular correction is computed from the
    distance between the nearest neighbor barycenter and the central atom
    `dist_bary`.

        Let us define `alpha` such that: dist_bary = alpha * radius Then,
        mathematical derivations show that: ang_corr = (alpha + sqrt(alpha**2
        + 3*alpha))/3
    """

    # Computing the ratio between the distance to the nearest neighbors
	# barycenter and the radius
    alpha = dist_to_barycenter(nearest_neighbors, nearest_distances, radius)[
                0] / radius
    vector = dist_to_barycenter(nearest_neighbors, nearest_distances, radius)[1]

    # Computing angular correction
    return ((alpha + math.sqrt(alpha ** 2 + 3 * alpha)) / 3, vector)


## ASANN IMPLEMENTATION ##
def get_ASANN(sorted_distances, sorted_vectors, sann_CNs, sann_radii):
    """Update ASANN-based coordination numbers using an angular correction term.

    Parameters: sorted_vectors: numpy 3D array containing pairwise vectors
    from `coords`, where each i-th row is sorted by increasing distance with
    respect to the i-th coordinates. Important: The vectors must be in
    cartesian coordinates. sorted_vectors[i,j] = (v_ij_x, v_ij_y, v_ij_z) = (
    r_j_x - r_i_x, r_j_y - r_i_y, r_j_z - r_i_z) Shape : (nb_atoms,
    nb_coords, nb_dim)

        sorted_distances: numpy 2D array containing pairwise distances from
        `coords`, where each i-th row is sorted by increasing distance with
        respect to the i-th coordinates. Important: Cartesian euclidian
        distance is used here. sorted_distances[i,j-1] <= sorted_distances[i,
        j] <= sorted_distances[i,j+1] Shape : (nb_atoms, nb_coords)

        sann_CNs: Numpy 1D array containing SANN-based coordination numbers (
        i.e. number of neighbors).

        sann_radii: Numpy 1D array containing SANN-based coordination radius
        (i.e. radius of the coordination spheres).

    Returns: list_asann_CN : Numpy 1D array containing ASANN-based
    coordination numbers (i.e. number of neighbors, with an angular
    correction term).

        list_asann_radius : Numpy 1D array containing ASANN-based
        coordination radius (i.e. coordination sphere radius).

    Computational details: ASANN-based coordination number is defined as the
    maximum coordination number m' such that forall m>=m', R_ang_i_m = sum(
    r_ij, j=1..m)/(m-2(1-ang_corr)) < r_i(m+1)

        It is easy to show that R_ang_i_m <= R_i_m, and therefore, m'<=m.
        Consequently, the ASANN algorithm is sure to converge.

        Unlike SC-ASANN algorithm, the angular correction is solely computed
        using the SANN radius (instead of a self-coherent approach, where the
        angular term is defined by (and defines) the ASANN radius itself)
    """

    # Retrieve number of atoms
    nb_coords = sorted_distances.shape[1]

    # Create coordination number vector (i-th element is the coordination
    # number of the i-th atom) and coordination radius
    list_asann_CN = list()
    list_asann_radius = list()
    list_bary_vector = list()

    # Treat each atom separately (since CN can be different for each atom,
    # Numpy methods are unsuited here)
    for (atom_distances, atom_neighbors, sann_CN, sann_radius) in zip(
            sorted_distances, sorted_vectors, sann_CNs, sann_radii):

        # Computes angular correction
        nearest_distances = atom_distances[1:sann_CN + 1]
        nearest_neighbors = atom_neighbors[1:sann_CN + 1]
        ang_corr, vec = angular_correction(nearest_neighbors, nearest_distances,
                                           sann_radius)
        beta = 2 * (1 - ang_corr)

        # ASANN algorithm (i.e. while ASANN radius sum(r_ij, j=1..m)/(m-2*(
        # 1-ang_corr)) >= r_i(m+1), increase m by 1)
        asann_CN = int(
            beta) + 1  # Set CN to floor(2*(1-ang_corr)) + 1 (i.e. the
        # minimum CN value for the ASANN algorithm)
        dist_sum = atom_distances[
                   1:asann_CN + 1].sum()  # Initialize sum of distances
        while (asann_CN + 1 < nb_coords) and (
                dist_sum / (asann_CN - beta) >= atom_distances[asann_CN + 1]):
            dist_sum += atom_distances[asann_CN + 1]
            asann_CN += 1

        # Store ASANN CN and radius found
        list_asann_CN.append(asann_CN)
        list_asann_radius.append(dist_sum / (asann_CN - beta))
        list_bary_vector.append(vec)

    return (np.array(list_asann_CN), np.array(list_asann_radius),
            np.array(list_bary_vector))


# VARIANTS DEFINITIONS ##
def get_self_consistent_ASANN(sorted_distances, sorted_vectors, sann_CNs,
                              radius_eps=1e-2):
    """Update ASANN-based coordination numbers using an angular correction
    term computed in a self-consistent manner.

    Parameters: sorted_vectors: numpy 3D array containing pairwise vectors
    from `coords`, where each i-th row is sorted by increasing distance with
    respect to the i-th coordinates. Important: The vectors must be in
    cartesian coordinates. sorted_vectors[i,j] = (v_ij_x, v_ij_y, v_ij_z) = (
    r_j_x - r_i_x, r_j_y - r_i_y, r_j_z - r_i_z) Shape: (nb_atoms, nb_atoms,
    nb_dim)

        sorted_distances: numpy 2D array containing pairwise distances from
        `coords`, where each i-th row is sorted by increasing distance with
        respect to the i-th coordinates. Important: Cartesian euclidian
        distance is used here. sorted_distances[i,j-1] <= sorted_distances[i,
        j] <= sorted_distances[i,j+1] Shape: (nb_atoms, nb_atoms)

        sann_CNs: Numpy 1D array containing SANN-based coordination numbers (
        i.e. number of neighbors).

        radius_eps: Convergence threshold used for stopping the
        self-consistent radius computation. Default: 1e-2

    Returns: list_asann_CN : Numpy 1D array containing ASANN-based
    coordination numbers (i.e. number of neighbors, with an angular
    correction term).

        list_asann_radius : Numpy 1D array containing ASANN-based
        coordination radius (i.e. coordination sphere radius).

    Computational details: SC-ASANN-based coordination number is defined as
    the maximum coordination number m' such that forall m>=m', R_ang_i_m =
    sum(r_ij, j=1..m)/(m-2(1-ang_corr)) < r_i(m+1)

        Note that the angular correction term is computed here using the ASANN radius (defined itself from the angular correction term). In this approach, the angular correction term is computed in a self-consistent fashion.
    """

    # Create coordination number vector (i-th element is the coordination
    # number of the i-th atom) and coordination radius
    list_asann_CN = list()
    list_asann_radius = list()
    list_vectors = list()

    # Treat each atom separately (since CN can be different for each atom,
    # Numpy methods are unsuited here)
    for (atom_distances, atom_neighbors, sann_CN) in zip(sorted_distances,
                                                         sorted_vectors,
                                                         sann_CNs):
        asann_CN = sann_CN  # Set initial CN to 1 above the maximum CN that
        # can break ASANN relation (i.e. sann_CN-1)
        radius = 0
        prev_radius = 0

        # ASANN update algorithm (i.e. while ASANN radius sum(r_ij, j=1..m)/(
        # m-2(1-ang_corr)) < r_i(m+1), decrease m by 1)
        while True:
            # Extract properties of nearest neighbors
            nearest_distances = atom_distances[1:asann_CN + 1]
            nearest_neighbors = atom_neighbors[1:asann_CN + 1]

            # Computes radius iteratively
            sum_nearest = np.sum(
                nearest_distances)  # store sum of nearest distances
            radius = sum_nearest / (
                    asann_CN - 2)  # set initial radius to SANN value
            delta_radius = math.inf
            # Update radius until convergence
            while delta_radius > radius_eps:
                delta_radius = sum_nearest / (asann_CN - 2 * (
                        1 - angular_correction(nearest_neighbors,
                                               nearest_distances,
                                               radius)[0])) - radius  #
                # new_radius(old_radius) - old_radius
                vec = angular_correction(nearest_neighbors, nearest_distances,
                                         radius)[1]
                radius += delta_radius

            # Check if ASANN relation is broken
            if radius >= atom_distances[asann_CN + 1]:
                break
            asann_CN -= 1
            prev_radius = radius
            if asann_CN < 1:  # ASANN CN is not defined for less than 1
                # neighbor
                break

        # Store ASANN CN and radius found (before breaking the ASANN relation)
        list_asann_CN.append(asann_CN + 1)
        list_asann_radius.append(prev_radius)
        list_vectors.append(vec)

    return (np.array(list_asann_CN), np.array(list_asann_radius),
            np.array(list_vectors))


def convert_continuous(list_CN, list_radius, sorted_distances):
    """Convert integer coordination numbers into continuous decimal values.

    Parameters: list_CN: Numpy 1D array containing discretized coordination
    numbers (i.e. number of neighbors).

        list_radius: Numpy 1D array containing coordination radius (i.e.
        radius of the coordination spheres).

        sorted_distances: Numpy 2D array containing pairwise distances
        between coordinates, where each i-th row is sorted by increasing
        distance with respect to the i-th coordinates.

    Returns: list_continuous_CN: Numpy 1D array containing continuous
    coordination numbers.

        list_weights: 2D array containing the distance-related weights to
        apply to each neighbors of each atom.

    Computational details: In the discretized version, each neighbor is
    comptabilized exactly once. In the continuous version, the contribution
    of each neighbor is scaled by the following weight : SA/SA_max (where SA
    is the solid angle associated with the corresponding neighbor, and SA_max
    is the maximum solid angle (i.e. the solid angle associated with the
    nearest neighbor)).

        The continuous coordination number is then the sum of all weights.
    """

    # Create array to store continuous coordination numbers
    list_continuous_CN = list()
    list_weights = list()

    # Treat each atom separately (since CN can be different for each atom,
    # Numpy methods are unsuited here)
    for (atom_CN, atom_radius, atom_distances) in zip(list_CN, list_radius,
                                                      sorted_distances):
        # Extract distances of nearest neighbors
        nearest_distances = atom_distances[1:atom_CN + 1]

        # Compute solid angles of nearest neighbors
        nearest_SA = 1 - nearest_distances / atom_radius

        # Retrieve maximum solid angle (whose contribution will be 1)
        max_SA = nearest_SA[0]

        # Compute and store weights
        weights = nearest_SA / max_SA
        list_weights.append(weights)

        # Compute sum of contributions (i.e. continuous CN)
        continuous_CN = np.sum(weights)
        list_continuous_CN.append(continuous_CN)

    return np.array(list_continuous_CN), list_weights


def get_generalized(list_CN, list_weights, sorted_indexes, max_CN=None):
    """Convert coordination numbers into generalized coordination numbers.

    Parameters:
        list_CN: Numpy 1D array containing the coordination numbers.

        list_weights: 2D list containing the distance-related weights to
        apply to each neighbors of each atom. Note: In the case of
        discretized version, each weight is equal to 1.

        sorted_indexes: Numpy 2D array containing for each row the
        distances-sorted indexes of neighbors. In other words, the atom
        indexed sorted_indexes[i,j] is the j-th nearest neighbor of atom i.
        Shape = (nb_atoms, nb_coords)

        max_CN (int, optional): Value to use for the maximum coordination
        number, i.e. bulk coordination. If None, the maximum/bulk
        coordination number will be guessed. Default: None

    Returns: list_generalized_CN: Numpy 1D array containing generalized
    coordination numbers.

    Computational details: The generalized coordination number algorithm
    scales the contributions of each neighbor by the following weight :
    CN/CN_max (where CN is the coordination number associated with the
    corresponding neighbor, and CN_max is the maximum coordination achievable
    (i.e. bulk coordination)) The generalized coordination number is then the
    sum of all weights (GCN = sum(CN/max_CN, neighbors))

        This sum can be weighted futhermore (GCN = sum(CN/max_CN*SA/max_SA,
        neighbors)), using the continuous algorithm if requested (see `help(
        convert_continuous)` for more details on this algorithm).
    """

    # Create array to store generalized coordination numbers
    list_generalized_CN = list()

    # Retrieve maximal coordination number
    if max_CN is None:
        max_CN = max(list_CN)

    # Treat each atom separately (since CN can be different for each atom,
    # Numpy methods are unsuited here)
    for (weights, indexes) in zip(list_weights, sorted_indexes):
        # Initialize atom coordination number, and maximal coordination number
        atom_CN = 0

        # Loop over all neighbors, compute and add the corresponding weights
        for (weight, index) in zip(weights, indexes[1:]):
            try:
                neighbor_CN = list_CN[index]
            except IndexError:
                # if neighbor is virtual, use corresponding original one instead
                neighbor_CN = list_CN[index % sorted_indexes.shape[0]]
            atom_CN += weight * neighbor_CN

        # Divide by maximal coordination number
        list_generalized_CN.append(atom_CN / max_CN)

    return (np.array(list_generalized_CN))


def get_edges(list_CN, sorted_indexes, reduce_mode=None, nb_atoms=None):
    """Compute all edges corresponding with the connectivity graph of the
    given structure, based on discretized coordination numbers.

    Parameters: list_CN: Numpy 1D array containing the discretized
    coordination numbers (i.e. number of neighbors).

        sorted_indexes: Numpy 2D array containing for each row the
        distances-sorted indexes of neighbors. In other words, the atom
        indexed sorted_indexes[i,j] is the j-th nearest neighbor of atom i.
        Shape: (nb_atoms, nb_coords)

        reduce_mode: Edges counting mode. The ASANN/SANN algorithm can only
        find directed edges (i.e. find i->j but not j->i). This parameter
        defines the conversion mode from directed to undirected edges.
        Possible values: None: All directed edges are given, no reduction is
        performed. 'both': An undirected edge (i,j) is present only if both
        related directed edges (i->j and j->i) are found. 'any': An
        undirected edge (i,j) is present if any related directed edge (i->j
        or j->i) is found.

        nb_atoms (int, optional): Number of real atoms coordinates (i.e. for
        which distances must be computed). This is particularly useful for
        excluding periodic images coordinates as central atoms. If None,
        all coordinates are considered. Default: None

    Returns:
        list_edges: List containing all edges of the connectivity graph.
            The edges are in the form of a couple (index_node_i, index_node_j)
            Shape: (nb_bonds_found, 2)
    """

    # Create array to store all edges
    list_edges = list()

    # Treat each atom separately (since CN can be different for each atom,
    # Numpy methods are unsuited here)
    for (atom_CN, indexes) in zip(list_CN, sorted_indexes):
        # Retrieve current atom index
        index_i = indexes[0]
        # Loop over all neighbors, and add the corresponding edges
        for index_j in indexes[1:atom_CN + 1]:
            list_edges.append(
                (index_i, index_j) if reduce_mode is None else tuple(sorted((
                    index_i,
                    index_j))))  # add sorted edge instead (representing an
            # undirected edge) if conversion is required (reduce_mode not None)

    # Re-map to correct atom index if explicit periodic images are included
    if nb_atoms is not None:
        list_edges = [(index_i % nb_atoms, index_j % nb_atoms) for
                      (index_i, index_j) in list_edges]

    # Conversion of directed edges set to undirected edges set
    if reduce_mode == 'both':
        # Extract only edges that are present multiple times
        seen = dict()
        duplicates = []
        for edge in list_edges:
            if edge in seen:
                duplicates.append(edge)
            else:
                seen[edge] = None
        list_edges = duplicates
    elif reduce_mode == 'any':
        # Retrieve all unique undirected edges
        list_edges = list(set(list_edges))

    return (list_edges)


# FULL WRAPPER FUNCTION ##
def coordination_numbers(list_coords, pbc=True, cell_vectors=np.eye(3),
                         continuous=False, generalized=False, edges=True,
                         correction='ASANN', parallel=False, reduce_mode=None):
    """Computes coordination numbers according to the CASANN algorithm.

    Parameters:
        list_coords (2D array): List of atoms coordinates to
        consider. Important: direct coordinates are expected (torus topology of
        side 1), unless `pbc` is set to False. Note: This will be converted into
        a numpy.ndarray. Shape expected: (nb_atoms, nb_dim), where nb_atoms is
        the number of atoms considered, and nb_dim is the dimensionnality of the
        system.

        pbc (bool or str, optional): Determines if periodic boundaries
        conditions should be applied. Default to True. If True, coordinates
        are interpreted as direct coordinates and the distance between points
        is computed as the minimum euclidian distance between all duplicates
        (due to periodic boundaries conditions) of these points. If False,
        coordinates are interpreted as cartesian coordinates and the metric
        used is simply the euclidian distance. To explicitely include all
        adjacent periodic images (not only the minimum image convention) set
        `pbc` to 'adjacent'. This mode is particularly pertinent for small
        enough cells, but increases 9-fold the number of atoms. To
        explicitely include all surrounding periodic images set `pbc` to
        'full'. This mode is particularly pertinent for very small cells,
        but increases 27-fold the number of atoms. Note: This option implies
        the use of cell vectors (see `cell` parameter) for the computation of
        distance. Default: True.

        cell_vectors (2D array, optional): List of cell vectors to consider
        when periodic boundaries conditions are considered. Note: This will
        be converted into a numpy.ndarray. Shape expected: (nb_dim, nb_dim),
        where nb_dim is the dimensionnality of the system considered.
        Important: For this parameter to be taken into account, `pbc` must be
        set to True. Exemple: cell_vectors=[[v1_x, v1_y, v1_z], [v2_x, v2_y,
        v2_z], [v3_x, v3_y, v3_z]] Default: numpy.eye(3) (i.e. a cubic cell
        of side 1).

        continuous (bool, optional): If True, computes continuous
        coordination numbers. If False, computes discretized coordination
        numbers. Default to True. In the discretized version,
        the coordination number is equal to the number of detected neighbors.
        In the continuous version, each neighbors' contribution to the
        coordination number is 1 weighted by SA/SA_max, where SA is the solid
        angle corresponding to this neighbor and SA_max is the solid angle
        corresponding to the nearest neighbor (i.e. the maximum solid angle
        amongs all neighbors detected). Default: False.

        generalized (bool, optional): If True, computes generalized
        coordination numbers, where each neighbor is weighted by its own
        coordination number Default: False.

        edges (bool, optional): If True, computes edges of the connectivity
        graph defined by the discretized coordination numbers computed.
        Default: True.

        correction (str, optional): Determines if a correction term should be
        used. Default to 'ASANN'. The SANN algorithm suffers
        overdetermination of the coordination numbers at interfaces (i.e.
        high density gradient) basically because neighbors are always
        expected to fill the whole neighboring space (i.e. the sum of solid
        angles must be 4π), but at interfaces, neighbors are only expected to
        fill a portion of that space depending on the geometry of the
        interface. Possible values: - 'ASANN': If `correction` is set to
        'ASANN', the total sum of solid angles is rescaled by the ASANN
        angular correction term. - 'SC-ASANN': If `correction` is set to
        'SC-ASANN', the total sum of solid angles is rescaled by the ASANN
        angular correction term, computed in a self-consistent manner.
        Arguably better CNs, but more time consuming and less regularized
        continuous CNs. - None (or anything else): No correction term
        applied. This is equivalent to the SANN algorithm.

        reduce_mode: Edges counting mode. The ASANN/SANN algorithm can only
        find directed edges (e.g. find i->j but not j->i). This parameter
        defines the conversion mode from directed to undirected edges.
        Possible values: - 'both': An undirected edge (i,j) is present only
        if both related directed edges (i->j and j->i) are found. - 'any': An
        undirected edge (i,j) is present if any related directed edge (i->j
        or j->i) is found. - None: All directed edges are given, no reduction
        is performed. Default: None.

    Returns: (asann_CNs, asann_radius, asann_edges) asann_CNs (numpy array):
    list of coordination numbers computed. The order is the same as provided
    in list_coords. asann_radius (numpy array): list of coordination radii
    computed. The order is the same as provided in list_coords. asann_edges (
    list of tuples): list of edges found. Each edge is represented as a tuple
    (i,j), meaning that j was found as a neighbor of i. Note: if `edges` is
    set to False, asann_edges is set to None.

    Computational details:
        Correction:

            The SANN algorithm computes the coordination number (i.e. CN) m of
            atom i, as the minimum integer m>=3 satisfying R_i_m = sum(r_ij,
            j=1..m)/(m-2) < r_i(m+1) (assuming r_ij are sorted: r_i(j-1) <=
            r_ij <= r_i(j+1))

            This formula is equivalent to determining a sphere of radius
            R_i_m, such that the sum of all solid angles of inner atoms is 4π.
            The solid angle of atom j with respect to atom i, is the solid
            angle of the spherical cap of height r_ij on the sphere of radius
            R_i_m (i.e. SA_ij = 2π(1-r_ij/R_i_m)). However, at interfaces,
            it is expected that neighbors do not fill the whole space (i.e.
            the sum of solid angles should not be 4π)

            Hence we propose here an angular correction which is exact in the
            case of infinitely evenly distributed neighbors along a spherical
            cap. Indeed, assuming that a normal vector can be meaningfully
            defined at the interface, a correction is needed when the
            neighbors barycenter is far from the central atom.

            Therefore, the neighbors barycenter is computed. From this
            barycenter, one deduces the corresponding spherical cap on the
            sphere of radius R_i_m. Its solid angle is then taken instead of
            4π.

            Consequently, this angular correction assumes a spherical
            cap-like distribution of the nearest neighbors.

        Continuous:

            The continuous coordination number algorithm scales the
            contributions of each neighbor by the following weight :
            SA/SA_max (where SA is the solid angle associated with the
            corresponding neighbor, and SA_max is the maximum solid angle (
            i.e. the solid angle associated with the nearest neighbor)) The
            continuous coordination number is then the sum of all weights.

        Generalized:

            The generalized coordination number algorithm scales the
            contributions of each neighbor by the following weight :
            CN/CN_max (where CN is the coordination number associated with
            the corresponding neighbor, and CN_max is the maximum
            coordination number (i.e. the coordination number associated with
            the most coordinated neighbor)) The generalized coordination
            number is then the sum of all weights (weighted futhermore,
            or not, by the continuous algorithm).
    """

    # Conversion to numpy.ndarray
    coords = np.array(list_coords)
    if pbc:
        cell = np.array(cell_vectors)
    else:
        cell = None
    nb_atoms = None

    # Retrieve parameters and check dimension consistency
    assert ((not pbc) or (coords.shape[1] == cell.shape[0] == cell.shape[1]))

    # Explicitely add adjacent periodic images if requested
    if pbc in ('adjacent', 'full'):
        nb_atoms, coords = add_periodic_images(coords, cell, pbc)

    # Retrieve distance-sorted pairwise distances, vectors and indexes
    sorted_distances, sorted_vectors, sorted_indexes = get_sorted_distances(
        coords, pbc, nb_atoms=nb_atoms, cell=cell)

    # Retrieve number of nearest neighbors and coordination radius with SANN
    # algorithm
    asann_CNs, asann_radius = get_SANN(sorted_distances)

    # Apply angular correction if requested
    if correction == 'ASANN':
        asann_CNs, asann_radius, vectors = get_ASANN(sorted_distances,
                                                     sorted_vectors, asann_CNs,
                                                     asann_radius)
    elif correction == 'SC-ASANN':
        asann_CNs, asann_radius, vectors = get_self_consistent_ASANN(
            sorted_distances, sorted_vectors, asann_CNs)

    # Compute edges
    if edges:
        asann_edges = get_edges(asann_CNs, sorted_indexes,
                                reduce_mode=reduce_mode, nb_atoms=nb_atoms)
    else:
        asann_edges = None

    # Convert coordination numbers into continuous values if requested
    if continuous:
        # Compute continuous CN by weighting each neighbor contribution
        asann_CNs, list_weights = convert_continuous(asann_CNs, asann_radius,
                                                     sorted_distances)
    elif generalized:
        list_weights = [[1] * asann_CN for asann_CN in asann_CNs]

    # Compute generalized coordination numbers if requested
    if generalized:
        asann_CNs = get_generalized(asann_CNs, list_weights, sorted_indexes)

    return asann_CNs, asann_radius, asann_edges, vectors


# Program being executed when used as a script (instead of a module)
if __name__ == '__main__':
    # Imports
    import sys

    # Import local structure reader sub-module
    try:
        from structure_reader import structure_from_file
    except ImportError as err:
        print(
            'Unable to find file structure_reader.py, which allows reading '
            'molecular structures. Aborting.',
            file=sys.stderr)
        print(
            'Plase copy structure_reader.py in either the same folder '
            'containing this script ({}), or in your working '
            'directory'.format(
                sys.argv[0]), file=sys.stderr)

        raise err

    # Retrieve filename of molecular structure to treat
    try:
        filename = sys.argv[1]
    except IndexError as err:
        print(
            'Unable to parse a filename to treat (containing coordinates in '
            'supported format: XYZ, POSCAR, CIF, ...)',
            file=sys.stderr)
        print('Usage: {} filename'.format(sys.argv[0]), file=sys.stderr)
        raise err

    # Read file structure
    file_structure = structure_from_file(filename)
    coordinates = file_structure.get_coords()
    cell_vectors = file_structure.get_cell_matrix()
    pbc_mode = file_structure.pbc_enabled

    # Explicitely add next periodic images if number of atoms is too low. A
    # simple modulo operation is not enough when a single point is found as
    # neighbor more than once (as part of periodic images)
    if pbc_mode and len(coordinates) < 100:
        pbc_mode = 'full'  # Explictely include all 27 next periodic cells
    elif pbc_mode and len(coordinates) < 1000:
        pbc_mode = 'adjacent'  # Explicitely include all 9 adjacent cells

    asann_CNs, asann_radii, asann_edges, vectors = coordination_numbers(
        coordinates, pbc=pbc_mode, cell_vectors=cell_vectors, continuous=False,
        generalized=False, edges=True, correction='ASANN')

    print('ASANN vectors')
    for l in vectors:
        print(-l[0], -l[1], -l[2])

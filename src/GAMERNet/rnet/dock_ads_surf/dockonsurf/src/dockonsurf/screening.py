import logging
import numpy as np
import ase

logger = logging.getLogger('DockOnSurf')


def select_confs(conf_list: list, magns: list, num_sel: int):
    """Takes a list ase.Atoms and selects the most different magnitude-wise.

    Given a list of ase.Atoms objects and a list of magnitudes, it selects a
    number of the most different conformers according to every magnitude
    specified.
     
    @param conf_list: list of ase.Atoms objects to select among.
    @param magns: list of str with the names of the magnitudes to use for the
        conformer selection. Supported magnitudes: 'energy', 'moi'.
    @param num_sel: number of conformers to select for every of the magnitudes.
    @return: list of the selected ase.Atoms objects.
    """
    selected_ids = []
    if num_sel >= len(conf_list):
        logger.warning('Number of conformers per magnitude is equal or larger '
                       'than the total number of conformers. Using all '
                       f'available conformers: {len(conf_list)}.')
        return conf_list

    # Assign mois
    if 'moi' in magns:
        for conf in conf_list:
            conf.info["moi"] = conf.get_moments_of_inertia().sum()

    # pick ids
    for magn in magns:
        sorted_list = sorted(conf_list, key=lambda conf: abs(conf.info[magn]))
        if sorted_list[-1].info['iso'] not in selected_ids:
            selected_ids.append(sorted_list[-1].info['iso'])
        if num_sel > 1:
            for i in range(0, len(sorted_list) - 1,
                           len(conf_list) // (num_sel - 1)):
                if sorted_list[i].info['iso'] not in selected_ids:
                    selected_ids.append(sorted_list[i].info['iso'])

    logger.info(f'Selected {len(selected_ids)} conformers for adsorption.')
    return [conf for conf in conf_list if conf.info["iso"] in selected_ids]


def get_vect_angle(v1: list, v2: list, ref=None, degrees=True):
    """Computes the angle between two vectors.

    @param v1: The first vector.
    @param v2: The second vector.
    @param ref: Orthogonal vector to both v1 and v2,
        along which the sign of the rotation is defined (i.e. positive if
        counterclockwise angle when facing ref)
    @param degrees: Whether the result should be in radians (True) or in
        degrees (False).
    @return: The angle in radians if degrees = False, or in degrees if
        degrees =True
    """
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    if ref is not None:
        # Give sign according to ref direction
        angle *= (1 if np.dot(np.cross(v1, v2), ref) >= 0 else -1)

    return angle if not degrees else angle * 180 / np.pi


def vect_avg(vects):
    """Computes the element-wise mean of a set of vectors.

    @param vects: list of lists-like: containing the vectors (num_vectors,
        length_vector).
    @return: vector average computed doing the element-wise mean.
    """
    from src.dockonsurf.utilities import try_command
    err = "vect_avg parameter vects must be a list-like, able to be converted" \
          " np.array"
    array = try_command(np.array, [(ValueError, err)], vects)
    if len(array.shape) == 1:
        return array
    else:
        num_vects = array.shape[1]
        return np.array([np.average(array[:, i]) for i in range(num_vects)])


def get_atom_coords(atoms: ase.Atoms, center=None):
    """Gets the coordinates of the specified center for an ase.Atoms object.

    If center is not an index but a list of indices, it computes the
    element-wise mean of the coordinates of the atoms specified in the inner
    list.
    @param atoms: ase.Atoms object for which to obtain the coordinates of.
    @param center: index/list of indices of the atoms for which the coordinates
                   should be extracted.
    @return: np.ndarray of atomic coordinates.
    """
    err_msg = "Argument 'ctr' must be an integer or a list of integers. "\
              "Every integer must be in the range [0, num_atoms)"
    if center is None:
        center = list(range(len(atoms)))
    if isinstance(center, int):
        if center not in list(range(len(atoms))):
            logger.error(err_msg)
            raise ValueError(err_msg)
        return atoms[center].position
    elif isinstance(center, list):
        for elm in center:
            if elm not in list(range(len(atoms))):
                logger.error(err_msg)
                raise ValueError(err_msg)
        return vect_avg([atoms[idx].position for idx in center])
    else:
        logger.error(err_msg)
        raise ValueError(err_msg)


def compute_norm_vect(atoms, idxs, cell):
    """Computes the local normal vector of a surface at a given site.

    Given an ase.Atoms object and a site defined as a linear combination of
    atoms it computes the vector perpendicular to the surface, considering the
    local environment of the site.
    @param atoms: ase.Atoms object of the surface.
    @param idxs: list or int: Index or list of indices of the atom/s that define
        the site
    @param cell: Unit cell. A 3x3 matrix (the three unit cell vectors)
    @return: numpy.ndarray of the coordinates of the vector locally
    perpendicular to the surface.
    """
    from src.dockonsurf.ASANN import coordination_numbers as coord_nums
    if isinstance(idxs, list):
        atm_vect = [-np.round(coord_nums(atoms.get_scaled_positions(),
                                         pbc=np.any(cell),
                                         cell_vectors=cell)[3][i], 2)
                    for i in idxs]
        norm_vec = vect_avg([vect / np.linalg.norm(vect) for vect in atm_vect])
    elif isinstance(idxs, int):
        norm_vec = -coord_nums(atoms.get_scaled_positions(),
                               pbc=np.any(cell),
                               cell_vectors=cell)[3][idxs]
    else:
        err = "'idxs' must be either an int or a list"
        logger.error(err)
        raise ValueError(err)
    norm_vec = np.round(norm_vec, 2) / np.linalg.norm(np.round(norm_vec, 2))
    if not np.isnan(norm_vec).any():
        logger.info(f"The perpendicular vector to the surface at site '{idxs}' "
                    f"is {norm_vec}")
    return norm_vec


def align_molec(orig_molec, ctr_coord, ref_vect):
    """Align a molecule to a vector by a center.

    Given a reference vector to be aligned to and some coordinates acting as
    alignment center, it first averages the vectors pointing to neighboring
    atoms and then tries to align this average vector to the target. If the
    average vector is 0 it takes the vector to the nearest neighbor.
    @param orig_molec: The molecule to align.
    @param ctr_coord: The coordinates to use ase alignment center.
    @param ref_vect: The vector to be aligned with.
    @return: ase.Atoms of the aligned molecule.
    """
    from copy import deepcopy
    from ase import Atom
    from ase.neighborlist import natural_cutoffs, neighbor_list

    molec = deepcopy(orig_molec)
    if len(molec) == 1:
        err_msg = "Cannot align a single atom"
        logger.error(err_msg)
        ValueError(err_msg)
    cutoffs = natural_cutoffs(molec, mult=1.2)
    # Check if ctr_coord are the coordinates of an atom and if not creates a
    # dummy one to extract the neighboring atoms.
    ctr_idx = None
    dummy_atom = False
    for atom in molec:
        if np.allclose(ctr_coord, atom.position, rtol=1e-2):
            ctr_idx = atom.index
            break
    if ctr_idx is None:
        molec.append(Atom('X', position=ctr_coord))
        cutoffs.append(0.2)
        ctr_idx = len(molec) - 1
        dummy_atom = True
    # Builds the neighbors and computes the average vector
    refs, vects = neighbor_list("iD", molec, cutoffs, self_interaction=False)
    neigh_vects = [vects[i] for i, atm in enumerate(refs) if atm == ctr_idx]
    # If no neighbors are present, the cutoff of the alignment center is
    # set to a value where at least one atom is a neighbor and neighbors are
    # recalculated.
    if len(neigh_vects) == 0:
        min_dist, min_idx = (np.inf, np.inf)
        for atom in molec:
            if atom.index == ctr_idx:
                continue
            if molec.get_distance(ctr_idx, atom.index) < min_dist:
                min_dist = molec.get_distance(ctr_idx, atom.index)
                min_idx = atom.index
        cutoffs[ctr_idx] = min_dist - cutoffs[min_idx] + 0.05
        refs, vects = neighbor_list("iD", molec, cutoffs,
                                    self_interaction=False)
        neigh_vects = [vects[i] for i, atm in enumerate(refs) if atm == ctr_idx]
    target_vect = vect_avg(neigh_vects)
    # If the target vector is 0 (the center is at the baricenter of its
    # neighbors). Assuming the adsorption center is coplanar or colinear to its
    # neighbors (it would not make a lot of sense to chose a center which is
    # the baricenter of neighbors distributed in 3D), the target_vector is
    # chosen perpendicular to the nearest neighbor.
    if np.allclose(target_vect, 0, rtol=1e-3):
        nn_vect = np.array([np.inf] * 3)
        for vect in neigh_vects:
            if np.linalg.norm(vect) < np.linalg.norm(nn_vect):
                nn_vect = vect
        cart_axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        axis = cart_axes[int(np.argmax([np.linalg.norm(np.cross(ax, nn_vect))
                                        for ax in cart_axes]))]
        target_vect = np.cross(axis, nn_vect)

    rot_vect = np.cross(target_vect, ref_vect)
    if np.allclose(rot_vect, 0):
        cart_axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        axis = cart_axes[int(np.argmax([np.linalg.norm(np.cross(ax, ref_vect))
                                        for ax in cart_axes]))]
        rot_vect = np.cross(ref_vect, axis)
    rot_angle = -get_vect_angle(ref_vect, target_vect, rot_vect)
    molec.rotate(rot_angle, rot_vect, ctr_coord)
    if dummy_atom:
        del molec[-1]
    return molec


def add_adsorbate(slab, adsorbate, site_coord, ctr_coord, height, offset=None,
                  norm_vect=(0, 0, 1)):
    """Add an adsorbate to a surface.

    This function extends the functionality of ase.build.add_adsorbate
    (https://wiki.fysik.dtu.dk/ase/ase/build/surface.html#ase.build.add_adsorbate)
    by enabling to change the z coordinate and the axis perpendicular to the
    surface.
    @param slab: ase.Atoms object containing the coordinates of the surface
    @param adsorbate: ase.Atoms object containing the coordinates of the
        adsorbate.
    @param site_coord: The coordinates of the adsorption site on the surface.
    @param ctr_coord: The coordinates of the adsorption center in the molecule.
    @param height: The height above the surface where to adsorb.
    @param offset: Offsets the adsorbate by a number of unit cells. Mostly
        useful when adding more than one adsorbate.
    @param norm_vect: The vector perpendicular to the surface.
    """
    from copy import deepcopy
    info = slab.info.get('adsorbate_info', {})
    pos = np.array([0.0, 0.0, 0.0])  # part of absolute coordinates
    spos = np.array([0.0, 0.0, 0.0])  # part relative to unit cell
    norm_vect_u = np.array(norm_vect) / np.linalg.norm(norm_vect)
    if offset is not None:
        spos += np.asarray(offset, float)
    if isinstance(site_coord, str):
        # A site-name:
        if 'sites' not in info:
            raise TypeError('If the atoms are not made by an ase.build '
                            'function, position cannot be a name.')
        if site_coord not in info['sites']:
            raise TypeError('Adsorption site %s not supported.' % site_coord)
        spos += info['sites'][site_coord]
    else:
        pos += site_coord
    if 'cell' in info:
        cell = info['cell']
    else:
        cell = slab.get_cell()
    pos += np.dot(spos, cell)
    # Convert the adsorbate to an Atoms object
    if isinstance(adsorbate, ase.Atoms):
        ads = deepcopy(adsorbate)
    elif isinstance(adsorbate, ase.Atom):
        ads = ase.Atoms([adsorbate])
    else:
        # Assume it is a string representing a single Atom
        ads = ase.Atoms([ase.Atom(adsorbate)])
    pos += height * norm_vect_u
    # Move adsorbate into position
    ads.translate(pos - ctr_coord)
    # Attach the adsorbate
    slab.extend(ads)


def check_collision(slab_molec, slab_num_atoms, min_height, vect, nn_slab=0,
                    nn_molec=0, coll_coeff=1.2, exclude_atom=False):
    """Checks whether a slab and a molecule collide or not.

    @param slab_molec: The system of adsorbate-slab for which to detect if there
        are collisions.
    @param nn_slab: Number of neigbors in the surface.
    @param nn_molec: Number of neighbors in the molecule.
    @param coll_coeff: The coefficient that multiplies the covalent radius of
        atoms resulting in a distance that two atoms being closer to that is
        considered as atomic collision.
    @param slab_num_atoms: Number of atoms of the bare slab.
    @param min_height: The minimum height atoms can have to not be considered as
        colliding.
    @param vect: The vector perpendicular to the slab.
    @param exclude_atom: Whether to exclude the adsorption center in the
        molecule in the collision detection.
    @return: bool, whether the surface and the molecule collide.
    """
    from copy import deepcopy
    from ase.neighborlist import natural_cutoffs, neighbor_list

    normal = 0
    for i, coord in enumerate(vect):
        if coord == 0:
            continue
        normal = i
    if vect[normal] > 0:
        surf_height = max(slab_molec[:slab_num_atoms].positions[:, normal])
    else:
        surf_height = min(slab_molec[:slab_num_atoms].positions[:, normal])

    # Check structure overlap by height
    if min_height is not False:
        cart_axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                     [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
        if vect.tolist() not in cart_axes:
            err_msg = "'min_coll_height' option is only implemented for " \
                      "'surf_norm_vect' to be one of the x, y or z axes. "
            logger.error(err_msg)
            raise ValueError(err_msg)
        for atom in slab_molec[slab_num_atoms:]:
            if exclude_atom is not False \
                    and atom.index == exclude_atom:
                continue
            for i, coord in enumerate(vect):
                if coord == 0:
                    continue
                if (atom.position[i] - surf_height) * coord < min_height:
                    return True

    # Check structure overlap by sphere collision
    if coll_coeff is not False:
        if exclude_atom is not False:
            slab_molec_wo_ctr = deepcopy(slab_molec)
            del slab_molec_wo_ctr[exclude_atom + slab_num_atoms]
            slab_molec_cutoffs = natural_cutoffs(slab_molec_wo_ctr,
                                                 mult=coll_coeff)
            slab_molec_nghbs = len(neighbor_list("i", slab_molec_wo_ctr,
                                                 slab_molec_cutoffs))
        else:
            slab_molec_cutoffs = natural_cutoffs(slab_molec, mult=coll_coeff)
            slab_molec_nghbs = len(neighbor_list("i", slab_molec,
                                                 slab_molec_cutoffs))
        if slab_molec_nghbs > nn_slab + nn_molec:
            return True

    return False


def correct_coll(molec, slab, ctr_coord, site_coord, num_pts,
                 min_coll_height, norm_vect, slab_nghbs, molec_nghbs,
                 coll_coeff, height=2.5, excl_atom=False):
    # TODO Rename this function
    """Tries to adsorb a molecule on a slab trying to avoid collisions by doing
    small rotations.

    @param molec: ase.Atoms object of the molecule to adsorb
    @param slab: ase.Atoms object of the surface on which to adsorb the
        molecule
    @param ctr_coord: The coordinates of the molecule to use as adsorption
        center.
    @param site_coord: The coordinates of the surface on which to adsorb the
        molecule
    @param num_pts: Number on which to sample Euler angles.
    @param min_coll_height: The lowermost height for which to detect a collision
    @param norm_vect: The vector perpendicular to the surface.
    @param slab_nghbs: Number of neigbors in the surface.
    @param molec_nghbs: Number of neighbors in the molecule.
    @param coll_coeff: The coefficient that multiplies the covalent radius of
        atoms resulting in a distance that two atoms being closer to that is
        considered as atomic collision.
    @param height: Height on which to try adsorption.
    @param excl_atom: Whether to exclude the adsorption center in the
        molecule in the collision detection.
    @return collision: bool, whether the structure generated has collisions
        between slab and adsorbate.
    """
    from copy import deepcopy
    slab_num_atoms = len(slab)
    slab_molec = []
    collision = True
    max_corr = 6  # Should be an even number
    d_angle = 180 / ((max_corr / 2.0) * num_pts)
    num_corr = 0
    while collision and num_corr <= max_corr:
        k = num_corr * (-1) ** num_corr
        slab_molec = deepcopy(slab)
        molec.euler_rotate(k * d_angle, k * d_angle / 2, k * d_angle,
                           center=ctr_coord)
        add_adsorbate(slab_molec, molec, site_coord, ctr_coord, height,
                      norm_vect=norm_vect)
        collision = check_collision(slab_molec, slab_num_atoms, min_coll_height,
                                    norm_vect, slab_nghbs, molec_nghbs,
                                    coll_coeff, excl_atom)
        num_corr += 1
    return slab_molec, collision


def dissociate_h(slab_molec_orig, h_idx, num_atoms_slab, h_acceptor,
                 neigh_cutoff=1):
    # TODO rethink
    """Tries to dissociate a H from the molecule and adsorbs it on the slab.

    Tries to dissociate a H atom from the molecule and adsorb in on top of the
    surface if the distance is shorter than two times the neigh_cutoff value.
    @param slab_molec_orig: The ase.Atoms object of the system adsorbate-slab.
    @param h_idx: The index of the hydrogen atom to carry out adsorption of.
    @param num_atoms_slab: The number of atoms of the slab without adsorbate.
    @param h_acceptor: List of atom types or atom numbers that are H-acceptors.
    @param neigh_cutoff: half the maximum distance between the surface and the
        H for it to carry out dissociation.
    @return: An ase.Atoms object of the system adsorbate-surface with H
    """
    from copy import deepcopy
    from ase.neighborlist import NeighborList
    slab_molec = deepcopy(slab_molec_orig)
    cutoffs = len(slab_molec) * [neigh_cutoff]
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True, skin=0.0)
    nl.update(slab_molec)
    surf_h_vect = np.array([np.infty] * 3)
    if h_acceptor == 'all':
        h_acceptor = list(range(num_atoms_slab))
    for neigh_idx in nl.get_neighbors(h_idx)[0]:
        for elm in h_acceptor:
            if isinstance(elm, int):
                if neigh_idx == elm and neigh_idx < num_atoms_slab:
                    dist = np.linalg.norm(slab_molec[neigh_idx].position -
                                          slab_molec[h_idx].position)
                    if dist < np.linalg.norm(surf_h_vect):
                        surf_h_vect = slab_molec[neigh_idx].position \
                                      - slab_molec[h_idx].position
            else:
                if slab_molec[neigh_idx].symbol == elm \
                        and neigh_idx < num_atoms_slab:
                    dist = np.linalg.norm(slab_molec[neigh_idx].position -
                                          slab_molec[h_idx].position)
                    if dist < np.linalg.norm(surf_h_vect):
                        surf_h_vect = slab_molec[neigh_idx].position \
                                      - slab_molec[h_idx].position

    if np.linalg.norm(surf_h_vect) != np.infty:
        trans_vect = surf_h_vect - surf_h_vect / np.linalg.norm(surf_h_vect)
        slab_molec[h_idx].position = slab_molec[h_idx].position + trans_vect
        return slab_molec


def dissociation(slab_molec, h_donor, h_acceptor, num_atoms_slab):
    # TODO multiple dissociation
    """Decides which H atoms to dissociate according to a list of atoms.

    Given a list of chemical symbols or atom indices it checks for every atom
    or any of its neighbor if it's a H and calls dissociate_h to try to carry
    out dissociation of that H. For atom indices, it checks both whether
    the atom index or its neighbors are H, for chemical symbols, it only checks
    if there is a neighbor H.
    @param slab_molec: The ase.Atoms object of the system adsorbate-slab.
    @param h_donor: List of atom types or atom numbers that are H-donors.
    @param h_acceptor: List of atom types or atom numbers that are H-acceptors.
    @param num_atoms_slab: Number of atoms of the bare slab.
    @return:
    """
    from ase.neighborlist import natural_cutoffs, NeighborList
    molec = slab_molec[num_atoms_slab:]
    cutoffs = natural_cutoffs(molec)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(molec)
    disso_structs = []
    for el in h_donor:
        if isinstance(el, int):
            if molec[el].symbol == 'H':
                disso_struct = dissociate_h(slab_molec, el + num_atoms_slab,
                                            num_atoms_slab, h_acceptor)
                if disso_struct is not None:
                    disso_structs.append(disso_struct)
            else:
                for neigh_idx in nl.get_neighbors(el)[0]:
                    if molec[neigh_idx].symbol == 'H':
                        disso_struct = dissociate_h(slab_molec, neigh_idx +
                                                    num_atoms_slab,
                                                    num_atoms_slab, h_acceptor)
                        if disso_struct is not None:
                            disso_structs.append(disso_struct)
        else:
            for atom in molec:
                if atom.symbol.lower() == el.lower():
                    for neigh_idx in nl.get_neighbors(atom.index)[0]:
                        if molec[neigh_idx].symbol == 'H':
                            disso_struct = dissociate_h(slab_molec, neigh_idx
                                                        + num_atoms_slab,
                                                        num_atoms_slab,
                                                        h_acceptor)
                            if disso_struct is not None:
                                disso_structs.append(disso_struct)
    return disso_structs


def ads_euler(orig_molec, slab, ctr_coord, site_coord, num_pts,
              min_coll_height, coll_coeff, norm_vect, slab_nghbs, molec_nghbs,
              h_donor, h_acceptor, height, excl_atom):
    """Generates adsorbate-surface structures by sampling over Euler angles.

    This function generates a number of adsorbate-surface structures at
    different orientations of the adsorbate sampled at multiple Euler (zxz)
    angles.
    @param orig_molec: ase.Atoms object of the molecule to adsorb.
    @param slab: ase.Atoms object of the surface on which to adsorb the
        molecule.
    @param ctr_coord: The coordinates of the molecule to use as adsorption
        center.
    @param site_coord: The coordinates of the surface on which to adsorb the
        molecule.
    @param num_pts: Number on which to sample Euler angles.
    @param min_coll_height: The lowest height for which to detect a collision.
    @param coll_coeff: The coefficient that multiplies the covalent radius of
        atoms resulting in a distance that two atoms being closer to that is
        considered as atomic collision.
    @param norm_vect: The vector perpendicular to the surface.
    @param slab_nghbs: Number of neigbors in the surface.
    @param molec_nghbs: Number of neighbors in the molecule.
    @param h_donor: List of atom types or atom numbers that are H-donors.
    @param h_acceptor: List of atom types or atom numbers that are H-acceptors.
    @param height: Height on which to try adsorption.
    @param excl_atom: Whether to exclude the adsorption center in the
        molecule in the collision detection.
    @return: list of ase.Atoms object conatining all the orientations of a given
        conformer.
    """
    from copy import deepcopy
    slab_ads_list = []
    prealigned_molec = align_molec(orig_molec, ctr_coord, norm_vect)
    # rotation around z
    for alpha in np.arange(0, 360, 360 / num_pts):
        # rotation around x'
        for beta in np.arange(0, 180, 180 / num_pts):
            # rotation around z'
            for gamma in np.arange(0, 360, 360 / num_pts):
                if beta == 0 and gamma > 0:
                    continue
                molec = deepcopy(prealigned_molec)
                molec.euler_rotate(alpha, beta, gamma, center=ctr_coord)
                slab_molec, collision = correct_coll(molec, slab, ctr_coord,
                                                     site_coord, num_pts,
                                                     min_coll_height, norm_vect,
                                                     slab_nghbs, molec_nghbs,
                                                     coll_coeff, height,
                                                     excl_atom)
                if not collision and not any([np.allclose(slab_molec.positions,
                                                          conf.positions)
                                              for conf in slab_ads_list]):
                    slab_molec.info = {**slab_molec.info, **molec.info}
                    slab_ads_list.append(slab_molec)
                    if h_donor is not False:
                        slab_ads_list.extend(dissociation(slab_molec, h_donor,
                                                          h_acceptor,
                                                          len(slab)))

    return slab_ads_list


def ads_internal(orig_molec, slab, ctr1_mol, ctr2_mol, ctr3_mol, ctr1_surf,
                 ctr2_surf, num_pts, min_coll_height, coll_coeff, norm_vect,
                 slab_nghbs, molec_nghbs, h_donor, h_acceptor, max_hel, height,
                 excl_atom):
    """Generates adsorbate-surface structures by sampling over internal angles.

    @param orig_molec: ase.Atoms object of the molecule to adsorb (adsorbate).
    @param slab: ase.Atoms object of the surface on which to adsorb the molecule
    @param ctr1_mol: The index/es of the center in the adsorbate to use as
        adsorption center.
    @param ctr2_mol: The index/es of the center in the adsorbate to use for the
        definition of the surf-adsorbate angle, surf-adsorbate dihedral angle
        and adsorbate dihedral angle.
    @param ctr3_mol: The index/es of the center in the adsorbate to use for the
        definition of the adsorbate dihedral angle.
    @param ctr1_surf: The index/es of the center in the surface to use as
        adsorption center.
    @param ctr2_surf: The index/es of the center in the surface to use for the
        definition of the surf-adsorbate dihedral angle.
    @param num_pts: Number on which to sample Euler angles.
    @param min_coll_height: The lowest height for which to detect a collision
    @param coll_coeff: The coefficient that multiplies the covalent radius of
        atoms resulting in a distance that two atoms being closer to that is
        considered as atomic collision.
    @param norm_vect: The vector perpendicular to the surface.
    @param slab_nghbs: Number of neigbors in the surface.
    @param molec_nghbs: Number of neighbors in the molecule.
    @param h_donor: List of atom types or atom numbers that are H-donors.
    @param h_acceptor: List of atom types or atom numbers that are H-acceptors.
    @param max_hel: Maximum value for sampling the helicopter
        (surf-adsorbate dihedral) angle.
    @param height: Height on which to try adsorption.
    @param excl_atom: Whether to exclude the adsorption center in the
        molecule in the collision detection.
    @return: list of ase.Atoms object conatining all the orientations of a given
        conformer.
    """
    from copy import deepcopy
    from src.dockonsurf.internal_rotate import internal_rotate
    slab_ads_list = []
    # Rotation over bond angle
    for alpha in np.arange(90, 180+1, 90 / max(1, num_pts-1))[:num_pts]:
        # Rotation over surf-adsorbate dihedral angle
        for beta in np.arange(0, max_hel, max_hel / num_pts):
            # Rotation over adsorbate bond dihedral angle
            for gamma in np.arange(90, 270+1, 180/max(1, num_pts-1))[:num_pts]:
                # Avoid duplicates as gamma rotation has no effect on plane
                # angles.
                if alpha == 180 and gamma > 90:
                    continue
                new_molec = deepcopy(orig_molec)
                internal_rotate(new_molec, slab, ctr1_mol, ctr2_mol, ctr3_mol,
                                ctr1_surf, ctr2_surf, norm_vect, alpha,
                                beta, gamma)
                site_coords = get_atom_coords(slab, ctr1_surf)
                ctr_coords = get_atom_coords(new_molec, ctr1_mol)
                slab_molec, collision = correct_coll(new_molec, slab,
                                                     ctr_coords, site_coords,
                                                     num_pts, min_coll_height,
                                                     norm_vect, slab_nghbs,
                                                     molec_nghbs, coll_coeff,
                                                     height, excl_atom)
                slab_molec.info = {**slab_molec.info, **new_molec.info}
                if not collision and \
                        not any([np.allclose(slab_molec.positions,
                                             conf.positions)
                                 for conf in slab_ads_list]):
                    slab_ads_list.append(slab_molec)
                    if h_donor is not False:
                        slab_ads_list.extend(dissociation(slab_molec, h_donor,
                                                          h_acceptor,
                                                          len(slab)))

    return slab_ads_list


def adsorb_confs(conf_list, surf, inp_vars):
    """Generates a number of adsorbate-surface structure coordinates.

    Given a list of conformers, a surface, a list of atom indices (or list of
    list of indices) of both the surface and the adsorbate, it generates a
    number of adsorbate-surface structures for every possible combination of
    them at different orientations.
    @param conf_list: list of ase.Atoms of the different conformers
    @param surf: the ase.Atoms object of the surface
    @param inp_vars: Calculation parameters from input file.
    @return: list of ase.Atoms for the adsorbate-surface structures
    """
    from copy import deepcopy
    from ase.neighborlist import natural_cutoffs, neighbor_list
    molec_ctrs = inp_vars['molec_ctrs']
    sites = inp_vars['sites']
    angles = inp_vars['set_angles']
    num_pts = inp_vars['sample_points_per_angle']
    inp_norm_vect = inp_vars['surf_norm_vect']
    min_coll_height = inp_vars['min_coll_height']
    coll_coeff = inp_vars['collision_threshold']
    exclude_ads_ctr = inp_vars['exclude_ads_ctr']
    h_donor = inp_vars['h_donor']
    h_acceptor = inp_vars['h_acceptor']
    height = inp_vars['adsorption_height']

    if inp_vars['pbc_cell'] is not False:
        surf.set_pbc(True)
        surf.set_cell(inp_vars['pbc_cell'])

    surf_ads_list = []
    sites_coords = np.array([get_atom_coords(surf, site) for site in sites])
    if coll_coeff is not False:
        surf_cutoffs = natural_cutoffs(surf, mult=coll_coeff)
        surf_nghbs = len(neighbor_list("i", surf, surf_cutoffs))
    else:
        surf_nghbs = 0
    for i, conf in enumerate(conf_list):
        molec_ctr_coords = np.array([get_atom_coords(conf, ctr)
                                     for ctr in molec_ctrs])
        if inp_vars['pbc_cell'] is not False:
            conf.set_pbc(True)
            conf.set_cell(inp_vars['pbc_cell'])
        for s, site in enumerate(sites_coords):
            if isinstance(inp_norm_vect, str) and inp_norm_vect == 'auto':
                norm_vect = compute_norm_vect(surf, sites[s],
                                              inp_vars['pbc_cell'])
                if np.isnan(norm_vect).any():
                    logger.warning(f"Could not compute the normal vector to "
                                   f"site '{sites[s]}'. Skipping site.")
                    continue
            else:
                norm_vect = inp_norm_vect
            for c, ctr in enumerate(molec_ctr_coords):
                if exclude_ads_ctr and isinstance(molec_ctrs[c], int):
                    exclude_atom = molec_ctrs[c]
                else:
                    exclude_atom = False
                    if exclude_ads_ctr and not isinstance(molec_ctrs[c], int):
                        logger.warning("'exclude_ads_ctr' only works for atomic"
                                       "centers and not for many-atoms "
                                       f"barycenters. {molec_ctrs[c]} are not "
                                       f"going to be excluded from collison.")
                if coll_coeff and exclude_atom is not False:
                    conf_wo_ctr = deepcopy(conf)
                    del conf_wo_ctr[exclude_atom]
                    conf_cutoffs = natural_cutoffs(conf_wo_ctr, mult=coll_coeff)
                    molec_nghbs = len(neighbor_list("i", conf_wo_ctr,
                                                    conf_cutoffs))
                elif coll_coeff and exclude_atom is False:
                    conf_cutoffs = natural_cutoffs(conf, mult=coll_coeff)
                    molec_nghbs = len(neighbor_list("i", conf, conf_cutoffs))
                else:
                    molec_nghbs = 0
                if angles == 'euler':
                    surf_ads_list.extend(ads_euler(conf, surf, ctr, site,
                                                   num_pts, min_coll_height,
                                                   coll_coeff, norm_vect,
                                                   surf_nghbs, molec_nghbs,
                                                   h_donor, h_acceptor, height,
                                                   exclude_atom))
                elif angles == 'internal':
                    mol_ctr1 = molec_ctrs[c]
                    mol_ctr2 = inp_vars["molec_ctrs2"][c]
                    mol_ctr3 = inp_vars["molec_ctrs3"][c]
                    surf_ctr1 = sites[s]
                    surf_ctr2 = inp_vars["surf_ctrs2"][s]
                    max_h = inp_vars["max_helic_angle"]
                    surf_ads_list.extend(ads_internal(conf, surf, mol_ctr1,
                                                      mol_ctr2, mol_ctr3,
                                                      surf_ctr1, surf_ctr2,
                                                      num_pts, min_coll_height,
                                                      coll_coeff, norm_vect,
                                                      surf_nghbs, molec_nghbs,
                                                      h_donor, h_acceptor,
                                                      max_h, height,
                                                      exclude_atom))
    return surf_ads_list


def run_screening(inp_vars):
    """Carries out the screening of adsorbate structures on a surface.

    @param inp_vars: Calculation parameters from input file.
    """
    import os
    import random
    from src.dockonsurf.formats import collect_confs, adapt_format
    from src.dockonsurf.calculation import run_calc, check_finished_calcs

    logger.info('Carrying out procedures for the screening of adsorbate-surface'
                ' structures.')
    if inp_vars['use_molec_file']:
        selected_confs = [adapt_format('ase', inp_vars['use_molec_file'],
                                       inp_vars['special_atoms'])]
        logger.info(f"Using '{inp_vars['use_molec_file']}' as only conformer.")
    else:
        if not os.path.isdir("isolated"):
            err = "'isolated' directory not found. It is needed in order to " \
                  "carry out the screening of structures to be adsorbed"
            logger.error(err)
            raise FileNotFoundError(err)

        finished_calcs, failed_calcs = check_finished_calcs('isolated',
                                                            inp_vars['code'])
        if not finished_calcs:
            err_msg = "No calculations on 'isolated' finished normally."
            logger.error(err_msg)
            raise FileNotFoundError(err_msg)

        logger.info(f"Found {len(finished_calcs)} structures of isolated "
                    f"conformers whose calculation finished normally.")
        if len(failed_calcs) != 0:
            logger.warning(
                f"Found {len(failed_calcs)} calculations more that "
                f"did not finish normally: {failed_calcs}. \n"
                f"Using only the ones that finished normally: "
                f"{finished_calcs}.")

        conf_list = collect_confs(finished_calcs, inp_vars['code'], 'isolated',
                                  inp_vars['special_atoms'])
        selected_confs = select_confs(conf_list, inp_vars['select_magns'],
                                      inp_vars['confs_per_magn'])
    surf = adapt_format('ase', inp_vars['surf_file'], inp_vars['special_atoms'])
    surf.info = {}
    surf_ads_list = adsorb_confs(selected_confs, surf, inp_vars)
    if len(surf_ads_list) > inp_vars['max_structures']:
        surf_ads_list = random.sample(surf_ads_list, inp_vars['max_structures'])
    # elif len(surf_ads_list) == 0:
    #     err_msg = "No configurations were generated: Try with a different " \
    #                 "combination of parameters."
    #     logger.error(err_msg)
    #     raise ValueError(err_msg)
    logger.info(f'Generated {len(surf_ads_list)} adsorbate-surface atomic '
                f'configurations to carry out a calculation of.')

    # run_calc('screening', inp_vars, surf_ads_list)
    logger.info('Finished the procedures for the screening of adsorbate-surface'
                ' structures section.')
    return surf_ads_list
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
import logging

import numpy as np

from src.dockonsurf.screening import get_atom_coords, get_vect_angle


logger = logging.getLogger('DockOnSurf')


def internal_rotate(molecule, surf, ctr1_mol, ctr2_mol, ctr3_mol, ctr1_surf,
                    ctr2_surf, bond_vector, bond_angle_target,
                    dihedral_angle_target=None, mol_dihedral_angle_target=None):
    """Performs translation and rotation of an adsorbate defined by an
    adsorption bond length, direction, angle and dihedral angle

    Carles modification of chemcat's transform_adsorbate to work with
    coordinates instead of ase.Atom
    Parameters:
        molecule (ase.Atoms): The molecule to adsorb.

        surf (ase.Atoms): The surface ontop of which to adsorb.

        ctr1_mol (int/list): The position of the adsorption center in the
        molecule that will be bound to the surface.

        ctr2_mol (int/list): The position of a second center of the
        adsorbate used to define the adsorption bond angle, and the dihedral
        adsorption angle.

        ctr3_mol (int/list): The position of a third center in the
        adsorbate used to define the adsorbate dihedral angle.

        ctr1_surf (int/list): The position of the site on the surface that
        will be bound to the molecule.

        ctr2_surf (int/list): The position of a second center of the
        surface used to define the dihedral adsorption angle.

        bond_vector (numpy.ndarray): The adsorption bond desired.
            Details: offset = vect(atom1_surf, atom1_mol)

        bond_angle_target (float or int): The adsorption bond angle desired (in
            degrees).
            Details: bond_angle_target = angle(atom1_surf, atom1_mol, atom2_mol)

        dihedral_angle_target (float or int): The dihedral adsorption angle
            desired (in degrees).
            Details: dihedral_angle_target = dihedral(atom2_surf, atom1_surf,
            atom1_mol, atom2_mol)
                The rotation vector is facing the adsorbate from the surface
                (i.e. counterclockwise rotation when facing the surface (i.e.
                view from top))

        mol_dihedral_angle_target (float or int): The adsorbate dihedral angle
            desired (in degrees).
            Details: mol_dihedral_angle_target = dihedral(atom1_surf, atom1_mol,
            atom2_mol, atom3_mol)
                The rotation vector is facing atom2_mol from atom1_mol

    Returns:
        None (the `molecule` object is modified in-place)
    """

    vect_surf = get_atom_coords(surf, ctr2_surf) - get_atom_coords(surf,
                                                                   ctr1_surf)
    vect_inter = get_atom_coords(molecule, ctr1_mol) \
        - get_atom_coords(surf, ctr1_surf)
    vect_mol = get_atom_coords(molecule, ctr2_mol) - get_atom_coords(molecule,
                                                                     ctr1_mol)
    vect2_mol = get_atom_coords(molecule, ctr3_mol) - get_atom_coords(molecule,
                                                                      ctr2_mol)

    # Check if dihedral angles can be defined
    do_dihedral = dihedral_angle_target is not None
    do_mol_dihedral = mol_dihedral_angle_target is not None
    dihedral_use_mol2 = False
    # Check if vect_surf and bond_vector are aligned
    if np.allclose(np.cross(vect_surf, bond_vector), 0):
        do_dihedral = False
    # Check if requested bond angle is not flat
    if np.isclose((bond_angle_target + 90) % 180 - 90, 0):
        do_mol_dihedral = False
        dihedral_use_mol2 = True
    # Check if vect_mol and vect2_mol are not aligned
    if np.allclose(np.cross(vect_mol, vect2_mol), 0):
        do_mol_dihedral = False

    ###########################
    #       Translation       #
    ###########################

    # Compute and apply translation of adsorbate
    translation = bond_vector - vect_inter
    molecule.translate(translation)

    # Update adsorption bond
    vect_inter = get_atom_coords(molecule, ctr1_mol) - \
        get_atom_coords(surf, ctr1_surf)

    # Check if translation was successful
    if np.allclose(vect_inter, bond_vector):
        pass  # print("Translation successfully applied (error: ~ {:.5g} unit "
        # "length)".format(np.linalg.norm(vect_inter - bond_vector)))
    else:
        err = 'An unknown error occured during the translation'
        logger.error(err)
        raise AssertionError(err)

    ###########################
    #   Bond angle rotation   #
    ###########################

    # Compute rotation vector
    rotation_vector = np.cross(-vect_inter, vect_mol)
    if np.allclose(rotation_vector, 0, atol=1e-5):
        # If molecular bonds are aligned, any vector orthogonal to vect_inter
        # can be used Such vector can be found as the orthogonal rejection of
        # either X-axis, Y-axis or Z-axis onto vect_inter (since they cannot
        # be all aligned)
        non_aligned_vector = np.zeros(3)
        # Select the most orthogonal axis (lowest dot product):
        non_aligned_vector[np.argmin(np.abs(vect_inter))] = 1
        rotation_vector = non_aligned_vector - np.dot(non_aligned_vector,
                                                      vect_inter) / np.dot(
            vect_inter, vect_inter) * vect_inter

    # Retrieve current bond angle
    bond_angle_ini = get_vect_angle(-vect_inter, vect_mol, rotation_vector)

    # Apply rotation to reach desired bond_angle
    molecule.rotate(bond_angle_target - bond_angle_ini, v=rotation_vector,
                    center=get_atom_coords(molecule, ctr1_mol))

    # Update molecular bonds
    vect_mol = get_atom_coords(molecule, ctr2_mol) - get_atom_coords(molecule,
                                                                     ctr1_mol)
    vect2_mol = get_atom_coords(molecule, ctr3_mol) - get_atom_coords(molecule,
                                                                      ctr2_mol)

    # Check if rotation was successful
    bond_angle = get_vect_angle(-vect_inter, vect_mol)
    if np.isclose((bond_angle - bond_angle_target + 90) % 180 - 90, 0,
                  atol=1e-3) and np.allclose(get_atom_coords(molecule, ctr1_mol)
                                             - get_atom_coords(surf,
                                                               ctr1_surf),
                                             vect_inter):
        pass  # print("Rotation successfully applied (error: {:.5f}°)".format(
        # (bond_angle - bond_angle_target + 90) % 180 - 90))
    else:
        err = 'An unknown error occured during the rotation'
        logger.error(err)
        raise AssertionError(err)

    ###########################
    # Dihedral angle rotation #
    ###########################

    # Perform dihedral rotation if possible
    if do_dihedral:
        # Retrieve current dihedral angle (by computing the angle between the
        # vector rejection of vect_surf and vect_mol onto vect_inter)
        vect_inter_inner = np.dot(vect_inter, vect_inter)
        vect_surf_reject = vect_surf - np.dot(vect_surf, vect_inter) / \
            vect_inter_inner * vect_inter
        if dihedral_use_mol2:
            vect_mol_reject = vect2_mol - np.dot(vect2_mol, vect_inter) / \
                              vect_inter_inner * vect_inter
        else:
            vect_mol_reject = vect_mol - np.dot(vect_mol, vect_inter) / \
                              vect_inter_inner * vect_inter
        dihedral_angle_ini = get_vect_angle(vect_surf_reject, vect_mol_reject,
                                            vect_inter)

        # Apply dihedral rotation along vect_inter
        molecule.rotate(dihedral_angle_target - dihedral_angle_ini,
                        v=vect_inter, center=get_atom_coords(molecule,
                                                             ctr1_mol))

        # Update molecular bonds
        vect_mol = get_atom_coords(molecule, ctr2_mol) - \
            get_atom_coords(molecule, ctr1_mol)
        vect2_mol = get_atom_coords(molecule, ctr3_mol) - \
            get_atom_coords(molecule, ctr2_mol)

        # Check if rotation was successful
        # Check dihedral rotation
        if dihedral_use_mol2:
            vect_mol_reject = vect2_mol - np.dot(vect2_mol, vect_inter) / \
                              vect_inter_inner * vect_inter
        else:
            vect_mol_reject = vect_mol - np.dot(vect_mol, vect_inter) / \
                              vect_inter_inner * vect_inter
        dihedral_angle = get_vect_angle(vect_surf_reject, vect_mol_reject,
                                        vect_inter)
        # Check bond rotation is unmodified
        bond_angle = get_vect_angle(-vect_inter, vect_mol)
        if np.isclose((dihedral_angle - dihedral_angle_target + 90) % 180 - 90,
                      0, atol=1e-3) \
                and np.isclose((bond_angle - bond_angle_target + 90) %
                               180 - 90, 0, atol=1e-5) \
                and np.allclose(get_atom_coords(molecule, ctr1_mol)
                                - get_atom_coords(surf, ctr1_surf),
                                vect_inter):
            pass  # print( "Dihedral rotation successfully applied (error: {
            # :.5f}°)".format((dihedral_angle - dihedral_angle_target + 90) %
            # 180 - 90))
        else:
            err = 'An unknown error occured during the dihedral rotation'
            logger.error(err)
            raise AssertionError(err)

    #####################################
    # Adsorbate dihedral angle rotation #
    #####################################

    # Perform adsorbate dihedral rotation if possible
    if do_mol_dihedral:
        # Retrieve current adsorbate dihedral angle (by computing the angle
        # between the orthogonal rejection of vect_inter and vect2_mol onto
        # vect_mol)
        vect_mol_inner = np.dot(vect_mol, vect_mol)
        bond_inter_reject = -vect_inter - np.dot(-vect_inter, vect_mol) / \
            vect_mol_inner * vect_mol
        bond2_mol_reject = vect2_mol - np.dot(vect2_mol, vect_mol) / \
            vect_mol_inner * vect_mol
        dihedral_angle_ini = get_vect_angle(bond_inter_reject,
                                            bond2_mol_reject, vect_mol)

        # Apply dihedral rotation along vect_mol
        molecule.rotate(mol_dihedral_angle_target - dihedral_angle_ini,
                        v=vect_mol, center=get_atom_coords(molecule, ctr1_mol))

        # Update molecular bonds
        vect_mol = get_atom_coords(molecule, ctr2_mol) \
            - get_atom_coords(molecule, ctr1_mol)
        vect2_mol = get_atom_coords(molecule, ctr3_mol) \
            - get_atom_coords(molecule, ctr2_mol)

        # Check if rotation was successful
        # Check adsorbate dihedral rotation
        vect_mol_inner = np.dot(vect_mol, vect_mol)
        bond2_mol_reject = vect2_mol - np.dot(vect2_mol, vect_mol) / \
            vect_mol_inner * vect_mol
        mol_dihedral_angle = get_vect_angle(bond_inter_reject,
                                            bond2_mol_reject, vect_mol)
        # Check dihedral rotation
        vect_inter_inner = np.dot(vect_inter, vect_inter)
        vect_surf_reject = vect_surf - np.dot(vect_surf, vect_inter) / \
            vect_inter_inner * vect_inter
        vect_mol_reject = vect_mol - np.dot(vect_mol, vect_inter) / \
            vect_inter_inner * vect_inter
        dihedral_angle = get_vect_angle(vect_surf_reject, vect_mol_reject,
                                        vect_inter)
        # Check bond rotation is unmodified
        bond_angle = get_vect_angle(-vect_inter, vect_mol)
        if np.isclose((mol_dihedral_angle - mol_dihedral_angle_target + 90) %
                      180 - 90, 0, atol=1e-3) \
                and np.isclose((dihedral_angle -
                                dihedral_angle_target + 90) % 180 - 90, 0,
                               atol=1e-5) \
                and np.isclose((bond_angle - bond_angle_target + 90) % 180 - 90,
                               0, atol=1e-5) \
                and np.allclose(get_atom_coords(molecule, ctr1_mol) -
                                get_atom_coords(surf, ctr1_surf),
                                vect_inter):
            pass  # print(
            # "Adsorbate dihedral rotation successfully applied (error:
            # {:.5f}°)".format((mol_dihedral_angle - mol_dihedral_angle_target
            # + 90) % 180 - 90))
        else:
            err = 'An unknown error occured during the adsorbate dihedral ' \
                  'rotation'
            logger.error(err)
            raise AssertionError(err)

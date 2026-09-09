"""
The s4 mosaic crystal base class (optical element and beamline element).
"""
import numpy

from syned.beamline.element_coordinates import ElementCoordinates
from syned.beamline.optical_elements.crystals.crystal import Crystal, DiffractionGeometry
from syned.beamline.shape import Rectangle, Ellipse

from shadow4.beam.s4_beam import S4Beam
from shadow4.beamline.s4_beamline_element import S4BeamlineElement
from shadow4.beamline.s4_beamline_element_movements import S4BeamlineElementMovements
from shadow4.tools.arrayofvectors import vector_modulus, vector_dot, vector_cross, vector_norm, vector_rotate_around_axis, vector_reflection
from shadow4.tools.logger import is_verbose, is_debug

from crystalpy.diffraction.DiffractionSetupDabax import DiffractionSetupDabax
from crystalpy.diffraction.GeometryType import BraggDiffraction

from dabax.dabax_xraylib import DabaxXraylib

import scipy.constants as codata


class S4MosaicCrystal(Crystal):
    """
    Shadow4 Mosaic Crystal Class
    This is a base class for mosaic crystal in reflection geometry (Bragg), using the diffracted beam.

    Use derived classes for plane or other curved crystal surfaces.

    Use other classes for (to be developed):
        * S4TransmissionCrystal : Perfect crystal in transmission (Bragg-transmitted beam, Laue-diffracted and Laue-transmited)
        * S4JohanssonCrystal : Johanssong curved mosaic crystals (in Bragg reflection).

    Constructor.

    Parameters
    ----------
    name :  str, optional
        A name for the crystal
    boundary_shape : instance of BoundaryShape, optional
        The information on the crystal boundaries.
    surface_shape : instance of SurfaceShape, optional
        The information on crystal surface.
    material : str, optional
        The crystal material name (a name accepted by crystalpy).
    miller_index_h : int, optional
        The Miller index H.
    miller_index_k : int, optional
        The Miller index K.
    miller_index_l : int, optional
        The Miller index L.
    thickness : float, optional
        For is_thick=0, the crystal thickness in m.
    f_central : int, optional
        Flag for autosetting the crystal to the corrected Bragg angle.
    f_phot_cent : int, optional
        0: setting photon energy in eV, 1:setting photon wavelength in A.
    phot_cent : float, optional
        for f_central=1, the value of the photon energy (f_phot_cent=0) or photon wavelength (f_phot_cent=1).
    mosaicity_fwhm_deg : float, optional
        The crystal mosaicity (FWHM) in degrees.
    mosaicity_profile_flag : int, optional
        The distribution function of the crystallites:
        0: Gaussian,
        1: External.
    material_constants_library_flag : int, optional
        Flag for indicating the origin of the crystal data:
        0: xraylib, 1: dabax, 2: preprocessor file v1, 3: preprocessor file v2.
    file_refl : str, optional
        for material_constants_library_flag=2,3, the name of the file containing the crystal parameters.
    dabax : None or instance of DabaxXraylib,
        The pointer to the dabax library  (used for material_constants_library_flag=1).

    Returns
    -------
    instance of S4MosaicCrystal.
    """
    def __init__(self,
                 name="Undefined",
                 boundary_shape=None,
                 surface_shape=None,
                 material="graphite",
                 miller_index_h=1,
                 miller_index_k=1,
                 miller_index_l=1,
                 thickness=0.010,
                 f_central=0,
                 f_phot_cent=0,
                 phot_cent=8000.0,
                 material_constants_library_flag=0, # 0=xraylib, 1=dabax
                                                    # 2=shadow preprocessor file v1
                                                    # 3=shadow preprocessor file v2
                 file_refl="",
                 dabax=None,
                 mosaicity_fwhm_deg=0.4,
                 mosaicity_profile_flag=0,  # 0=Gaussian, 1=External
                 ):


        Crystal.__init__(self,
                         name=name,
                         surface_shape=surface_shape,
                         boundary_shape=boundary_shape,
                         material=material,
                         diffraction_geometry=DiffractionGeometry.BRAGG,
                         miller_index_h=miller_index_h,
                         miller_index_k=miller_index_k,
                         miller_index_l=miller_index_l,
                         thickness=thickness,
                        )


        self._f_central = f_central
        self._f_phot_cent = f_phot_cent
        self._phot_cent = phot_cent
        self._material_constants_library_flag = material_constants_library_flag
        self._file_refl = file_refl

        self._dabax = dabax

        # support text containg name of variable, help text and unit. Will be stored in self._support_dictionary
        self._mosaicity_fwhm_deg = mosaicity_fwhm_deg
        self._mosaicity_profile_flag = mosaicity_profile_flag

        self._add_support_text([
                    ("f_central",           "S4: autotuning",                              ""),
                    ("f_phot_cent",         "S4: for f_central=1: tune to eV(0) or A (1)", ""),
                    ("phot_cent",           "S4: for f_central=1: value in eV or A",       ""),
                    ("material_constants_library_flag", "S4: crystal data from: 0=xraylib, 1=dabax, 2=file v1, 3=file v1", ""),
                    ("file_refl",           "S4: preprocessor file name",                  ""),
                    ("mosaicity_fwhm_deg", "Mosaicity fwhm", "deg"),
                    ("mosaicity_profile_flag", "Mosaic distribution profile 0=Gaussian, 1=External", ""),
            ] )


    def get_info(self):
        """
        Returns the specific information of the S4 crystal optical element.

        Returns
        -------
        str
        """
        txt = "\n\n"
        txt += "MOSAICCRYSTAL\n"
        if self._material_constants_library_flag == 0:
            txt += "Crystal data using xraylib for %s %d%d%d\n" % (self._material,
                                                                   self._miller_index_h,
                                                                   self._miller_index_k,
                                                                   self._miller_index_l)
        elif self._material_constants_library_flag == 1:
            txt += "Crystal data using dabax for %s %d%d%d\n" % (self._material,
                                                                   self._miller_index_h,
                                                                   self._miller_index_k,
                                                                   self._miller_index_l)
        elif self._material_constants_library_flag == 2:
           txt += "Crystal data using preprocessor (bragg V1) file: %s \n" % self._file_refl
        elif self._material_constants_library_flag == 3:
           txt += "Crystal data using preprocessor (bragg V2) file: %s \n" % self._file_refl

        if self._f_central == 0:
            txt += "Using EXTERNAL incidence and reflection angles.\n"
        else:
            txt += "Using INTERNAL or calculated incidence and reflection angles for "
            if self._f_phot_cent == 0:
                txt += "photon energy %.6f eV\n" % self._phot_cent
            else:
                txt += "photon wavelength %f A\n" % (self._phot_cent)


        txt += "\n"
        ss = self.get_surface_shape()
        if ss is None:
            txt += "Surface shape is: Plane (** UNDEFINED?? **)\n"
        else:
            txt += "Surface shape is: %s\n" % ss.__class__.__name__

        #
        if ss is not None: txt += "\nParameters:\n %s\n" % ss.info()

        txt += self.get_optical_surface_instance().info() + "\n"

        boundary = self.get_boundary_shape()
        if boundary is None:
            txt += "Surface boundaries not considered (infinite)"
        else:
            txt += "Surface boundaries are: %s\n" % boundary.__class__.__name__
            txt += "    Limits: " + repr( boundary.get_boundaries()) + "\n"
            txt += boundary.info()

        return txt

    def to_python_code_boundary_shape(self):
        """
        Creates a code block with information of boundary shape.

        Returns
        -------
        str
            The text with the code.
        """
        txt = "" # "\nfrom shadow4.beamline.optical_elements.mirrors.s4_plane_mirror import S4PlaneMirror"
        bs = self._boundary_shape
        if bs is None:
            txt += "\nboundary_shape = None"
        elif isinstance(bs, Rectangle):
            txt += "\nfrom syned.beamline.shape import Rectangle"
            txt += "\nboundary_shape = Rectangle(x_left=%g, x_right=%g, y_bottom=%g, y_top=%g)" % bs.get_boundaries()
        elif isinstance(bs, Ellipse):
            txt += "\nfrom syned.beamline.shape import Ellipse"
            txt += "\nboundary_shape = Ellipse(a_axis_min=%g, a_axis_max=%g, b_axis_min=%g, b_axis_max=%g)" % bs.get_boundaries()
        return txt

    def _get_dabax_txt(self):
        if self._material_constants_library_flag == 1:
            if isinstance(self._dabax, DabaxXraylib):
                dabax_txt = 'DabaxXraylib(file_f0="%s", file_f1f2="%s")' % (self._dabax.get_file_f0(), self._dabax.get_file_f1f2())
            else:
                dabax_txt = "DabaxXraylib()"
        else:
            dabax_txt = "None"

        return dabax_txt

class S4MosaicCrystalElement(S4BeamlineElement):
    """
    The base class for Shadow4 crystal element.
    It is made of a S4MosaicCrystal and an ElementCoordinates instance. It also includes the input beam.

    Use derived classes for plane or other curved crystal surfaces.

    Constructor.

    Parameters
    ----------
    optical_element : instance of OpticalElement, optional
        The syned optical element.
    coordinates : instance of ElementCoordinates, optional
        The syned element coordinates.
    movements : instance of S4BeamlineElementMovements, optional
        The S4 element movements.
    input_beam : instance of S4Beam, optional
        The S4 incident beam.


    Returns
    -------
    instance of S4MosaicCrystalElement.
    """

    def __init__(self,
                 optical_element : S4MosaicCrystal = None,
                 coordinates : ElementCoordinates = None,
                 movements: S4BeamlineElementMovements = None,
                 input_beam : S4Beam = None):
        super().__init__(optical_element=optical_element if optical_element is not None else S4MosaicCrystal(),
                         coordinates=coordinates if coordinates is not None else ElementCoordinates(),
                         movements=movements,
                         input_beam=input_beam)

        self._crystalpy_diffraction_setup = None


    def set_crystalpy_diffraction_setup(self):
        """
        Returns the crystalpy DiffractionSetup.

        Returns
        -------
        instance of crystalpy DiffractionSetupAbstract
        """
        oe = self.get_optical_element()
        
        self._crystalpy_diffraction_setup = DiffractionSetupDabax(
            geometry_type=BraggDiffraction(),
            crystal_name=oe._material,
            thickness=oe._thickness,  # metri
            miller_h=oe._miller_index_h,
            miller_k=oe._miller_index_k,
            miller_l=oe._miller_index_l,
            azimuthal_angle=0.0,
            dabax=oe._dabax,
        )

    def align_crystal(self):
        """
        Sets the adequate incident and reflection angles to match the tuning energy.
        """
        oe = self.get_optical_element()
        coor = self.get_coordinates()

        if oe is None:
            raise Exception("Undefined optical element")

        if oe._f_central:
            if oe._f_phot_cent == 0:
                energy = oe._phot_cent
            else:
                energy = codata.h * codata.c / codata.e * 1e2 / (oe._phot_cent * 1e-8)


            setting_angle = self._crystalpy_diffraction_setup.angleBragg(energy)
            if isinstance(setting_angle, (list, tuple, numpy.ndarray)): setting_angle = setting_angle[0]

            theta_in_grazing  = setting_angle

            if is_verbose():
                print("    align_crystal: dSpacingSI [m]: " , (self._crystalpy_diffraction_setup.dSpacingSI()))
                print("    align_crystal: Bragg angle (uncorrected) for E=%f eV is %f deg" % (energy, numpy.degrees(self._crystalpy_diffraction_setup.angleBragg(energy))))
                print("    align_crystal: angle set at %f deg" % (numpy.degrees(setting_angle)))
                print("    align_crystal: (normal) Incident   angle [deg]",  numpy.degrees(numpy.pi/2 - (theta_in_grazing ) ))
                print("    align_crystal: grazing incident angle [deg]: ", numpy.degrees(theta_in_grazing ))

                theta_out_grazing = setting_angle
                print("    align_crystal: (normal) Reflection angle [LAUE EQUATION] [deg]",  numpy.degrees(numpy.pi/2 - (theta_out_grazing) ))
                print("    align_crystal: grazing output angle [LAUE EQUATION] [deg]: ", numpy.degrees(theta_out_grazing))


            _, _, angle_azimuthal = coor.get_angles()

            coor.set_angles(angle_radial     = numpy.pi/2 - theta_in_grazing,
                            angle_radial_out = numpy.pi/2 - theta_in_grazing,
                            angle_azimuthal  = angle_azimuthal)
        else:
            if is_verbose(): print("align_crystal: nothing to align: f_central=0")

        if is_verbose(): print(coor.info())

    def trace_beam(self, **params):
        """
        Runs (ray tracing) the input beam through the element.

        Parameters
        ----------
        **params : accepted parameters, in particular:

        flag_lost_value: float
            numeric value to set in the flag column when ray is lost.

        Returns
        -------
        tuple
            (output_beam, footprint) instances of S4Beam.
        """

        if not isinstance(self.get_optical_element(), Crystal): raise Exception("Undefined Crystal")
        flag_lost_value = params.get("flag_lost_value", -1)
        change_reference_system_in = params.get("change_reference_system_in", True)
        change_reference_system_out = params.get("change_reference_system_out", True)
        print(">>>>>> change_reference_system: ", change_reference_system_in, change_reference_system_out)

        if is_verbose():
            if not change_reference_system_in:
                print("change_reference_system_in = False: skipping reference change to o.e.")
            if not change_reference_system_out:
                print("change_reference_system_out = False: skipping reference change from o.e. to image")

        if self._crystalpy_diffraction_setup is None:  # todo: supress if?
            self.set_crystalpy_diffraction_setup()
            self.align_crystal()

        p = self.get_coordinates().p()
        q = self.get_coordinates().q()
        theta_grazing1 = numpy.pi / 2 - self.get_coordinates().angle_radial()
        theta_grazing2 = numpy.pi / 2 - self.get_coordinates().angle_radial_out()
        alpha1 = self.get_coordinates().angle_azimuthal()

        #
        input_beam = self.get_input_beam().duplicate()

        soe = self.get_optical_element()

        if is_verbose():
            b_S, b_P = input_beam.get_efield_directions()
            print("\n\n")
            print(">>> input beam e_S, mod e_s", b_S[0], vector_modulus(b_S)[0])
            print(">>> input beam e_P, mod e_P, e_S.e_P: ", b_P[0], vector_modulus(b_P)[0], vector_dot(b_S, b_P)[0])

        #
        # put input_beam in crystal reference system
        #
        if change_reference_system_in:
            input_beam.rotate(alpha1,         axis=2)
            input_beam.rotate(theta_grazing1, axis=1)

            if is_verbose():
                b_S, b_P = input_beam.get_efield_directions()
                print("")
                print(">>> local beam e_S, mod e_s", b_S[0], vector_modulus(b_S)[0])
                print(">>> local beam e_P, mod e_P, e_S.e_P: ", b_P[0], vector_modulus(b_P)[0], vector_dot(b_S, b_P)[0])

            input_beam.translation([0.0, -p * numpy.cos(theta_grazing1), p * numpy.sin(theta_grazing1)])

        # crystal movement (forward):
        movements = self.get_movements()
        if movements is not None:
            if movements.f_move:
                input_beam.rot_for(OFFX=movements.offset_x,
                                   OFFY=movements.offset_y,
                                   OFFZ=movements.offset_z,
                                   X_ROT=movements.rotation_x,
                                   Y_ROT=movements.rotation_y,
                                   Z_ROT=movements.rotation_z)

        #
        # crystal diffraction
        #
        footprint, normal = self._apply_crystal_diffraction(input_beam)

        #
        # apply crystal movements (backwards) and boundaries
        #
        if movements is not None:
            if movements.f_move:
                footprint.rot_back(OFFX=movements.offset_x,
                                   OFFY=movements.offset_y,
                                   OFFZ=movements.offset_z,
                                   X_ROT=movements.rotation_x,
                                   Y_ROT=movements.rotation_y,
                                   Z_ROT=movements.rotation_z)

        footprint.apply_boundaries_syned(soe.get_boundary_shape(), flag_lost_value=flag_lost_value)

        #
        # from element reference system to image plane
        #
        output_beam = footprint.duplicate()
        if change_reference_system_out:
            output_beam.change_to_image_reference_system(theta_grazing2, q)

            if is_verbose():
                b_S, b_P = output_beam.get_efield_directions()
                print("")
                print(">>> image e_S, mod e_s", b_S[0], vector_modulus(b_S)[0])
                print(">>> image e_P, mod e_P, e_S.e_P: ", b_P[0], vector_modulus(b_P)[0], vector_dot(b_S, b_P)[0])

        return output_beam, footprint

    def _apply_crystal_diffraction(self, input_beam):
        """
        Applies mosaic crystal diffraction to the input beam.

        Calculates the surface intercepts, mosaic reflectivity, outgoing directions,
        and internal diffraction positions. Updates the beam positions, directions,
        Jones components, and electric field directions.

        Parameters
        ----------
        input_beam : instance of S4Beam
            The incident beam in the optical element reference system.

        Returns
        -------
        tuple
            (footprint, normal), where footprint is the updated S4Beam and normal
            is a numpy array of shape (3, nrays) containing the surface normals
            at the intercept points.
        """

        footprint, normal = self.get_optical_element().get_optical_surface_instance().calculate_intercept_on_beam(input_beam)

        if is_debug():
            print("    >>>>>> intercept: ", footprint.get_columns([1, 2, 3])[:, 0])
            print("    >>>>>> vout: ", footprint.get_columns([4, 5, 6])[:, 0])
            print("    >>>>>> normal: ", normal.shape, normal[:, 0])

        r_SS, r_PP = self._calculate_mosaic_reflectivity(footprint, normal)
        vIn, vOut = self._calculate_mosaic_reflection(footprint, normal)
        rIn, rOut = self._sample_mosaic_penetration(footprint, normal)

        jv_out_0, jv_out_1, ee_S, ee_P = self._calculate_jones_and_efield_directions(footprint, normal,
                                                                                        vIn, vOut, r_SS, r_PP)
        # update beam array with the new position
        footprint.set_column(1, rOut[:, 0])
        footprint.set_column(2, rOut[:, 1])
        footprint.set_column(3, rOut[:, 2])
        # update beam array with the new direction
        footprint.set_column(4, vOut[:, 0])
        footprint.set_column(5, vOut[:, 1])
        footprint.set_column(6, vOut[:, 2])
        # update beam array with the new electric fields
        footprint.set_jones_components(jv_out_0, jv_out_1, e_S=ee_S, e_P=ee_P)

        if is_verbose():
            print(">>> Orthogonal footprint: ", footprint.efields_orthogonal(),
                vector_dot(ee_S, ee_P)[0],
                vector_dot(ee_S, vOut)[0],
                vector_dot(ee_P, vOut)[0])

            b_S, b_P = footprint.get_efield_directions()
            print("")
            print(">>> reflected beam e_S, mod e_s", b_S[0], vector_modulus(b_S)[0])
            print(">>> reflected beam e_P, mod e_P, e_S.e_P: ", b_P[0], vector_modulus(b_P)[0], vector_dot(b_S, b_P)[0])


            print(">>> Intensity foot s, beam in s, foot p,  beam in p:",
                    footprint.get_column(24)[0], input_beam.get_column(24)[0],
                    footprint.get_column(25)[0], input_beam.get_column(25)[0],)


        return footprint, normal

    @staticmethod
    def _incident_facing_normal(footprint, normal):
        """Orient surface normals toward the incident beam, independently of shape.

        Multiplying conic coefficients by -1 reverses their normals without
        changing the surface. Hyperboloids and toroid branches also differ in
        this convention. All mosaic calculations require an incident-facing normal.
        """
        v_in = footprint.get_columns([4, 5, 6]).T
        reverse = vector_dot(v_in, normal.T) > 0
        return normal * numpy.where(reverse, -1.0, 1.0)[numpy.newaxis, :]

    def _calculate_mosaic_reflectivity(self, footprint, normal):
        """
        Calculates the amplitude reflectivity for a symmetric mosaic crystal.

        Uses a Gaussian crystallite orientation profile and the finite-thickness
        reflectivity of the 1992 model, with corrected Q coefficients.
        No orientation or penetration sampling is performed. The footprint is
        not modified.

        Parameters
        ----------
        footprint : instance of S4Beam
            The incident beam at the surface intercepts, in the optical element
            reference system.
        normal : numpy array shape (3, nrays)
            The surface normals at the intercept points.

        Returns
        -------
        tuple
            (r_SS, r_PP), two complex numpy arrays of shape (nrays,) containing
            the amplitude factors for the S and P polarizations. Their squared
            moduli give the corresponding intensity reflectivities.

        Raises
        ------
        NotImplementedError
            If the crystal cut is asymmetric.
        ValueError
            If the thickness is not finite and non-negative, the Gaussian
            mosaicity FWHM is not finite and positive, or the absorption
            coefficients are not finite and non-negative.
        """
        if self._crystalpy_diffraction_setup is None:
            self.set_crystalpy_diffraction_setup()
            
        setup = self._crystalpy_diffraction_setup
        soe = self.get_optical_element()
        if soe._thickness < 0 or not numpy.isfinite(soe._thickness):
            raise ValueError("Crystal thickness must be finite and non-negative.")

        surface_normal = self._incident_facing_normal(footprint, normal)

        energies = footprint.get_photon_energy_eV()
        v_in = footprint.get_columns([4, 5, 6]).T
        sin_theta = -vector_dot(v_in, surface_normal.T)
        theta = numpy.arcsin(numpy.clip(sin_theta, -1.0, 1.0))
        theta_bragg = setup.angleBragg(energies)
        theta_diff = theta - theta_bragg

        lambda_cm = codata.h * codata.c / (codata.e * energies) * 100
        mu = -2 * numpy.pi / lambda_cm * numpy.imag(setup.psi0(energies))
        Q_s = (numpy.pi**2 * numpy.abs(setup.psiH(energies) * setup.psiH_bar(energies))
               / (lambda_cm * numpy.sin(2 * theta_bragg)))
        Q_p = Q_s * numpy.cos(2 * theta_bragg)**2

        # Gaussian profile is intentionally fixed; its FWHM comes from the element.
        kappa = numpy.radians(soe._mosaicity_fwhm_deg) / numpy.sqrt(8 * numpy.log(2))
        if not numpy.isfinite(kappa) or kappa <= 0:
            raise ValueError("Gaussian mosaicity FWHM must be finite and positive.")
        w = numpy.exp(-0.5 * (theta_diff / kappa)**2) / (kappa * numpy.sqrt(2 * numpy.pi))
        if numpy.any(~numpy.isfinite(mu)) or numpy.any(mu < 0):
            raise ValueError("Absorption coefficients must be finite and non-negative.")

        path_cm = soe._thickness * 100 / numpy.sin(theta_bragg)

        def amplitude(Q):
            eta = w * Q
            # Algebraically equivalent to Eq. (8), without division by mu or
            # tanh(0). Includes the zero-thickness and zero-absorption limits.
            x = path_cm * numpy.sqrt(mu * (mu + 2 * eta))
            tanhc = numpy.ones_like(x)
            numpy.divide(numpy.tanh(x), x, out=tanhc, where=x != 0)
            effective_path = path_cm * tanhc
            reflectivity = eta * effective_path / (1 + (mu + eta) * effective_path)
            reflectivity = numpy.where(sin_theta > 0, reflectivity, 0.0)
            return numpy.sqrt(reflectivity).astype(complex)

        return amplitude(Q_s), amplitude(Q_p)

    def _calculate_mosaic_reflection(self, footprint, normal):
        """
        Samples the outgoing directions for a symmetric mosaic crystal.

        Rotates the surface normal to satisfy the Bragg condition, then samples
        an additional rotation around the incident direction using the Gaussian
        approximation of the 2013 model. Calculates the outgoing directions by
        reflection about the sampled crystallite normals. The footprint is
        not modified.

        Parameters
        ----------
        footprint : instance of S4Beam
            The incident beam at the surface intercepts, in the optical element
            reference system.
        normal : numpy array shape (3, nrays)
            The surface normals at the intercept points.

        Returns
        -------
        tuple
            (vIn, vOut), two numpy arrays of shape (nrays, 3) containing the
            incident and sampled outgoing unit directions.

        Raises
        ------
        NotImplementedError
            If the crystal cut is asymmetric.
        ValueError
            If the thickness is not finite and non-negative, the Gaussian
            mosaicity FWHM is not finite and positive, or the beta sampling
            coefficient s1 is not finite and positive.
        """
        if self._crystalpy_diffraction_setup is None:
            self.set_crystalpy_diffraction_setup()
            
        setup = self._crystalpy_diffraction_setup
        soe = self.get_optical_element()
        if soe._thickness < 0 or not numpy.isfinite(soe._thickness):
            raise ValueError("Crystal thickness must be finite and non-negative.")

        surface_normal = self._incident_facing_normal(footprint, normal)

        energies = footprint.get_photon_energy_eV()
        vIn = footprint.get_columns([4, 5, 6]).T

        # 1. Calculate Delta and a1
        sin_theta = -vector_dot(vIn, surface_normal.T)
        theta = numpy.arcsin(numpy.clip(sin_theta, -1.0, 1.0))
        theta_bragg = setup.angleBragg(energies)
        delta = theta_bragg - theta
        a1 = vector_cross(vIn, surface_normal.T)

        # 2. Calculate the reflected direction using the Bragg condition
        n1 = vector_rotate_around_axis(surface_normal.T, a1, delta)

        # 3. Sample beta using the Gaussian approximation of the 2013 model
        cos_alpha = -vector_dot(vIn, surface_normal.T)
        alpha = numpy.arccos(numpy.clip(cos_alpha, -1.0, 1.0))
        theta_D = numpy.pi / 2 - theta_bragg

        kappa = (
            numpy.radians(soe._mosaicity_fwhm_deg)
            / numpy.sqrt(8 * numpy.log(2))
        )
        if not numpy.isfinite(kappa) or kappa <= 0:
            raise ValueError("Gaussian mosaicity FWHM must be finite and positive.")

        # sinc(u/pi) = sin(u)/u, with the correct limit at u = 0
        u = alpha - theta_D
        s1 = numpy.sin(alpha) * numpy.sin(theta_D) / numpy.sinc(u / numpy.pi)

        if numpy.any(~numpy.isfinite(s1)) or numpy.any(s1 <= 0):
            raise ValueError("Gaussian beta sampling requires finite, positive s1.")

        sigma_beta = kappa / numpy.sqrt(s1)
        beta = numpy.random.normal(loc=0.0, scale=sigma_beta)

        # 4. Calculate n2
        n2 = vector_rotate_around_axis(n1, vIn, beta)

        # 5. Calculcate the reflected direction vOut
        vOut = vector_reflection(vIn, n2)     

        return vIn, vOut

    def _sample_mosaic_penetration(self, footprint, normal):
        """
        Samples the internal diffraction positions for a symmetric mosaic crystal.

        Samples the distance along each incident ray from a truncated exponential
        distribution, limited by the crystal thickness. Uses the Gaussian
        orientation profile and the S-polarization scattering coefficient.
        The footprint is not modified.

        Parameters
        ----------
        footprint : instance of S4Beam
            The incident beam at the surface intercepts, in the optical element
            reference system.
        normal : numpy array shape (3, nrays)
            The surface normals at the intercept points.

        Returns
        -------
        tuple
            (rin, rout), two numpy arrays of shape (nrays, 3) containing the
            surface entry positions and sampled internal diffraction positions
            in meters. Both are expressed in the optical element reference
            system.

        Raises
        ------
        NotImplementedError
            If the crystal cut is asymmetric.
        ValueError
            If the thickness is not finite and non-negative or the Gaussian
            mosaicity FWHM is not finite and positive.
        """
        if self._crystalpy_diffraction_setup is None:
            self.set_crystalpy_diffraction_setup()
            
        setup = self._crystalpy_diffraction_setup
        soe = self.get_optical_element()
        if soe._thickness < 0 or not numpy.isfinite(soe._thickness):
            raise ValueError("Crystal thickness must be finite and non-negative.")

        surface_normal = self._incident_facing_normal(footprint, normal)

        energies = footprint.get_photon_energy_eV()
        vIn = footprint.get_columns([4, 5, 6]).T
        sin_theta = -vector_dot(vIn, surface_normal.T)
        theta = numpy.arcsin(numpy.clip(sin_theta, -1.0, 1.0))
        theta_bragg = setup.angleBragg(energies)
        theta_diff = theta - theta_bragg

        lambda_cm = codata.h * codata.c / (codata.e * energies) * 100
        # Only s polarization for now!
        Q_s = (numpy.pi**2 * numpy.abs(setup.psiH(energies) * setup.psiH_bar(energies))
               / (lambda_cm * numpy.sin(2 * theta_bragg)))

        kappa = numpy.radians(soe._mosaicity_fwhm_deg) / numpy.sqrt(8 * numpy.log(2))
        if not numpy.isfinite(kappa) or kappa <= 0:
            raise ValueError("Gaussian mosaicity FWHM must be finite and positive.")
        w = numpy.exp(-0.5 * (theta_diff / kappa)**2) / (kappa * numpy.sqrt(2 * numpy.pi))

        eta_cm = w * Q_s
        if numpy.any(~numpy.isfinite(eta_cm)) or numpy.any(eta_cm < 0):
            raise ValueError("Scattering coefficients must be finite and non-negative.")

        # Absolute cosine of the incidence angle relative to the surface normal
        cos_incidence = numpy.abs(vector_dot(vIn, surface_normal.T))
        if numpy.any(~numpy.isfinite(cos_incidence)) or numpy.any(cos_incidence == 0):
            raise ValueError("Penetration sampling requires finite, non-zero incidence cosines.")

        # Maximum path length in cm: _thickness is in meters
        L_cm = 100.0 * soe._thickness / cos_incidence
        if numpy.any(~numpy.isfinite(L_cm)) or numpy.any(L_cm < 0):
            raise ValueError("Maximum path lengths must be finite and non-negative.")

        # Uniform random numbers in [0, 1)
        u = numpy.random.random(size=vIn.shape[0])

        # Uniform limit for eta_cm == 0
        s_cm = u * L_cm

        # Truncated exponential for eta_cm > 0
        numpy.divide(
            -numpy.log1p(u * numpy.expm1(-eta_cm * L_cm)),
            eta_cm,
            out=s_cm,
            where=eta_cm > 0,
        )

        # Entry positions in meters, shape (N, 3)
        rin = footprint.get_columns([1, 2, 3]).T

        # Internal diffraction positions in meters
        rout = rin + (s_cm / 100.0)[:, None] * vIn

        return rin, rout
                

    def _calculate_jones_and_efield_directions(self, footprint, normal, vIn, vOut, r_SS, r_PP):
        """
        Calculates the Jones vector after crystal diffraction. It also returns the directions of the
        S and P polarized components of the electric field.


        Parameters
        ----------
        footprint : instance of S4Beam
            The input beam
        normal : numpy array shape (nrays, 3)
            The normal to the surface at the intercept points.
        vIn :  numpy array shape (nrays, 3)
            The incident directions
        vOut :  numpy array shape (nrays, 3)
            The incident directions
        r_SS : numpy array complex shape (nrays)
            The crystal reflectivity for the S polarization
        r_PP : numpy array complex shape (nrays)
            The crystal reflectivity for the P polarization

        Returns
        -------
        tuple
            (jv_out_0, jv_out_1, ee_S, ee_P) the two components of the Jones vector and the two vectors of
            shape(nrays, 3) with the electric vectors for the S and P polarizations.

        """
        #
        # get versors with the sigma and pi directions for:
        #     e_S, e_P: the incident beam (as it is)
        #     es_S, es_P: the scattering plane spanned by vIn (incident)
        #     ee_S, ee_P: the scattering plane spanned by vOut (incident)
        #
        # Note that vector_norm() is not needed (for the vectors that should be unitary),
        # but renormalizing them improves accuracy in the calculation of c, s
        #
        e_S, e_P = footprint.get_efield_directions()  # these are \hat{u}_{\sigma,\pi} in Eq. 3

        axis = vector_norm(vector_cross(vIn, vOut))

        es_S = axis  # \hat{u}_{\sigma,i} in Eq. 12
        es_P = vector_norm(vector_cross(es_S, vIn))  # \hat{u}_{\pi,i} in Eq. 12

        ee_S = axis  # \hat{u}_{\sigma,f} in Eq. 13
        ee_P = vector_norm(vector_cross(ee_S, vOut))  # \hat{u}_{\pi,f} in Eq. 13

        if is_verbose():
            print(">>>>> e_S, perp vIn: ", e_S[0], vector_dot(e_S, vIn)[0])
            print(">>>>> e_P, perp vIn: ", e_P[0], vector_dot(e_P, vIn)[0])

            print(">>>>> axis, mod, perp vIn: ", axis[0], vector_modulus(axis)[0], vector_dot(axis, vIn)[0])
            print(">>>>> final ee_S, perp vOut: ", ee_S[0], vector_dot(ee_S, vOut)[0])
            print(">>>>> final ee_P, perp vOut: ", ee_P[0], vector_dot(ee_P, vOut)[0])

        #
        # Jones calculus of refletivity
        #

        # Jones matrix (local)
        J00 = r_SS
        J01 = 0
        J10 = 0
        J11 = r_PP

        # rotation matrix R
        if True:  # todo delete, only for test
            # c = e_S[:, 0] # cos of angle between e_S and the x axis
            c = vector_dot(e_S, ee_S)
            s = numpy.sqrt(1 - c ** 2)  # sin
            if is_verbose(): print(">>> s, c, angle: ", s[0], c[0], numpy.degrees(numpy.arctan2(s[0], c[0])))
            R00 = c
            R01 = -s
            R10 = s
            R11 = c
            if is_verbose(): print(">>> R angles: ", R00[0], R01[0], R10[0], R11[0])

        R00 = vector_dot(e_S, es_S)
        R01 = vector_dot(e_S, es_P)
        R10 = vector_dot(e_P, es_S)
        R11 = vector_dot(e_P, es_P)

        # J x R(alpha), the Jones matrix to apply to the Jones vector of the incident rays
        Jrotated_00 = J00 * R00 + J01 * R10  # r_SS * c
        Jrotated_01 = J00 * R01 + J01 * R11  # -r_SS * s
        Jrotated_10 = J10 * R00 + J11 * R10  # r_PP * s
        Jrotated_11 = J10 * R01 + J11 * R11  # r_PP * c

        if is_verbose():
            print(">>> R dotprd: ", R00[0], R01[0], R10[0], R11[0])
            print(">>> J: ", Jrotated_00[0], Jrotated_01[0], Jrotated_10[0], Jrotated_11[0])
            print(">>> |J|: ", numpy.abs(Jrotated_00[0]), numpy.abs(Jrotated_01[0]), numpy.abs(Jrotated_10[0]),
                  numpy.abs(Jrotated_11[0]))

        # Jones vector of incident rays
        jv_in_0, jv_in_1 = footprint.get_jones_components()
        # Jones vector or reflected rays
        jv_out_0 = Jrotated_00 * jv_in_0 + Jrotated_01 * jv_in_1
        jv_out_1 = Jrotated_10 * jv_in_0 + Jrotated_11 * jv_in_1

        return jv_out_0, jv_out_1, ee_S, ee_P


if __name__ == "__main__":
    c = S4MosaicCrystal(
        name="Undefined",
        boundary_shape=None,
        surface_shape=None,
        material="graphite",
        miller_index_h=1,
        miller_index_k=1,
        miller_index_l=1,
        thickness=0.010,
        f_central=0,
        f_phot_cent=0,
        phot_cent=8000.0,
        material_constants_library_flag=0,  # 0=xraylib, 1=dabax
        # 2=shadow preprocessor file v1
        # 3=shadow preprocessor file v2
        file_refl="",
        dabax=None,
        mosaicity_fwhm_deg=0.4,
        mosaicity_profile_flag=0,  # 0=Gaussian, 1=External
    )

    print(c.info())


    ce = S4MosaicCrystalElement(optical_element=c)
    print(ce.info())


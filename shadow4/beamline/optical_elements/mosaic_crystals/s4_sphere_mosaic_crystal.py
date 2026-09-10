"""
The s4 spherical or cylindrical mosaic crystal (optical element and beamline element).
"""
from syned.beamline.element_coordinates import ElementCoordinates
from syned.beamline.shape import Sphere, SphericalCylinder, Convexity, Direction

from shadow4.beam.s4_beam import S4Beam
from shadow4.beamline.optical_elements.mosaic_crystals.s4_mosaic_crystal import S4MosaicCrystalElement, S4MosaicCrystal
from shadow4.beamline.s4_optical_element_decorators import SurfaceCalculation, S4SphereOpticalElementDecorator
from shadow4.beamline.s4_beamline_element_movements import S4BeamlineElementMovements


class S4SphereMosaicCrystal(S4MosaicCrystal, S4SphereOpticalElementDecorator):
    """
    Shadow4 Sphere Mosaic Crystal Class
    This is a spherical or cylindrical mosaic crystal in reflection geometry (Bragg), using the diffracted beam.

    Constructor.

    Parameters
    ----------
    name :  str, optional
        A name for the crystal
    boundary_shape : instance of BoundaryShape, optional
        The information on the crystal boundaries.
    material : str, optional
        The crystal material name (a name accepted by crystalpy).
    miller_index_h : int, optional
        The Miller index H.
    miller_index_k : int, optional
        The Miller index K.
    miller_index_l : int, optional
        The Miller index L.
    thickness : float, optional
        The diffracting crystal thickness in m.
    f_central : int, optional
        Flag for autosetting the crystal to the corrected Bragg angle.
    f_phot_cent : int, optional
        0: setting photon energy in eV, 1:setting photon wavelength in m.
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

    radius : float, optional
        The radius of curvature in m.
    is_cylinder : bool, optional
        Use a cylindrical surface instead of a sphere.
    cylinder_direction : int, optional
        Direction.TANGENTIAL or Direction.SAGITTAL, as in S4SphereCrystal.
    convexity : int, optional
        Convexity.UPWARD for concave or Convexity.DOWNWARD for convex.

    Returns
    -------
    instance of S4SphereMosaicCrystal.
    """
    def __init__(self,
                 name="Sphere mosaic crystal",
                 boundary_shape=None,
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
                 radius=1.0,
                 is_cylinder=False,
                 cylinder_direction=Direction.TANGENTIAL,
                 convexity=Convexity.UPWARD,
                 ):
        S4SphereOpticalElementDecorator.__init__(
            self,
            surface_calculation=SurfaceCalculation.EXTERNAL,
            is_cylinder=is_cylinder,
            cylinder_direction=cylinder_direction,
            convexity=convexity,
            radius=radius,
        )
        S4MosaicCrystal.__init__(self,
                        name=name,
                        surface_shape=self.get_surface_shape_instance(),
                        boundary_shape=boundary_shape,
                        material=material,
                        miller_index_h=miller_index_h,
                        miller_index_k=miller_index_k,
                        miller_index_l=miller_index_l,
                        thickness=thickness,
                        f_central=f_central,
                        f_phot_cent=f_phot_cent,
                        phot_cent=phot_cent,
                        material_constants_library_flag=material_constants_library_flag, # 0=xraylib, 1=dabax
                        file_refl=file_refl,
                        dabax=dabax,
                        mosaicity_fwhm_deg=mosaicity_fwhm_deg,
                        mosaicity_profile_flag=mosaicity_profile_flag,  # 0=Gaussian, 1=External
                        )

        self.__inputs = {
            "radius": radius,
            "is_cylinder": is_cylinder,
            "cylinder_direction": cylinder_direction,
            "convexity": convexity,
            "name": name,
            "boundary_shape": boundary_shape,
            "material": material,
            "miller_index_h": miller_index_h,
            "miller_index_k": miller_index_k,
            "miller_index_l": miller_index_l,
            "thickness": thickness,
            "f_central": f_central,
            "f_phot_cent": f_phot_cent,
            "phot_cent": phot_cent,
            "file_refl": file_refl,
            "material_constants_library_flag": material_constants_library_flag,
            "dabax": self._get_dabax_txt(),
            "mosaicity_fwhm_deg": mosaicity_fwhm_deg,
            "mosaicity_profile_flag": mosaicity_profile_flag,  # 0=Gaussian, 1=External
            }

    def to_python_code(self, **kwargs):
        """
        Creates the python code for defining the element.

        Parameters
        ----------
        **kwargs

        Returns
        -------
        str
            Python code.
        """
        txt = self.to_python_code_boundary_shape()
        txt_pre = """

from shadow4.beamline.optical_elements.mosaic_crystals.s4_sphere_mosaic_crystal import S4SphereMosaicCrystal
optical_element = S4SphereMosaicCrystal(name={name!r},
    boundary_shape=boundary_shape, material={material!r},
    miller_index_h={miller_index_h}, miller_index_k={miller_index_k}, miller_index_l={miller_index_l},
    thickness={thickness},
    f_central={f_central}, f_phot_cent={f_phot_cent}, phot_cent={phot_cent},
    file_refl={file_refl!r},
    material_constants_library_flag={material_constants_library_flag}, # 0=xraylib,1=dabax,2=preprocessor v1,3=preprocessor v2
    dabax={dabax}, # used when material_constants_library_flag=1,
    radius={radius!r}, is_cylinder={is_cylinder},
    cylinder_direction={cylinder_direction}, convexity={convexity},
    mosaicity_fwhm_deg={mosaicity_fwhm_deg},
    mosaicity_profile_flag={mosaicity_profile_flag},  # 0=Gaussian, 1=External
    )"""
        txt += txt_pre.format(**self.__inputs)

        return txt

class S4SphereMosaicCrystalElement(S4MosaicCrystalElement):
    """
    The Shadow4 spherical or cylindrical mosaic crystal element.
    It is made of a S4SphereMosaicCrystal and an ElementCoordinates instance. It also includes the input beam.

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

    """
    def __init__(self,
                 optical_element : S4SphereMosaicCrystal = None,
                 coordinates : ElementCoordinates = None,
                 movements: S4BeamlineElementMovements = None,
                 input_beam : S4Beam = None):
        super().__init__(optical_element=optical_element if optical_element is not None else S4SphereMosaicCrystal(),
                         coordinates=coordinates if coordinates is not None else ElementCoordinates(),
                         movements=movements,
                         input_beam=input_beam)

        if not isinstance(self.get_optical_element().get_surface_shape(), (Sphere, SphericalCylinder)):
            raise ValueError("Wrong Optical Element: only Sphere or Spherical Cylinder shape is accepted")

    def to_python_code(self, **kwargs):
        """
        Creates the python code for defining the element.

        Parameters
        ----------
        **kwargs

        Returns
        -------
        str
            Python code.
        """
        txt = "\n\n# optical element number XX"
        txt += self.get_optical_element().to_python_code()
        txt += self.to_python_code_coordinates()
        txt += self.to_python_code_movements()
        txt += "\nfrom shadow4.beamline.optical_elements.mosaic_crystals.s4_sphere_mosaic_crystal import S4SphereMosaicCrystalElement"
        txt += "\nbeamline_element = S4SphereMosaicCrystalElement(optical_element=optical_element,coordinates=coordinates, movements=movements, input_beam=beam)"
        txt += "\n\nbeam, footprint = beamline_element.trace_beam()"
        return txt

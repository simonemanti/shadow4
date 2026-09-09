"""
The s4 plane crystal (optical element and beamline element).
"""
import numpy
from syned.beamline.element_coordinates import ElementCoordinates

from shadow4.beam.s4_beam import S4Beam
from shadow4.beamline.optical_elements.mosaic_crystals.s4_mosaic_crystal import S4MosaicCrystalElement, S4MosaicCrystal
from shadow4.beamline.s4_optical_element_decorators import S4PlaneOpticalElementDecorator
from shadow4.beamline.s4_beamline_element_movements import S4BeamlineElementMovements

from syned.beamline.optical_elements.crystals.crystal import DiffractionGeometry

class S4PlaneMosaicCrystal(S4MosaicCrystal, S4PlaneOpticalElementDecorator):
    """
    Shadow4 Plane Crystal Class
    This is a plane perfect crystal in reflection geometry (Bragg), using the diffracted beam.

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
    f_bragg_a : int, optional
        Asymmetric crystal 0:No, 1:Yes.
    asymmetry_angle : float, optional
        For f_bragg_a=1, the asymmetry angle (angle between crystal planes and surface) in rads.
    is_thick : int, optional
        Use thick crystal approximation.
    thickness : float, optional
        For is_thick=0, the crystal thickness in m.
    f_central : int, optional
        Flag for autosetting the crystal to the corrected Bragg angle.
    f_phot_cent : int, optional
        0: setting photon energy in eV, 1:setting photon wavelength in m.
    phot_cent : float, optional
        for f_central=1, the value of the photon energy (f_phot_cent=0) or photon wavelength (f_phot_cent=1).
    f_ext : inf, optional
        Flag for autosetting the crystal surface parameters.
        0: internal/calculated parameters, 1:external/user defined parameters. TODO: delete?
    material_constants_library_flag : int, optional
        Flag for indicating the origin of the crystal data:
        0: xraylib, 1: dabax, 2: preprocessor file v1, 3: preprocessor file v2.
    file_refl : str, optional
        for material_constants_library_flag=2,3, the name of the file containing the crystal parameters.
    dabax : None or instance of DabaxXraylib,
        The pointer to the dabax library  (used for material_constants_library_flag=1).

    Returns
    -------
    instance of S4PlaneCrystal.
    """
    def __init__(self,
                 name="Undefined",
                 boundary_shape=None,
                 material="graphite",
                 miller_index_h=1,
                 miller_index_k=1,
                 miller_index_l=1,
                 asymmetry_angle=0.0,
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
                 mosaicity_profile_flag=0,  # 0=Gaussian, 1=Lorentzian
                 ):
        S4PlaneOpticalElementDecorator.__init__(self)
        S4MosaicCrystal.__init__(self,
                        name=name,
                        surface_shape=self.get_surface_shape_instance(),
                        boundary_shape=boundary_shape,
                        material=material,
                        miller_index_h=miller_index_h,
                        miller_index_k=miller_index_k,
                        miller_index_l=miller_index_l,
                        # f_bragg_a=False,
                        asymmetry_angle=asymmetry_angle,
                        # is_thick=0,          # 1=Use thick crystal approximation
                        thickness=thickness,
                        f_central=f_central,
                        f_phot_cent=f_phot_cent,
                        phot_cent=phot_cent,
                        material_constants_library_flag=material_constants_library_flag, # 0=xraylib, 1=dabax
                        file_refl=file_refl,
                        dabax=dabax,
                        mosaicity_fwhm_deg=mosaicity_fwhm_deg,
                        mosaicity_profile_flag=mosaicity_profile_flag,  # 0=Gaussian, 1=Lorentzian
                        )

        self.__inputs = {
            "name": name,
            "boundary_shape": boundary_shape,
            "material": material,
            # "diffraction_geometry": diffraction_geometry,
            "miller_index_h": miller_index_h,
            "miller_index_k": miller_index_k,
            "miller_index_l": miller_index_l,
            "asymmetry_angle": asymmetry_angle,
            # "is_thick": is_thick,
            "thickness": thickness,
            "f_central": f_central,
            "f_phot_cent": f_phot_cent,
            "phot_cent": phot_cent,
            "file_refl": file_refl,
            # "f_bragg_a": f_bragg_a,
            # "f_ext": f_ext,
            "material_constants_library_flag": material_constants_library_flag,
            # "method_efields_management": method_efields_management,
            "dabax": self._get_dabax_txt(),
            "mosaicity_fwhm_deg": mosaicity_fwhm_deg,
            "mosaicity_profile_flag": mosaicity_profile_flag,  # 0=Gaussian, 1=Lorentzian
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

from shadow4.beamline.optical_elements.mosaic_crystals.s4_plane_mosaic_crystal import S4PlaneMosaicCrystal
optical_element = S4PlaneMosaicCrystal(name='{name}',
    boundary_shape=boundary_shape, material='{material}',
    miller_index_h={miller_index_h}, miller_index_k={miller_index_k}, miller_index_l={miller_index_l},
    asymmetry_angle={asymmetry_angle},
    thickness={thickness},
    f_central={f_central}, f_phot_cent={f_phot_cent}, phot_cent={phot_cent},
    file_refl='{file_refl}',
    material_constants_library_flag={material_constants_library_flag}, # 0=xraylib,1=dabax,2=preprocessor v1,3=preprocessor v2
    dabax={dabax}, # used when material_constants_library_flag=1,
    mosaicity_fwhm_deg={mosaicity_fwhm_deg},
    mosaicity_profile_flag={mosaicity_profile_flag},  # 0=Gaussian, 1=Lorentzian
    )"""
        txt += txt_pre.format(**self.__inputs)

        return txt

class S4PlaneMosaicCrystalElement(S4MosaicCrystalElement):
    """
    The Shadow4 plane crystal element.
    It is made of a S4PlaneCrystal and an ElementCoordinates instance. It also includes the input beam.

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
                 optical_element : S4PlaneMosaicCrystal = None,
                 coordinates : ElementCoordinates = None,
                 movements: S4BeamlineElementMovements = None,
                 input_beam : S4Beam = None):
        super().__init__(optical_element=optical_element if optical_element is not None else S4PlaneMosaicCrystal(),
                         coordinates=coordinates if coordinates is not None else ElementCoordinates(),
                         movements=movements,
                         input_beam=input_beam)

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
        txt += "\nfrom shadow4.beamline.optical_elements.mosaic_crystals.s4_plane_mosaic_crystal import S4PlaneMosaicCrystalElement"
        txt += "\nbeamline_element = S4PlaneMosaicCrystalElement(optical_element=optical_element,coordinates=coordinates, movements=movements, input_beam=beam)"
        txt += "\n\nbeam, footprint = beamline_element.trace_beam()"
        return txt


if __name__ == "__main__":
    import numpy as np
    from dabax.dabax_xraylib import DabaxXraylib
    from shadow4.beamline.s4_beamline import S4Beamline

    beamline = S4Beamline()

    #
    #
    #
    from shadow4.sources.source_geometrical.source_geometrical import SourceGeometrical

    light_source = SourceGeometrical(name='Geometrical Source', nrays=15000, seed=5676561)
    light_source.set_spatial_type_point()
    light_source.set_depth_distribution_off()
    light_source.set_angular_distribution_gaussian(sigdix=1e-06, sigdiz=1e-06)
    light_source.set_energy_distribution_uniform(value_min=9000, value_max=11000, unit='eV')
    light_source.set_polarization(polarization_degree=1, phase_diff=0, coherent_beam=0)
    beam = light_source.get_beam()

    beamline.set_light_source(light_source)

    # optical element number XX
    boundary_shape = None

    from shadow4.beamline.optical_elements.crystals.s4_plane_crystal import S4PlaneCrystal

    optical_element = S4PlaneMosaicCrystal(name='Generic Crystal',
                                     boundary_shape=boundary_shape, material='Si',
                                     miller_index_h=1, miller_index_k=1, miller_index_l=1,
                                     # f_bragg_a=False,
                                     asymmetry_angle=0.0,
                                     # is_thick=1,
                                     thickness=0.001,
                                     f_central=1, f_phot_cent=0, phot_cent=10000.0,
                                     file_refl='bragg.dat',
                                     # f_ext=0,
                                     material_constants_library_flag=1,
                                     # 0=xraylib,1=dabax,2=preprocessor v1,3=preprocessor v2
                                     # method_efields_management=0,  # 0=new in S4; 1=like in S3
                                     dabax=DabaxXraylib(file_f0="f0_InterTables.dat", file_f1f2="f1f2_Windt.dat"),
                                     # used when material_constants_library_flag=1,
                                     )
    from syned.beamline.element_coordinates import ElementCoordinates

    coordinates = ElementCoordinates(p=30, q=0, angle_radial=1.371743969, angle_azimuthal=0,
                                     angle_radial_out=1.371743969)
    movements = None
    from shadow4.beamline.optical_elements.crystals.s4_plane_crystal import S4PlaneCrystalElement

    beamline_element = S4PlaneMosaicCrystalElement(optical_element=optical_element, coordinates=coordinates,
                                             movements=movements, input_beam=beam)

    beam, footprint = beamline_element.trace_beam()

    beamline.append_beamline_element(beamline_element)

    # test plot
    if True:
        from srxraylib.plot.gol import plot_scatter

        plot_scatter(beam.get_photon_energy_eV(nolost=1), beam.get_column(23, nolost=1),
                     title='(Intensity,Photon Energy)', plot_histograms=0)#, yrange=[0,1.1])
        plot_scatter(1e6 * beam.get_column(1, nolost=1), 1e6 * beam.get_column(3, nolost=1), title='(X,Z) in microns')


    print(beamline_element.info())

    print(beam.get_intensity(nolost=1))

    print(beamline.to_python_code())

import cortex
from scipy import stats
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt


class Subsurface(object):

    """subsurface
        This is a utility that uses pycortex for making sub-surfaces for CF fitting.

    """

    def __init__(self, cx_sub, boolmasks, surftype='fiducial'):
        """__init__
        Parameters
        ----------
        cx_sub : The name of the cx subject (string). This is used to get surfaces from the pycx database.
        boolmasks: A list of boolean arrays that define the vertices that correspond to the ROI one wants to make a subsurface from [left hem, right hem].
        surftype: The surface (default = fiducial).
        """

        self.cx_sub = cx_sub
        self.surftype = surftype
        self.boolmasks = boolmasks
        # Put the mask into int format for plotting.
        self.mask = np.concatenate(
            [self.boolmasks[0], self.boolmasks[1]]).astype(int)

    def create(self):
        """get_surfaces
        Function that creates the subsurfaces.
        """

        self.get_surfaces()
        self.generate()
        self.get_geometry()
        self.pad_distance_matrices()

    def get_surfaces(self):
        """get_surfaces
        Accesses the pycortex database to return the subject surfaces (left and right).

        Returns
        -------
        subsurface_L, subsurface_R: A pycortex subsurfaces classes for each hemisphere (These are later deleted by 'get_geometry', but can be re-created with a call to this function).
        self.subsurface_verts_L,self.subsurface_verts_R : The whole brain indices of each vertex in the subsurfaces.

        """

        self.surfaces = [cortex.polyutils.Surface(*d)
                         for d in cortex.db.get_surf(self.cx_sub, self.surftype)]

    def generate(self):
        """generate
        Use the masks defined in boolmasks to define subsurfaces.
        """

        print('Generating subsurfaces')
        # Create sub-surface, left hem.
        self.subsurface_L = self.surfaces[0].create_subsurface(
            vertex_mask=self.boolmasks[0])
        # Create sub-surface, right hem.
        self.subsurface_R = self.surfaces[1].create_subsurface(
            vertex_mask=self.boolmasks[1])

        # Get the whole-brain indices for those vertices contained in the subsurface.
        self.subsurface_verts_L = np.where(self.subsurface_L.subsurface_vertex_map != stats.mode(
            self.subsurface_L.subsurface_vertex_map)[0][0])[0]
        self.subsurface_verts_R = np.where(self.subsurface_R.subsurface_vertex_map != stats.mode(
            self.subsurface_R.subsurface_vertex_map)[0][0])[0]+self.subsurface_L.subsurface_vertex_map.shape[-1]

    def get_geometry(self):
        """get_geometry
        Calculates geometric info about the sub-surfaces. Computes geodesic distances from each point of the sub-surface.

        Returns
        -------
        dists_L, dists_R: Matrices of size n vertices x n vertices that describes the distances between all vertices in each hemisphere of the subsurface.
        subsurface_verts: The whole brain indices of each vertex in the subsurface.
        leftlim: The index that indicates the boundary between the left and right hemisphere. 
        """

        # Assign some variables to determine where the boundary between the hemispheres is.
        self.leftlim = np.max(self.subsurface_verts_L)
        self.subsurface_verts = np.concatenate(
            [self.subsurface_verts_L, self.subsurface_verts_R])

        # Make the distance x distance matrix.
        ldists, rdists = [], []

        print('Creating distance by distance matrices')

        for i in range(len(self.subsurface_verts_L)):
            ldists.append(self.subsurface_L.geodesic_distance([i]))
        self.dists_L = np.array(ldists)

        for i in range(len(self.subsurface_verts_R)):
            rdists.append(self.subsurface_R.geodesic_distance([i]))
        self.dists_R = np.array(rdists)

        # Get rid of these as they are harmful for pickling. We no longer need them.
        self.surfaces, self.subsurface_L, self.subsurface_R = None, None, None

    def pad_distance_matrices(self, padval=np.Inf):
        """pad_distance_matrices
        Pads the distance matrices so that distances to the opposite hemisphere are np.inf
        Stack them on top of each other so they will have the same size as the design matrix


        Returns
        -------
        distance_matrix: A matrix of size n vertices x n vertices that describes the distances between all vertices in the subsurface.
        """

        # Pad the right hem with np.inf.
        padL = np.pad(
            self.dists_L, ((0, 0), (0, self.dists_R.shape[-1])), constant_values=np.Inf)
        # pad the left hem with np.inf..
        padR = np.pad(
            self.dists_R, ((0, 0), (self.dists_L.shape[-1], 0)), constant_values=np.Inf)

        self.distance_matrix = np.vstack([padL, padR])  # Now stack.
        self.distance_matrix = (self.distance_matrix + self.distance_matrix.T)/2 # Make symmetrical

    def elaborate(self):
        """elaborate
        Prints information about the created subsurfaces.

        """

        print(
            f"Maximum distance across left subsurface: {np.max(self.dists_L)} mm")
        print(
            f"Maximum distance across right subsurface: {np.max(self.dists_R)} mm")
        print(f"Vertices in left hemisphere: {self.dists_L.shape[-1]}")
        print(f"Vertices in right hemisphere: {self.dists_R.shape[-1]}")
        
    
    def limit_vertices(self, ecc, rsq, maxecc = 8.91, maxrsq = 0.05):
        """limit_vertices
        Limits the vertices to sample from based on eccentricity and rsq from separate pRF results.
        This is useful for limiting the vertices to visually active vertices that also 'see' the
        screen. Visually inspect the limited vertices with the intmask to see what you sample.
        
        Parameters
        ----------
        ecc: array of eccentricities acquired from separate pRF analysis (shape: 118584,)
        rsq: array of rsq values acquired from separate pRF analysis (shape: 118584,)
        maxecc : maximum eccentricity allowed for a vertices to have. Set to 8.91 for nprf_ss
        experiment. Acquired with PRFStimulus2D and taking .screen_size_degrees/2.0
        maxrsq : Rsq threshold for the vertices to sample from
        
        Returns
        -------
        distance_matrix: A matrix of size n vertices x n vertices that describes the distances between all vertices in the subsurface, but limited to specified eccentricity and rsq.
        intmask : a mask that can be plotted with pycortex to show which vertices are sampled
        for visual inspection.

        """
        
        # create boolean mask for eccentricity and rsq and split between left and right
        eccMask = ecc < maxecc
        rsqMask = rsq > maxrsq
        combinedmask = rsqMask * eccMask
        
        V1mask = np.concatenate([self.boolmasks[0], self.boolmasks[1]])
        self.intmask = (combinedmask * V1mask).astype(int)
            
        
        n = len(combinedmask)
        half = int(n/2)
        maskL, maskR = combinedmask[:half], combinedmask[n-half:]
        V1maskL = maskL[self.boolmasks[0]]
        V1maskR = maskR[self.boolmasks[1]]
        
        # modify the left and right distance matrices and create new distance matrix
        self.dists_L = self.dists_L[V1maskL][:,V1maskL]
        self.dists_R = self.dists_R[V1maskR][:,V1maskR]
        self.pad_distance_matrices()
        self.subsurface_verts = np.concatenate([self.subsurface_verts_L[V1maskL],self.subsurface_verts_R[V1maskR]])
        
        
    def create_visual_space(self, x, y):
        """limit_vertices
        Limits the vertices to sample from based on eccentricity and rsq from separate pRF results.
        This is useful for limiting the vertices to visually active vertices that also 'see' the
        screen. Visually inspect the limited vertices with the intmask to see what you sample.
        
        Parameters
        ----------
        x : array of x coordinates acquired from separate pRF analysis (shape: 118584,)
        y : array of y coordinates acquired from separate pRF analysis (shape: 118584,)
        
        Returns
        -------
        visual_distance_matrix: distance matrix in log visual space based on previous pRF analysis

        """
        
        # get distance in visual space for the V1 vertices in our Subsurface
        dxy = {'x': x, 'y': y}
        xy_frame = pd.DataFrame(data = dxy)
        self.xy_frame_all = xy_frame
        spliced_lookup_xy = xy_frame.iloc[self.subsurface_verts.astype(int),:]
        
        self.xy_verts = spliced_lookup_xy
        visual_distances = euclidean_distances(spliced_lookup_xy)
        self.visual_distance_matrix = visual_distances
        
        # calculate eccentricity and polar angle
        ecc = np.abs(x + y * 1j)
        angle = np.angle(x + y * 1j)
        
        # create dataframe to easily index ecc for the V1 vertices in our Subsurface
        decc = {'eccentricity': ecc}
        ecc_frame = pd.DataFrame(data = decc)
        
        spliced_lookup_ecc = ecc_frame.iloc[self.subsurface_verts.astype(int),:]
        
        
        
        # calculate the cortical magnification factors
        CMF_matrix = self.distance_matrix/visual_distances
        CMF_matrix[CMF_matrix == np.inf] = np.nan
        
        # fit the CMF curve to find the parameters necesaary to transform x and y coordinates
        # to their log x and y coordinates to accounts for CMF        
        tofit_x = np.array(spliced_lookup_ecc['eccentricity'])
        self.tofit_x = tofit_x
        tofit_y = np.nanmean(CMF_matrix, axis=1)
        self.tofit_y = tofit_y
        
        
        
        pars, cov = curve_fit(f=magnification, xdata=tofit_x, 
                      ydata=tofit_y, p0=[50, 3])
        

        
#         pars = [46.60976917, 3.22366307]
        
        # extract the parameters
        lamb, ecc0 = pars
        
        print(pars)
        # transform the coordinates
        logx = lamb*np.log(1+ecc/ecc0) * np.cos(angle)
        logy = lamb*np.log(1+ecc/ecc0) * np.sin(angle)
        
        # create the logvisual space distance matrix
        dlog = {'logx': logx, 'logy': logy}
        logxy_frame = pd.DataFrame(data = dlog)
        self.logxy_frame_all = logxy_frame
        spliced_lookup_logxy = logxy_frame.iloc[self.subsurface_verts.astype(int),:]
        
        self.logxy_verts = spliced_lookup_logxy
        
        self.logvisual_distance_matrix = euclidean_distances(spliced_lookup_logxy)
        self.ecc = tofit_x
        self.CMF = tofit_y
        plotx = np.linspace(start=np.nanmin(tofit_x), stop=np.nanmax(tofit_x), num=200)
        fig, ax = plt.subplots()
        ax.scatter(tofit_x, tofit_y)
        ax.plot(plotx, magnification(plotx, *pars), linestyle='--', color='black')
        ax.set_xlabel('Eccentricity (deg)')
        ax.set_ylabel('Cortical magnification factor (mm/deg)')
        print(pars)
# from scipy.optimize import curve_fit
# from joblib import Parallel, delayed

# # assign what to fit
# # fig, ax = plt.subplots()
# tofit_x2 = spliced_lookup_ecc['ecc']
# tofit_y2 = np.nanmean(CMF, axis=1)
# # ax.scatter(tofit_x, tofit_y)

# plotx2 = np.linspace(start=np.nanmin(tofit_x), stop=np.nanmax(tofit_x), num=200)

# # define the function
# def magnification(x, a, b):
#     return a/(b+x)


# pars2, cov = curve_fit(f=magnification, xdata=tofit_x, 
#                       ydata=tofit_y, p0=[1, 0])

# fig, ax = plt.subplots()
# ax.scatter(tofit_x2, tofit_y2)
# ax.plot(plotx2, magnification(plotx2, *pars), linestyle='--', linewidth=2, color='black')
# print(pars2)



def squaresign(vec):
    """squaresign
        Raises something to a power in a sign-sensive way.
        Useful for if dot products happen to be negative.
    """
    vec2 = (vec**2)*np.sign(vec)
    return vec2


def magnification(x, a, b):
    """magnification
        Function used to fit cortical magnification factor parameters
    """
    return a/(b+x)

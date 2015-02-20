import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def align_eyes(fix_L, fix_R, loc_L, loc_R, check_results=0, plot_results=0):
    """
    The data from the Eyelink rarely results in gaze directions for the two eyes that actually
    intersection. The goal of this function is to take the skew gaze rays for a given time point
    and force them lie in a common epipolar plane. To do this, we find the equation of a plane 
    that contains the optical centers of the left and right eyes (loc_L and loc_R), 
    and has the minimum least squares distance from the fixation points (or probably more accurately 
    gaze directions, since theyre monocular) of the two eyes (fix_L and fix_R). All vectors are expected
    to be 3X1

    The functional optionally checks to make sure the resulting elevation angles are the same
    for each eye, and generates at plot of the original and new gaze vectors.
    """


    # ADJUST fix_L AND fix_R to still lie along gaze direction, but each be unit
    # length from eye locations loc_L and p2

    fix_L = fix_L - loc_L                       # translate fix_L into coordinate system with loc_L at origin
    fix_L = fix_L / np.linalg.norm(fix_L)       # make unit length
    fix_L = fix_L + loc_L                       # translate back in to cyclopean coordinates

    fix_R = fix_R - loc_R                       # translate fix_R into coordinate system with loc_R at origin
    fix_R = fix_R / np.linalg.norm(fix_R)       # make unit length
    fix_R = fix_R + loc_R                       # translate back in to cyclopean coordinates


    # SOLVE FOR PLANE THROUGH loc_L AND loc_R, WHILE MINIMIZING DISTANCE TO fix_L AND fix_R: 
    # CONSTRAINED LEAST-SQUARES (solve: y = bz)
    # the only degree of freedom is the rotation of the plane around the x axis that runs between loc_L and loc_R
    # so we find the linear least squares solution for y = bz
    # http://en.wikipedia.org/wiki/Linear_least_squares_(mathematics)

    y = np.matrix([[loc_L[1,0], fix_L[1,0], loc_R[1,0], fix_R[1,0]]]).T # y vector
    Z = np.matrix([[loc_L[2,0], fix_L[2,0], loc_R[2,0], fix_R[2,0]]]).T # Z vector

    b = np.linalg.inv(Z.T * Z) * Z.T * y    # least squares solution for b


    # PROJECT VECTORS ONTO PLANE to get new point coordinates
    fix_L_new  = np.array([[fix_L[0,0], b*fix_L[2,0], fix_L[2,0]]]).T # project (x and z are the same, y = bz)
    fix_R_new  = np.array([[fix_R[0,0], b*fix_R[2,0], fix_R[2,0]]]).T # project


    if check_results:
        
        # check that elevation angles are the same
        P = np.array([0, -1, b])    # vector normal to the epipolar plane
        P = P / np.linalg.norm(P)               # normalize

        # angle between original vectors and epipolar plane
        th_L = np.degrees(np.arcsin(np.dot( P, fix_L / np.linalg.norm(fix_L))))
        th_R = np.degrees(np.arcsin(np.dot( P, fix_R / np.linalg.norm(fix_R ))))
        
        th_L_new = np.degrees(np.arcsin(np.dot( P, (fix_L_new) / np.linalg.norm(fix_L_new))))
        th_R_new = np.degrees(np.arcsin(np.dot( P, (fix_R_new) / np.linalg.norm(fix_R_new))))

        print "Align Eyes Adjustment (L/R in deg)", th_L, th_R
        if np.absolute(th_L_new) > 1e-100 or np.absolute(th_R_new) > 1e-100:
            raise ValueError("Projection to epipolar plane failed")

    if plot_results:

        #points for epipolar plane
        [xramp,zramp]   = np.meshgrid( np.linspace(-1.5,1.5, 5), np.linspace(-1.5,1.5, 5) );
        yramp           = zramp.copy()
        yramp           = b[0,0]*yramp

        # Need Bill's help to get the plotting working in iPython Notebook
        #%matplotlib inline
        fig = plt.figure()
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)
        axes = fig.gca(projection='3d')

        # epipolar plane
        axes.plot_wireframe( xramp, yramp, zramp);

        #cyclopean eye and interocular axis
        axes.plot( [0.], [0.], [0.],'ko')
        axes.plot( [loc_R[0,0],loc_L[0,0]], [loc_R[1,0],loc_L[1,0]], [loc_R[2,0],loc_L[2,0]],'c')
        # original gaze vectors
        axes.plot( [loc_L[0,0],fix_L[0,0]], [loc_L[1.0],fix_L[1,0]], [loc_L[2,0],fix_L[2,0]],'k-o')
        axes.plot( [loc_R[0,0],fix_R[0,0]], [loc_R[1,0],fix_R[1,0]], [loc_R[2,0],fix_R[2,0]],'k-o')
        # new gaze vectors
        axes.plot( [loc_L[0,0],fix_L_new[0,0]], [loc_L[1,0],fix_L_new[1,0]], [loc_L[2,0],fix_L_new[2,0]],'g-o')
        axes.plot( [loc_R[0,0],fix_R_new[0,0]], [loc_R[1,0],fix_R_new[1,0]], [loc_R[2,0],fix_R_new[2,0]],'r-o')

        #box on; grid on; axis square; axis tight;
        #xlabel( 'x' ); ylabel( 'y' ); zlabel( 'z' );
        #hold off;


    return fix_L_new, fix_R_new


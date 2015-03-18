import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
except ImportError as e:
    pass

def align_eyes(gaze_df, ipd):

    df = gaze_df.loc[:, (['left', 'right'], ['bref x', 'bref y', 'bref z'])].copy()
    fix_L = []
    fix_R = []
    idxs = []

    for idx, row in df.iterrows():
        new_fix_L, new_fix_R = fit_plane(np.atleast_2d(row['left'].values).T,
                                         np.atleast_2d(row['right'].values).T,
                                         ipd, check_results=True)

        fix_L.append(new_fix_L.squeeze())
        fix_R.append(new_fix_R.squeeze())
        idxs.append(idx)

    midx = pd.MultiIndex.from_tuples(idxs, names=['rep', 'time'])
    cols = ['bref x', 'bref y', 'bref z']
    fix_L = pd.DataFrame(np.array(fix_L), index=midx,
                         columns=pd.MultiIndex.from_product([['left'], cols]))
    fix_R = pd.DataFrame(np.array(fix_R), index=midx,
                         columns=pd.MultiIndex.from_product([['right'], cols]))

    gaze_df.loc[:, ('left',['bref x', 'bref y', 'bref z'])] = fix_L
    gaze_df.loc[:, ('right',['bref x', 'bref y', 'bref z'])] = fix_R


def fit_plane(fix_L, fix_R, ipd, check_results=False, plot_results=False):
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
    loc_L = np.array([[-ipd/2., 0, 0]]).T
    loc_R = np.array([[ipd/2., 0, 0]]).T

    assert fix_L.shape == (3,1)
    assert fix_R.shape == (3,1)

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

    y = np.matrix([[loc_L[1,0], fix_L[1,0], loc_R[1,0], fix_R[1,0]]]).T # y column vector
    Z = np.matrix([[loc_L[2,0], fix_L[2,0], loc_R[2,0], fix_R[2,0]]]).T # Z column vector

    b = np.linalg.inv(Z.T * Z) * Z.T * y    # least squares solution for b


    # PROJECT VECTORS ONTO PLANE to get new point coordinates
    fix_L_new  = np.array([[fix_L[0,0], b*fix_L[2,0], fix_L[2,0]]]).T # project (x and z are the same, y = bz)
    fix_R_new  = np.array([[fix_R[0,0], b*fix_R[2,0], fix_R[2,0]]]).T # project


    # TODO Move check results to testing code
    if check_results:

        # check that elevation angles are the same
        P = np.array([0, -1, b])    # vector normal to the epipolar plane
        P = P / np.linalg.norm(P)               # normalize

        # angle between original vectors and epipolar plane
        th_L = np.degrees(np.arcsin(np.dot( P, fix_L / np.linalg.norm(fix_L))))
        th_R = np.degrees(np.arcsin(np.dot( P, fix_R / np.linalg.norm(fix_R ))))

        th_L_new = np.degrees(np.arcsin(np.dot( P, (fix_L_new) / np.linalg.norm(fix_L_new))))
        th_R_new = np.degrees(np.arcsin(np.dot( P, (fix_R_new) / np.linalg.norm(fix_R_new))))

#         print "Align Eyes Adjustment (L/R in deg)", th_L, th_R
        if np.absolute(th_L_new) > 1e-10 or np.absolute(th_R_new) > 1e-10:
            raise ValueError("Projection to epipolar plane failed")

    if plot_results:
        #points for epipolar plane
        [xramp,zramp]   = np.meshgrid( np.linspace(-1.5,1.5, 5), np.linspace(-0.1,1.1, 5) );
        yramp           = zramp.copy()
        yramp           = b[0,0]*yramp

        fig = plt.figure()
        ax = plt.subplot(111,projection='3d')

        # # epipolar plane
        ax.plot_wireframe( xramp, zramp, yramp);

        #cyclopean eye and interocular axis
        ax.plot( [0.], [0.], [0.],'ko', zdir='y')
        ax.plot( [loc_R[0,0],loc_L[0,0]], [loc_R[1,0],loc_L[1,0]], [loc_R[2,0],loc_L[2,0]],'c', zdir='y')
        # original gaze vectors
        ax.plot( [loc_L[0,0],fix_L[0,0]], [loc_L[1.0],fix_L[1,0]], [loc_L[2,0],fix_L[2,0]],'k-o', zdir='y')
        ax.plot( [loc_R[0,0],fix_R[0,0]], [loc_R[1,0],fix_R[1,0]], [loc_R[2,0],fix_R[2,0]],'k-o', zdir='y')
        # new gaze vectors
        ax.plot( [loc_L[0,0],fix_L_new[0,0]], [loc_L[1,0],fix_L_new[1,0]], [loc_L[2,0],fix_L_new[2,0]],'g-o', zdir='y')
        ax.plot( [loc_R[0,0],fix_R_new[0,0]], [loc_R[1,0],fix_R_new[1,0]], [loc_R[2,0],fix_R_new[2,0]],'r-o', zdir='y')

        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')

        ax.set_zlim3d(.1, -.1)  # Adjust the vertical limits
        ax.set_ylim3d(-.1, 1.1)  # Adjust the depth limits

    return fix_L_new, fix_R_new




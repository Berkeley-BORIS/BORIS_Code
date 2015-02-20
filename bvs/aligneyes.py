import numpy as np
import numpy.linalg as linalg

def align_eyes(fix_L, fix_R, loc_L, loc_R, plot_Rsults=0):
    """
    The data from the Eyelink rarely results in gaze directions for the two eyes that actually
    intersection. The goal of this function is to take the skew gaze rays for a given time point
    and force them lie in a common epipolar plane. To do this, we find the equation of a plane 
    that contains the optical centers of the left and right eyes (loc_L and loc_R), 
    and has the minimum least squares distance from the fixation points (or probably more accurately 
    gaze directions, since theyre monocular) of the two eyes (fix_L and fix_R).

    The functional optionally generates at plot of the original and new gaze vectors.
    """


# ADJUST fix_L AND fix_R to still lie along gaze direction, but each be unit
# length from eye locations loc_L and p2

fix_L = fix_L - loc_L                   # translate fix_L into coordinate system with loc_L at origin
fix_L = fix_L / linalg.norm(fix_L)      # make unit length
fix_L = fix_L + loc_L                   # translate back in to cyclopean coordinates

fix_R = fix_R - loc_R                   # translate fix_R into coordinate system with loc_R at origin
fix_R = fix_R / linalg.norm(fix_R)      # make unit length
fix_R = fix_R + loc_R                   # translate back in to cyclopean coordinates


# SOLVE FOR PLANE THROUGH loc_L AND loc_R, WHILE MINIMIZING DISTANCE TO fix_L AND fix_R: 
# CONSTRAINED LEAST-SQUARES (solve: y = bz)
# the only degree of freedom is the rotation of the plane around the x axis that runs between loc_L and loc_R
# so we find the linear least squares solution for y = bz
# http://en.wikipedia.org/wiki/Linear_least_squares_(mathematics)

y      = [p1(2) p2(2) q1(2) q2(2)]      # y vector
Z      = [p1(3) p2(3) q1(3) q2(3)]      # Z vector
b      = inv(Z'*Z)*Z'*y;                # least squares solution for b

# PROJECT VECTORS ONTO PLANE to get new point coordinates
p2_new  = [p2(1) b*p2(3) p2(3)]; # project (x and z are the same, y = bz)
q2_new  = [q2(1) b*q2(3) q2(3)]; # project



    function [p2_new,q2_new] = fitplane(p2,q2,p1,q1)
%
% find the equation of a plane that (1) contains points p1 and q1, and (2)
% has the minimum least squares distance from p2 and q2
%
% p1 and q1 are head referenced location vectors for left and right eye
%
% p2 & q2 are head referenced direction vectors for left and
% right eye fixation points
function [p2_new,q2_new] = fitplanetest(p2,q2,p1,q1)
%
% find the equation of a plane that (1) contains points p1 and q1, and (2)
% has the minimum least squares distance from p2 and q2
%
% p1 and q1 are head referenced location vectors for left and right eye
%
% p2 & q2 are head referenced direction vectors for left and
% right eye fixation points


plotResults = 1;

%%% ADJUST p2 AND q2 to still lie along gaze direction, but each be unit
%%% length from eye locations p1 and p2
p2 = p2 - p1;       % translate p2 into coordinate system with p1 at origin
p2 = p2 / norm(p2); % make unit length
p2 = p2 + p1;       % translate back in to cyclopean coordinates

q2 = q2 - q1;       % translate q2 into coordinate system with q1 at origin
q2 = q2 / norm(q2); % make unit length
q2 = q2 + q1;       % translate back in to cyclopean coordinates

%%% SOLVE FOR PLANE THROUGH p1 AND q1, WHILE MINIMIZING DISTANCE TO
%%% p2 AND q2: CONSTRAINED LEAST-SQUARES (solve: y = bz)
%% the only degree of freedom is the rotation of the plane around the x axis that runs between p1 and q1
%% so we find the least squares solution for this equation
%X      = [p1(1) p2(1) q1(1) q2(1)]';
y      = [p1(2) p2(2) q1(2) q2(2)]';
Z      = [p1(3) p2(3) q1(3) q2(3)]';
u      = inv(Z'*Z)*Z'*y;                % least squares solution
u      = [0 u 0]; % equation of plane y = ax + bz + c

%%% PROJECT VECTORS ONTO PLANE to get new point coordinates
p2_new  = [p2(1) u(2)*p2(3) p2(3)]; % project
q2_new  = [q2(1) u(2)*q2(3) q2(3)]; % project

if(plotResults)
    
    %%% MAKE SURE EVERYTHING WORKED
    P       = [u(1) -1 u(2)];                       % vector normal to plane
    P       = P / norm(P);
    thp     = asind( P*((p2-p1)'/norm(p2-p1)) );            % angle between original
    thq     = asind( P*((q2-q1)'/norm(q2-q1)) );            %  vector and plane
    thp_new = asind( P*((p2_new-p1)'/norm(p2_new-p1)) );    % angle between new
    thq_new = asind( P*((q2_new-q1)'/norm(q2_new-q1)) );    %  vector and plane
    
    if thp_new > 1e-100 || thp_new > 1e-100
        error('projection to epipolar plane failed');
    end
    
    %%% DISPLAY
    figure; hold on;
    
    % new epipolar plane
    [xramp,zramp]   = meshgrid( [-1.5:0.5:1.5], [-1.5:0.5:1.5] );
    yramp           = u(1)*xramp + u(2)*zramp + u(3);
    
    % plot points
    mesh( xramp, yramp, zramp, 'EdgeColor', [1 0 0] );
    plot3( 0, 0, 0, 'bo' );
    line( [p1(1) p2(1)], [p1(2) p2(2)], [p1(3) p2(3)] , 'marker','o');
    line( [q1(1) q2(1)], [q1(2) q2(2)], [q1(3) q2(3)] , 'marker','o');
    line( [p1(1) q1(1)], [p1(2) q1(2)], [p1(3) q1(3)] , 'marker','o');
    line( [p1(1) p2_new(1)], [p1(2) p2_new(2)], [p1(3) p2_new(3)] , 'marker','o', 'color','k');
    line( [q1(1) q2_new(1)], [q1(2) q2_new(2)], [q1(3) q2_new(3)] , 'marker','o', 'color','k');
    
    box on; grid on; axis square; axis tight;
    xlabel( 'x' ); ylabel( 'y' ); zlabel( 'z' );
    hold off;
    
end
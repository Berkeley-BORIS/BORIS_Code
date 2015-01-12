%%% Emily Cooper, Banks Lab, Nov 2011
%%% PARSE AND SAVE DATA FROM EYELINK2 
%%% 9/12 - no longer using BUTTONS now using MESSAGES
function [] = ET_parse(filename,start_line)
%clear all; close all;

fid = fopen( ['/Users/natdispstats/Documents/eyes/data_processing/data/' filename '/' filename '.ASC'], 'r' ); %open eye tracking data file

%%% INITIALIZE DATA STRUCTURES
cnt = 0; %counter
D   = []; % HREF data (SAMPLES)
% format for events:    TIMESTAMP XL YL PupilL XR YR PupilR

%read EVENT DATA also
SL = []; % saccade data (left eye)
SR = []; % saccade data (right eye)
FL = []; % fixation data (left eye)
FR = []; % fixation data (right eye)
BL = []; % blink data (left eye)
BR = []; % blink data (right eye)
T  = []; % radial target data (MSG *timestamp* RADIAL_TARGET *view_dist* *slant(deg)* *tilt(deg)* *1 or 0 for start or end*)
M  = []; % computer clock time (MSG *timestamp* DISPLAY_TIME *display time*
B  = []; % camera frame capture (*timestamp* button number 1=on, 0=off)
G  = []; % luminance ramping data used to apply pupil size dependent gaze correction (MSG *timestamp* PUPIL_CORRECT *x pixels* *y pixels* *1 or 0 for start or end*)
I  = []; %input
TG1 = []; %target start
TG2 = []; %target end
CH1 = []; %checkerboard start
CH2 = []; %checkerboard end
TK1 = []; %task start
TK2 = []; %task end


%%% PARSE DATA FILE (grab all samples, and then event info for each event type)
while(1)
    L = fgetl( fid ); % read a line from file
    cnt = cnt + 1;
    if( L == -1 ) % reached end of file
        fclose(fid);
        break;
    end
    if( cnt >= start_line ) % skip header, start at this line 
        if( strfind( L, '.....' ) ) %the dots indicate that the data in this line have good corneal reflections, ignore lines with no CRs or interpolated
        %if( isempty(strfind( L, ['.' sprintf('\t')] )) )
            if ~isempty(str2num( L(1:end-6) ))
                D(end+1,:) = str2num( L(1:end-6) );
            end
        elseif( strfind( L, 'ESACC L' ) )
            %SL(end+1,:) = str2num( L(9:end) );
            if ~isempty(str2num( L(9:end) ))
                SL(end+1,:) = str2num( L(9:end) );
            end
        elseif( strfind( L, 'ESACC R' ) )
            if ~isempty(str2num( L(9:end) ))
                SR(end+1,:) = str2num( L(9:end) );
            end
        elseif( strfind( L, 'EFIX L' ) )
            FL(end+1,:) = str2num( L(8:end) );
        elseif( strfind( L, 'EFIX R' ) )
            FR(end+1,:) = str2num( L(8:end) );
        elseif( strfind( L, 'EBLINK L' ) )
            BL(end+1,:) = str2num( L(10:end) );
        elseif( strfind( L, 'EBLINK R' ) )
            BR(end+1,:) = str2num( L(10:end) );
        elseif( strfind( L, 'MSG' ) )
            [token, remain] = strtok( L(4:end) );
            temp = [remain ' ' token];
            if( strfind( temp, 'RADIAL_TARGET ' ) )
                T(end+1,:) = str2num( temp(16:end) );
            elseif( strfind( temp, 'DISPLAY_TIME' ) )
                M(end+1,:) = str2num( temp(15:end) );  
            elseif( strfind( temp, 'PUPIL_CORRECT ' ) )
                G(end+1,:) = str2num( temp(16:end) );    
            elseif( strfind( temp, 'CHECKERBOARD_STARTED' ) )
                [token, remain] = strtok( temp );
                CH1(end+1,:) = str2num( remain );  
            elseif( strfind( temp, 'CHECKERBOARD_STOPPED' ) )
                [token, remain] = strtok( temp );
                CH2(end+1,:) = str2num( remain );     
            elseif( strfind( temp, 'TASK_STARTED' ) )
                [token, remain] = strtok( temp );
                TK1(end+1,:) = str2num( remain );  
            elseif( strfind( temp, 'TASK_STOPPED' ) )
                [token, remain] = strtok( temp );
                TK2(end+1,:) = str2num( remain );    
            elseif( strfind( temp, 'RADIAL_TARGET_STARTED' ) )
                [token, remain] = strtok( temp );
                TG1(end+1,:) = str2num( remain );  
            elseif( strfind( temp, 'RADIAL_TARGET_STOPPED' ) )
                [token, remain] = strtok( temp );
                TG2(end+1,:) = str2num( remain );     
            end
        elseif( strfind( L, 'BUTTON') )
            B(end+1,:) = str2num( L(7:end) );
        elseif( strfind( L, 'INPUT') )
            I(end+1,:) = str2num( L(6:end) );
        end
    end
end

%%MSG lines got rearranged in parsing. Move timestamp back to the start of
%%the line
if ~isempty(T)
    T = [T(:,5) T(:,1:4)];
end
if ~isempty(G)
    G = [G(:,4) G(:,1:3)];
end


%%%REMOVE SAMPLES TAKEN DURING BLINKS AND SACCADES

%%NOTE: There is a little weirdness in how time points get assigned to
%%fixations, saccades and blinks. Sometimes (rarely), an event is flagged as a
%%saccade, then reinterpreted as a blink. This leads to ESACC events where
%%the start and beginning timestamp are identical. The few timepoints
%%before the blink go unflagged (not fixation, saccade, or blink). 

D_orig = D; %retain all samples for posterity
if ~isempty(D)
for b = 1:size(BL,1)
    if (D(D(:,1) >= BL(b,1) & D(:,1) <= BL(b,2),:)) > 0
        display(['REMOVED LBLINK ' num2str(b)]);
    end
    D = D(D(:,1) < BL(b,1) | D(:,1) > BL(b,2),:); %trim out timestamps from D where timestamp falls between start and end of LE blink
end
for b = 1:size(BR,1)
    if (D(D(:,1) >= BR(b,1) & D(:,1) <= BR(b,2),:)) > 0
        display(['REMOVED RBLINK ' num2str(b)]);
    end    
    D = D(D(:,1) < BR(b,1) | D(:,1) > BR(b,2),:); %trim out timestamps from D where timestamp falls between start and end of RE blink
end
for b = 1:length(SL)
    if (D(D(:,1) >= SL(b,1) & D(:,1) <= SL(b,2),:)) > 0
        display(['REMOVED LSACC ' num2str(b)]);
    end    
    D = D(D(:,1) < SL(b,1) | D(:,1) > SL(b,2),:); %trim out timestamps from D where timestamp falls between start and end of LE saccade
end
for b = 1:length(SR)
    if (D(D(:,1) >= SR(b,1) & D(:,1) <= SR(b,2),:) > 0)
        display(['REMOVED RSACC ' num2str(b)]);
    end    
    D = D(D(:,1) < SR(b,1) | D(:,1) > SR(b,2),:); %trim out timestamps from D where timestamp falls between start and end of RE saccade
end
end

save(['/Users/natdispstats/Documents/eyes/data_processing/data/' filename '/' filename '.mat'],'D','D_orig','SL','SR','FL','FR','BL','BR','T','M','B','G','I','CH1','CH2','TK1','TK2','TG1','TG2')
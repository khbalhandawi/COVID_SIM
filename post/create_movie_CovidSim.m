clearvars
clc
format compact
close all

%% Preprocess CovidSim
global days
days = 1;

prefix = '0-6-Canada_PC7_CI_HQ_SD_V2_R0=3.0';
folder = [pwd,'\maps\',prefix,'.ge\'];
processed_folder = '.\maps\processed';

a=dir([folder, '\*.png']); % number of pngs files in directory
n_imgs=size(a,1);

% % Comment to avoid clearing folder
% check_folder('./maps/processed')
% clear_folder('./maps/processed')
% for i = n_imgs-720:1:(n_imgs-720 + 720)
%     Visualize_CovSim(folder,prefix,i)
% end

%% Single image
Visualize_CovSim(folder,prefix,n_imgs-720+90,'eps',false)

%% CovidSim
Create a movie
a=dir([processed_folder, '\*.png']); % number of pngs files in directory
n_imgs=size(a,1);

img_i = 1; images = {};
 for i=1:2:n_imgs
     images{img_i} = imread([processed_folder,'\',sprintf('map_%d.png',i)]);
     img_i = img_i + 1;
 end

% create the video writer with 30 fps
writerObj = VideoWriter('.\maps\Canada.avi');
writerObj.FrameRate = 15;
% open the video writer
open(writerObj);
% write the frames to the video
for u=1:length(images) 
    % convert the image to a frame
    frame = im2frame(images{u});
    writeVideo(writerObj, frame);
end
% close the writer object
close(writerObj);
implay('.\maps\Canada.avi');

%% COVID_SIM_UI
% Create a movie
processed_folder = '.\maps\animation_OutOfControl';
a=dir([processed_folder, '\*.png']); % number of pngs files in directory
n_imgs=size(a,1);
N = 1.2;

img_i = 1; images = {};
for i=1:8:n_imgs
    inputImage = imread([processed_folder,'\',sprintf('sim_%05d.png',i)]);
    [rows, columns, numColorChannels] = size(inputImage);
    numOutputRows = round(rows/N);
    numOutputColumns = round(columns/N);
    outputImage = imresize(inputImage, [numOutputRows, numOutputColumns]);
    
    images{img_i} = outputImage;
    img_i = img_i + 1;
end
 
% create the video writer with 30 fps
writerObj = VideoWriter('.\maps\ABM.avi');
writerObj.FrameRate = 15;
% open the video writer
open(writerObj);
% write the frames to the video
for u=1:length(images)    
    % convert the image to a frame
    frame = im2frame(images{u});
    writeVideo(writerObj, frame);
end
% close the writer object
close(writerObj);
implay('.\maps\ABM.avi');


%% Utility functions
%=========================================================%
%                     Clear directory                     %
%=========================================================%
function clear_folder(folder)
    % clear all files inside folder %
    delete([folder,'\*'])
end
            
%=========================================================%
%                 Create empty directory                  %
%=========================================================%
function check_folder(folder)
	% check if folder exists, make if not present %
    if not(isfolder(folder))
        mkdir(folder)
    end
end
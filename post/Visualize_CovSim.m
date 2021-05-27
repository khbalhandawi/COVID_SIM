clearvars
clc
format compact
close all

addpath borders

figure(1)
% borders('United Kingdom','k')
borders('Canada','k')
axis tight
hold on

% Most populous Canadian municipalities
Canadian_cities = {...
    {'Vancouver'        ,[-123.1207, 49.2827]},...
    {'Montreal'         ,[-73.5673 , 45.5017]},...
    {'Toronto'          ,[-79.3832 , 43.6532]},...
    {'Calgary'          ,[-114.0719, 51.0447]},...
    {'Winnipeg'         ,[-97.1384 , 49.8951]},...
    {'Saskatoon'        ,[-106.6702, 52.1579]},...
    {'St_Johns'         ,[-64.7782 , 46.0878]},...
    {'Yellowknife'      ,[-114.3718, 62.4540]},...
    {'Halifax'          ,[-63.5752 , 44.6488]},...
    {'Iqaluit'          ,[-68.5170 , 63.7467]},...
    {'Charlottetown'    ,[-63.1311 , 46.2382]},...
    {'Whitehorse'       ,[-135.0568, 60.7212]},...
    };

for i = 1:1:length(Canadian_cities)
    city_details = Canadian_cities{i};
    name = city_details{1};
    coordinates = city_details{2};
    plot(coordinates(1),coordinates(2),'r*','markersize',5,'linewidth',2)
end

% axis equal %so images show up with right aspect ratio
% set(gca, 'XLimMode', 'manual', 'YLimMode', 'manual');

xl = get(gca, 'XLim');
yl = get(gca, 'YLim');

sizefactor_x = -0.001;    %play with this to get right size, smaller is bigger image
sizefactor_y = 0.139;    %play with this to get right size, smaller is bigger image

lower_corner = [0,-5.9];
imgx = [sizefactor_x * xl(2) + (1-sizefactor_x) * xl(1), xl(2)] + lower_corner(1);
imgy = [yl(2),sizefactor_y * yl(2) + (1-sizefactor_y) * yl(1)] + lower_corner(2);

% CovidSim_img_path = "C:\Users\Khalil\Desktop\repos\COVID_SIM\post\maps\0-3-United_Kingdom_PC7_CI_HQ_SD_R0=3.0.01295.bmp";
% [CovidSim_img, map] = imread(CovidSim_img_path);

CovidSim_img_path = "C:\Users\Khalil\Desktop\repos\COVID_SIM\post\maps\1-Canada_PC7_CI_HQ_SD_R0=3.0.00842.png";
[CovidSim_img, map] = imread(CovidSim_img_path, 'png', 'BackgroundColor', [1 1 1]);

im_size_pixels = size(CovidSim_img);
rect = [0, 0, floor(im_size_pixels(1)*0.1), im_size_pixels(2)];

for i = 1:1:im_size_pixels(1)
    for j = 1:1:im_size_pixels(2)
        if i >= rect(1) && i <= rect (3) && j >= rect(2) && j <= rect (4)
            CovidSim_img(i,j) = 0.0;
        end
    end
end

image(imgx, imgy, CovidSim_img);
colormap(gca,map);

% CovidSim_img.AlphaData = 0.1;    % set transparency to maximum cloud value
set(gca,'children',flipud(get(gca,'children')))
set(gca,'xtick',[])
set(gca,'xticklabel',[])
set(gca,'ytick',[])
set(gca,'yticklabel',[])
set(gca,'Visible','off')
set(gcf,'color','w');
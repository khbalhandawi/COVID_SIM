function Visualize_CovSim(folder,prefix,index,format,print_text)

global days

if (nargin==3)
    format = 'png';
    print_text = true;
end

addpath borders
addpath export_fig

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
    {'Whitehorse'       ,[-135.0568, 60.7212]},...
    {'Edmonton'         ,[-113.4938, 53.5461]},...
    };

for i = 1:1:length(Canadian_cities)
    city_details = Canadian_cities{i};
    name = city_details{1};
    coordinates = city_details{2};
    cities = plot(coordinates(1),coordinates(2),'b+','markersize',2,'linewidth',0.5);
end

% axis equal %so images show up with right aspect ratio
% set(gca, 'XLimMode', 'manual', 'YLimMode', 'manual');

xl = get(gca, 'XLim');
yl = get(gca, 'YLim');

% Scale and stretch image
sizefactor_x = -0.001;    %play with this to get right size, smaller is bigger image
sizefactor_y = 0.139;    %play with this to get right size, smaller is bigger image

lower_corner = [0,-5.9];
imgx = [sizefactor_x * xl(2) + (1-sizefactor_x) * xl(1), xl(2)] + lower_corner(1);
imgy = [yl(2),sizefactor_y * yl(2) + (1-sizefactor_y) * yl(1)] + lower_corner(2);

% Load image
CovidSim_img_path = [folder,prefix,sprintf('.%05d',index),'.png'];
[CovidSim_img, map] = imread(CovidSim_img_path, 'png', 'BackgroundColor', [1 1 1]);

% Remove borders from raw image (crop)
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

% Clean up figure window
% CovidSim_img.AlphaData = 0.1;    % set transparency to maximum cloud value
set(gca,'children',flipud(get(gca,'children')))
set(gca,'xtick',[])
set(gca,'xticklabel',[])
set(gca,'ytick',[])
set(gca,'yticklabel',[])
set(gca,'Visible','off')
set(gcf,'color','w');

% Create a legend for the plot
hold on;
[h1,] = plot(NaN,NaN,'r*','markersize',3,'linewidth',2);
[h2,] = plot(NaN,NaN,'g*','markersize',3,'linewidth',2);
[h3,] = plot(NaN,NaN,'b+','markersize',5,'linewidth',1);

legend_objects = [h1,h2,h3];
legend_labels = [{'infections'},{'recoveries and fatalitites'},{'major Canadian municipalities'}];

lh = legend(legend_objects,legend_labels,'Orientation','vertical');

% Position legend
rect = [0.2, 0.85, .27, .0528];
set(lh, 'Position', rect, 'interpreter', 'latex', 'fontsize', 11)

% Add text label
if print_text
    text(-140, 45,['Day ',num2str(ceil(days/2))], 'interpreter', 'latex', 'fontsize', 13)
end

% Save figure image
% if you don't like the border around it type:
iptsetpref('ImshowBorder','tight');
% code to show image number i

% saveas(gcf,['./maps/processed/filename_' num2str(index) '.',format]);
export_fig(['./maps/processed/map_' num2str(days) '.',format],'-p0.002','-r200'); 

close(figure(1))

days = days + 1;
end
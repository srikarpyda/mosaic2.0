function map = tileMap(imgPath, imgType)
%TILEMAP Summary of this function goes here
%   Detailed explanation goes here
map= containers.Map();
images  = dir([imgPath '/*.' imgType]);
N = length(images);


for i=1:N
    imageName=images(i).name;
    image=imread([imgPath '/' imageName],imgType);

  %% imread(imgPath+imageName);
    average(1:3) = AverageColorImage(image);
   % average + " 3Tuple "
    map(imageName)=average;
end



end


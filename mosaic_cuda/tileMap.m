function map = tileMap(imgPath, imgType)
%TILEMAP Summary of this function goes here
%   Detailed explanation goes here
map= containers.Map();
images  = dir([imgPath '/*.' imgType]);
N = length(images);


for i=1:N
    imageName=images(i).name;
    image=imread([imgPath '/' imageName],imgType);

    average(1:3) = AverageColorImage(image);
    map(imageName)=average;
end



end


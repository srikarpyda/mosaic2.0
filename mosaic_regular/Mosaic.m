function mosaic = Mosaic( img,tilePath, tileType, tileSize)

tic 

%% construct tile map
tiles = tileMap(tilePath, tileType);


%% read in image
image = imread(img);
[imgHeight, imgWidth, colours] = size(image);

%%tile indices
i=1;
j=1;

%% need to figure out how to declare size of mosaic ahead of time so that its not slow
% Good image sizes to do on GPU: 2000x2000 image, 40x40 tile or 25x25 tile
% Bottlenecks: Nearest (and Distance, which gets called during Nearest)
% imresize, cell2mat, AverageColorTile

% Use CUDA to parallelize the whole process of getting the average color of
% a tile and finding the nearest image

% Use MATLAB built-ins to do imresize and cell2mat on GPU

%%

for y =1:tileSize:imgHeight-tileSize+1
    for x =1:tileSize:imgWidth-tileSize+1
      
         %% find rgb tuple for tile
         rgbValues(1:3) = AverageColorTile(image,x,y,x+tileSize-1,y+tileSize-1);

         %%find nearest image
         nearestImg= Nearest(rgbValues,tiles);
         nearestImgChar= char(nearestImg);
         
        nearestImage= imread(strcat(tilePath,nearestImgChar));

         %% resize image to fit tile-size
        nearestImage= imresize(nearestImage, [tileSize tileSize]);

         %% construct cell-array mosaic
         mosaic{i,j} = double(nearestImage);
         
         %%iterate cell-array indexes
         i=i+1;
         
    end
    j=j+1;
end

toc

mosaic = uint8(cell2mat(mosaic));
figure;
imshow(mosaic);

imwrite(mosaic,'mosaic.png');

end


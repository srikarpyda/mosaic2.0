function mosaic = Mosaic( img,tilePath, tileType, tileSize)

%% construct tile map
tiles = tileMap(tilePath, tileType);


%% read in image
image = imread(img);
[imgHeight, imgWidth, colours] = size(image);

%%tile indices
i=1;
j=1;

%% need to figure out how to declare size of mosaic ahead of time so that its not slow

%% iterate through tiled indices of image
%% maybe consider iterating different lengths over x and y


for y =1:tileSize:imgHeight-tileSize
    for x =1:tileSize:imgWidth-tileSize
      
         %% find rgb tuple for tile
         rgbValues(1:3) = AverageColorTile(image,x,y,x+tileSize,y+tileSize);

         %%find nearest image
         nearestImg= Nearest(rgbValues,tiles);
         nearestImgChar= char(nearestImg);
        %% nearestImage= imread(tilePath+(string(nearestImg(1))));
         
        nearestImage= imread(strcat(tilePath,nearestImgChar));

        %figure;
        %imshow(nearestImage)
         %% resize image to fit tile-size
        nearestImage= imresize(nearestImage, [tileSize tileSize]);
         
         %figure;
        %imshow(nearestImage)
         
         %% construct cell-array mosaic
         mosaic{i,j} = double(nearestImage);
         
         %%iterate cell-array indexes
         i=i+1;
         
         
         
         
        %% xRange=(x:x+tileSize);
        %% yRange=(y:y+tileSize);
        %% mosaic(xRange,yRange,:)=nearestImage(xRange,yRange,:);
         
    end
    j=j+1;
end


mosaic = uint8(cell2mat(mosaic));
imshow(mosaic)

imwrite(mosaic,'mosaic.png');

end


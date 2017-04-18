function mosaic = mosaic_cuda( img,tilePath, tileType, tileSize)

%% construct tile map, read in target image
tiles = tileMap(tilePath, tileType);
tileValues = values(tiles);
tileKeys = keys(tiles);

numSamples = size(tileValues);

image = imread(img);
[imgHeight, imgWidth, colours] = size(image);

%% Initializing arrays and transferring data to GPU

reds = zeros(1, numSamples(2));
blues = zeros(1, numSamples(2));
greens = zeros(1, numSamples(2));

for i = 1: numSamples(2)
    temp = cell2mat(tileValues(i));
    reds(1, i) = temp(1);
    greens(1, i) = temp(2);
    blues(1, i) = temp(3);
end

reds(11)
reds(1)

numTiles = (imgHeight/tileSize);
nearestTiles = ones(numTiles*numTiles,1,'int32');

imGPU = gpuArray(int32(image));
redGPU = gpuArray(int32(reds));
greenGPU = gpuArray(int32(greens));
blueGPU = gpuArray(int32(blues));
nearestTilesGPU = gpuArray(nearestTiles);



%% GPU Function Invocation

kernel = parallel.gpu.CUDAKernel(...
    'mosaic_cuda.ptx',...
    'mosaic_cuda.cu');

threadsPerBlock = 16;
numBlocks = ceil(numTiles/threadsPerBlock);

%Each block has 16 x 16 threads
kernel.ThreadBlockSize = [threadsPerBlock, threadsPerBlock, 1];
kernel.GridSize = [numBlocks, numBlocks];
nearestTilesGPU= feval(kernel, nearestTilesGPU, imGPU, redGPU, greenGPU, blueGPU, numSamples(2), tileSize, numTiles, threadsPerBlock);
nearestImageIndices =  gather(nearestTilesGPU);
for i=1:numTiles
    for j=1:numTiles
        imgIdx = nearestImageIndices(((i-1)*numTiles)+j);
        nearestImage = imread(tileKeys{imgIdx+1});
        nearestImage= imresize(nearestImage, [tileSize tileSize]);

        mosaic{i,j}=double(nearestImage);
    end
end


mosaic = uint8(cell2mat(mosaic));
figure;
imshow(mosaic);

imwrite(mosaic,'mosaic.png');

%     j COLUMN INDEX
%   i 1 2 3 4
% R   5 6 7 8
% O   9 10 11 12
% W   13 14 15 16
% IDX row major indexing


%nearestImageIndices = gather(nearestTilesGPU)



%% need to figure out how to declare size of mosaic ahead of time so that its not slow
% Good image sizes to do on GPU: 2000x2000 image, 40x40 tile or 25x25 tile
% Bottlenecks: Nearest (and Distance, which gets called during Nearest)
% imresize, cell2mat, AverageColorTile

% Use CUDA to parallelize the whole process of getting the average color of
% a tile and finding the nearest image

% Use MATLAB built-ins to do imresize and cell2mat on GPU




%%

% for y =1:tileSize:imgHeight-tileSize+1
%     for x =1:tileSize:imgWidth-tileSize+1
%
%          %%find nearest image
%          nearestImg= Nearest(rgbValues,tiles);
%          nearestImgChar= char(nearestImg);
%
%          nearestImage= imread(strcat(tilePath,nearestImgChar));
%
%          %% resize image to fit tile-size
%         nearestImage= imresize(nearestImage, [tileSize tileSize]);
%
%          %% construct cell-array mosaic
%          mosaic{i,j} = double(nearestImage);
%
%          %%iterate cell-array indexes
%          i=i+1;
%
%     end
%     j=j+1;
% end
%
%
% mosaic = uint8(cell2mat(mosaic));
% figure;
% imshow(mosaic);
%
% imwrite(mosaic,'mosaic.png');

end


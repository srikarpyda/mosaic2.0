function mosaic = mosaic_cuda( img,tilePath, tileType, tileSize)

tic 

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

toc

mosaic = uint8(cell2mat(mosaic));
figure;
imshow(mosaic);

imwrite(mosaic,'mosaic.png');


end


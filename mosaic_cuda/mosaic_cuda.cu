#include <math.h>
#include <stdio.h>
#include <stdint.h>

__device__ __forceinline__ 
int getLinearIndex(int row, int col, int slice, int nRows, int nCols){
	
	//image indexing is column major
	return slice*nRows*nCols + col * nRows + row;
}

__device__ __forceinline__ 
double getTileAverage(int row, int col, int slice, int tileSize, int imageSize, int* image){
	
	int i, j;
	double sum = 0.0;
	double size = tileSize * tileSize;

	for(i = 0; i < tileSize; i++){
		for(j = 0; j < tileSize; j++){

			int tempRow = row + i;
			int tempCol = col + j;
			int tempLinearIndex = getLinearIndex(tempRow, tempCol, slice, imageSize, imageSize);

			sum = sum + image[tempLinearIndex];

			if(slice == 0){
				printf("IMAGE VALUE: %d ; SUM: %d\n", image[tempLinearIndex], sum);
			}
		}
	}
	printf("SUM: %d \n", sum);

	return sum/size;
}


//tileSize = side length of tile
//numTiles = num of tiles per side
//threadsPerBlock = fixed at 16

template <typename T>
__device__ __forceinline__
void mosaic(T* image, const T* reds, const T* greens, const T* blues, int numSamples, int* nearestTiles, int tileSize, 
			int numTiles, int threadsPerBlock){

	//Calculate what tile this is
	int tileRowIdx = blockIdx.x * threadsPerBlock + threadIdx.x;
	int tileColIdx = blockIdx.y * threadsPerBlock + threadIdx.y;

	//Calculate top-left pixel of current tile,
	int pixelRow = tileRowIdx * tileSize;
	int pixelCol = tileColIdx * tileSize;

	//targetImageSize = side length of target image in pixels 
	int targetImageSize = tileSize * numTiles;

	if(pixelRow >= targetImageSize || pixelCol >= targetImageSize){
		return;
	}

	double avgR = getTileAverage(pixelRow, pixelCol, 0, tileSize, targetImageSize, image);
	double avgG = getTileAverage(pixelRow, pixelCol, 1, tileSize, targetImageSize, image);
	double avgB = getTileAverage(pixelRow, pixelCol, 2, tileSize, targetImageSize, image);

	printf("Tuple of averages: %d, %d, %d \n", avgR, avgG, avgB);

	double minDistance = -1;	
	int minDistanceIndex = -1;

	int i;
	for(i = 0; i < numSamples; i = i+1){

		double tempDistance = fabs(pow(avgR-reds[i], 2) + pow(avgB-blues[i], 2) + pow(avgG-greens[i], 2));

		if(fabs(tempDistance) < minDistance || minDistance == -1){
			minDistance = tempDistance;
			minDistanceIndex = i;
		}
	}

	//Tiles are indexed in row-major order

	int tileLinearIndex = tileRowIdx * numTiles + tileColIdx;
	nearestTiles[tileLinearIndex] = minDistanceIndex;

	return;

}

__global__
void mosaic_cuda_double(int* nearestTile, int* image, const int* red, const int* green, const int* blue, int numSamples, 
				int tileSize, int numTiles, int threadsPerBlock){

			mosaic(image, red, green, blue, numSamples, nearestTile, tileSize, numTiles, threadsPerBlock);
    
			return;
}

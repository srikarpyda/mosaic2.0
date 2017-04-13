#include <math.h>

__device__ __forceinline__ 
int getLinearIndex(int row, int col, int slice, int nRows, int nCols){
	
	//image indexing is column major
	return slice*nRows*nCols + col * nRows + row;
}

__device__ __forceinline__ 
double getTileAverage(int row, int col, int slice, int tileSize, int imageSize, double* image){
	
	int i, j;
	double sum;

	for(i = 0; i < tileSize; i++){
		for(j = 0; j < tileSize; j++){

			int tempRow = row + i;
			int tempCol = col + j;
			int tempLinearIndex = getLinearIndex(tempRow, tempCol, slice, imageSize, imageSize);

			sum = sum + image[tempLinearIndex];
		}
	}

	return sum/(tileSize * tileSize);
}


//tileSize = side length of tile
//numTiles = num of tiles per side
//threadsPerBlock = fixed at 16

template <typename T>
__device__ __forceinline__
void mosaic(T* image, const T* reds, const T* greens, const T* blues, int numSamples, T* nearestTiles, int tileSize, 
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

	double minDistance = -1;	
	int minDistanceIndex = -1;

	int i;
	for(i = 0; i < numSamples; i = i+1){

		double tempDistance = sqrt(pow(avgR-reds[i], 2) + pow(avgG-greens[i], 2) + pow(avgB-blues[i], 2));

		if(tempDistance < minDistance || minDistance == -1){
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
void mosaic_cuda_double(double* image, const double* red, const double* green, const double* blue, int numSamples, 
				 double* nearestTile, int tileSize, int numTiles, int threadsPerBlock){

			mosaic(image, red, green, blue, numSamples, nearestTile, tileSize, numTiles, threadsPerBlock);
			return;
}

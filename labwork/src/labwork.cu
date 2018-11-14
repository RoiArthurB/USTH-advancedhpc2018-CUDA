#include <stdio.h>
#include <include/labwork.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#define ACTIVE_THREADS 4

int main(int argc, char **argv) {
    printf("USTH ICT Master 2018, Advanced Programming for HPC.\n");
    if (argc < 2) {
        printf("Usage: labwork <lwNum> <inputImage>\n");
        printf("   lwNum        labwork number\n");
        printf("   inputImage   the input file name, in JPEG format\n");
        return 0;
    }

    int lwNum = atoi(argv[1]);
    std::string inputFilename;

    // pre-initialize CUDA to avoid incorrect profiling
    printf("Warming up...\n");
    char *temp;
    cudaMalloc(&temp, 1024);

    Labwork labwork;
    if (lwNum != 2 ) {
        inputFilename = std::string(argv[2]);
        labwork.loadInputImage(inputFilename);
    }

    printf("Starting labwork %d\n", lwNum);
    Timer timer;
    timer.start();
    switch (lwNum) {
        case 1:
            labwork.labwork1_CPU();
            labwork.saveOutputImage("labwork2-cpu-out.jpg");
            printf("labwork 1 CPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            timer.start();
            labwork.labwork1_OpenMP();
            labwork.saveOutputImage("labwork2-openmp-out.jpg");
            break;
        case 2:
            labwork.labwork2_GPU();
            break;
        case 3:
            labwork.labwork3_GPU();
            labwork.saveOutputImage("labwork3-gpu-out.jpg");
            break;
        case 4:
            labwork.labwork4_GPU();
            labwork.saveOutputImage("labwork4-gpu-out.jpg");
            break;
        case 5:
            labwork.labwork5_CPU();
            labwork.saveOutputImage("labwork5-cpu-out.jpg");
            labwork.labwork5_GPU();
            labwork.saveOutputImage("labwork5-gpu-out.jpg");
            break;
        case 6:
            labwork.labwork6_GPU();
            labwork.saveOutputImage("labwork6-gpu-out.jpg");
            break;
        case 7:
            labwork.labwork7_GPU();
            labwork.saveOutputImage("labwork7-gpu-out.jpg");
            break;
        case 8:
            labwork.labwork8_GPU();
            labwork.saveOutputImage("labwork8-gpu-out.jpg");
            break;
        case 9:
            labwork.labwork9_GPU();
            labwork.saveOutputImage("labwork9-gpu-out.jpg");
            break;
        case 10:
            labwork.labwork10_GPU();
            labwork.saveOutputImage("labwork10-gpu-out.jpg");
            break;
    }
    printf("labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
}

void Labwork::loadInputImage(std::string inputFileName) {
    inputImage = jpegLoader.load(inputFileName);
}

void Labwork::saveOutputImage(std::string outputFileName) {
    jpegLoader.save(outputFileName, outputImage, inputImage->width, inputImage->height, 90);
}

void Labwork::labwork1_CPU() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    for (int j = 0; j < 100; j++) {     // let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

void Labwork::labwork1_OpenMP() {

}

int getSPcores(cudaDeviceProp devProp) {
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major) {
        case 2: // Fermi
            if (devProp.minor == 1) cores = mp * 48;
            else cores = mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            if (devProp.minor == 1) cores = mp * 128;
            else if (devProp.minor == 0) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        default:
            printf("Unknown device type\n");
            break;
    }
    return cores;
}

void Labwork::labwork2_GPU() {
    
}

void Labwork::labwork3_GPU() {
   
}

void Labwork::labwork4_GPU() {
   
}

// CPU implementation of Gaussian Blur
void Labwork::labwork5_CPU() {
    int kernel[] = { 0, 0, 1, 2, 1, 0, 0,  
                     0, 3, 13, 22, 13, 3, 0,  
                     1, 13, 59, 97, 59, 13, 1,  
                     2, 22, 97, 159, 97, 22, 2,  
                     1, 13, 59, 97, 59, 13, 1,  
                     0, 3, 13, 22, 13, 3, 0,
                     0, 0, 1, 2, 1, 0, 0 };
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = (char*) malloc(pixelCount * sizeof(char) * 3);
    for (int row = 0; row < inputImage->height; row++) {
        for (int col = 0; col < inputImage->width; col++) {
            int sum = 0;
            int c = 0;
            for (int y = -3; y <= 3; y++) {
                for (int x = -3; x <= 3; x++) {
                    int i = col + x;
                    int j = row + y;
                    if (i < 0) continue;
                    if (i >= inputImage->width) continue;
                    if (j < 0) continue;
                    if (j >= inputImage->height) continue;
                    int tid = j * inputImage->width + i;
                    unsigned char gray = (inputImage->buffer[tid * 3] + inputImage->buffer[tid * 3 + 1] + inputImage->buffer[tid * 3 + 2])/3;
                    int coefficient = kernel[(y+3) * 7 + x + 3];
                    sum = sum + gray * coefficient;
                    c += coefficient;
                }
            }
            sum /= c;
            int posOut = row * inputImage->width + col;
            outputImage[posOut * 3] = outputImage[posOut * 3 + 1] = outputImage[posOut * 3 + 2] = sum;
        }
    }
}

void Labwork::labwork5_GPU() {
    
}

//labwork6a
__global__ void binarize(uchar3 *input, uchar3 *output, int imgWidth, int imgHeight) {
    //Calculate tid
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidx >= imgWidth || tidy >= imgHeight) return;

    int tid =  tidx + (tidy * imgWidth);

    //Process pixel
    output[tid].z = output[tid].y = output[tid].x = (((int)(input[tid].x + input[tid].y + input[tid].z) / 3)/127)*255;
}
//labwork6b
__global__ void brightnessControll(uchar3 *input, uchar3 *output, int imgWidth, int imgHeight, int value) {
    //Calculate tid
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidx >= imgWidth || tidy >= imgHeight) return;

    int tid =  tidx + (tidy * imgWidth);

    //Process pixel
    output[tid].x = min(255, max(0, input[tid].x + value));
    output[tid].y = min(255, max(0, input[tid].y + value));
    output[tid].z = min(255, max(0, input[tid].z + value));  
}
//labwork6c
__global__ void blendingImg(uchar3 *input, uchar3 *output, int imgWidth, int imgHeight, uchar3 *secondImg, double weight) {
    //Calculate tid
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidx >= imgWidth || tidy >= imgHeight) return;

    int tid =  tidx + (tidy * imgWidth);

    //Process pixel
    output[tid].x = (weight * (double)input[tid].x) + ((1.0 - weight) * (double)secondImg[tid].x);
    output[tid].y = (weight * (double)input[tid].y) + ((1.0 - weight) * (double)secondImg[tid].y);
    output[tid].z = (weight * (double)input[tid].z) + ((1.0 - weight) * (double)secondImg[tid].z);
}
void Labwork::labwork6_GPU() {
    // Preparing var
    //======================
    //Calculate number of pixels
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    uchar3 *devInput;
    uchar3 *devImgProcessed;

    //Allocate CUDA memory    
    cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
    cudaMalloc(&devImgProcessed, pixelCount * sizeof(uchar3));
    // Copy CUDA Memory from CPU to GPU
    cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);

    // Processing
    //======================
    // Start GPU processing (KERNEL)
    //Create 32x32 Blocks
    dim3 blockSize = dim3(32, 32);
    dim3 gridSize = dim3((inputImage->width + (blockSize.x-1))/blockSize.x, 
        (inputImage->height  + (blockSize.y-1))/blockSize.y);
    
    //labwork6a
    //binarize<<<gridSize, blockSize>>>(devInput, devImgProcessed, inputImage->width, inputImage->height);
    
    //labwork6b
    brightnessControll<<<gridSize, blockSize>>>(devInput, devImgProcessed, inputImage->width, inputImage->height, 50);
    
    //labwork6c
    /*
    uchar3 *secondImg;

    cudaMalloc(&secondImg, pixelCount * sizeof(uchar3));
    cudaMemcpy(secondImg, inputSecondImage->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);

    blendingImg<<<gridSize, blockSize>>>(devInput, devImgProcessed, inputImage->width, inputImage->height, secondImg, 0.5);

    // Copy CUDA Memory from GPU to CPU
    cudaMemcpy(outputImage, devImgProcessed, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);

    cudaFree(&secondImg);
    */

    // Cleaning
    //======================
    // Free CUDA Memory
    cudaFree(&devInput);
    cudaFree(&devImgProcessed);
}

__global__ void grayscale2D(uchar3 *input, int *histo, int imgWidth, int imgHeight) {
    //Calculate tid
    unsigned int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidx >= imgWidth || tidy >= imgHeight) return;

    int localtid =  tidx + (tidy * imgWidth);

    //Process pixel
    unsigned int g = ((int)input[localtid].x + (int)input[localtid].y + (int)input[localtid].z) / 3;
    histo[localtid] = g;
}
__global__ void stretching(int *input, uchar3 *output, int imgWidth, int imgHeight, int min, int max) {
    //Calculate tid
    unsigned int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidx >= imgWidth || tidy >= imgHeight) return;

    int localtid =  tidx + (tidy * imgWidth);
    

    //Process pixel
    float num = (input[localtid] - min);
    float gStretch = ((num / (max - min)) * 255);
    
    //Store to output image
    output[localtid].z = output[localtid].y = output[localtid].x = (char)gStretch;
}
void Labwork::labwork7_GPU() {
    // GRAYSCALING
    //======================
    
    // Preparing var
    //----------------------
    //Calculate number of pixels
    int pixelCount = inputImage->width * inputImage->height;
//    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    uchar3 *devInput;
    int *devHisto;

    //Allocate CUDA memory    
    cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
    cudaMalloc(&devHisto, pixelCount * sizeof(int));
    // Copy CUDA Memory from CPU to GPU
    cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);
    
    //Create 32x32 Blocks
    dim3 blockSize = dim3(32, 32);
    dim3 gridSize = dim3((inputImage->width + (blockSize.x-1))/blockSize.x, 
        (inputImage->height  + (blockSize.y-1))/blockSize.y);

    // Processing
    //----------------------
    // Start GPU processing (KERNEL)
    grayscale2D<<<gridSize, blockSize>>>(devInput, devHisto, inputImage->width, inputImage->height);

    // Cleaning
    //----------------------
    cudaFree(&devInput);

    //======================
    // !GRAYSCALING

    // REDUCE
    //======================

    // Prep
    //----------------------
    int *temp = static_cast<int *>(malloc(pixelCount * sizeof(int)));    
    cudaMemcpy(temp, devHisto, pixelCount * sizeof(int), cudaMemcpyDeviceToHost);
    int hostMax = 0;
    int hostMin = 255;

    // Processing
    //----------------------    
    for (int i = 0; i < pixelCount; i++){
        hostMax = max(hostMax, temp[i]);
        hostMin = min(hostMin, temp[i]);
    }

    // Cleaning
    //----------------------
    // Free CPU Memory
    free(temp);

    //======================
    // !REDUCE


    // STRETCHING
    //======================
    // Prep
    //----------------------
    //Calculate number of pixels
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    uchar3 *devGray;

    //Allocate CUDA memory    
    cudaMalloc(&devGray, pixelCount * sizeof(uchar3));

    // Processing
    //----------------------
    // Start GPU processing (KERNEL)
    stretching<<<gridSize, blockSize>>>(devHisto, devGray, inputImage->width, inputImage->height, hostMin, hostMax);
    
    // Copy CUDA Memory from GPU to CPU
    cudaMemcpy(outputImage, devGray, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);
    
    // Cleaning
    //----------------------
    // Free CUDA Memory
    cudaFree(&devHisto);
    cudaFree(&devGray);
    //======================
    // !STRETCHING
}

typedef struct hsv {
    double *h, *s, *v;
} Hsv ;
__global__ void RGB2HSV(uchar3 *in, Hsv out, int imgWidth, int imgHeight) {
    //Calculate tid
    unsigned int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidx >= imgWidth || tidy >= imgHeight) return;
    
    int tid =  tidx + (tidy * imgWidth);
    double s, v, h = 0;
    
    // Scaling from [0 .. 255] to [0 .. 1]
    // Local var for optimization
    double pixelR = (double)in[tid].x / 255.0;
    double pixelG = (double)in[tid].y / 255.0;
    double pixelB = (double)in[tid].z / 255.0;
    
    double pxMax = max(pixelR, max(pixelG, pixelB));
    //int pxMin = // No need of this variable => Only 1 use 
    
    // V
    //======================

    v = pxMax;
    
    //======================
    // ! V

    // S
    //======================
    double delta = pxMax - min(pixelR, min(pixelG, pixelB));

    if( pxMax <= 0.0 ) { // NOTE: if Max is == 0, this divide would cause a crash
        // if max is 0, then r = g = b = 0              
        // s = 0, h is undefined
        s = 0.0;
    } else {
        s = (delta / pxMax);
    }
    //======================
    // ! S

    // H
    //======================
    if( pixelR >= pxMax ){ // between yellow & magenta
        h = ( pixelG - pixelB ) / delta;
        int decimal = static_cast<int>(h*10)%10;
        h = ((int)h % 6) + decimal * 0.1;
    }else{
        if( pixelG >= pxMax )
            h = 2.0 + ( ( pixelB - pixelR ) / delta );  // between cyan & yellow
        else
            h = 4.0 + ( ( pixelR - pixelG ) / delta );  // between magenta & cyan
    }
    
    // degrees
    h *= 60.0;
    //======================
    // ! H

    // Save new val in SoA
    out.h[tid] = h;
    out.s[tid] = s;
    out.v[tid] = v;
}
__global__ void HSV2RGB(Hsv in, uchar3 *out, int imgWidth, int imgHeight) {
    //Calculate tid
    unsigned int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidx >= imgWidth || tidy >= imgHeight) return;
    
    int tid =  tidx + (tidy * imgWidth);
    
    // Prepare local value for optimization
    double pixelH = in.h[tid];
    double pixelS = in.s[tid];
    double pixelV = in.v[tid];
    
    double d = pixelH / 60.0;
    // No need hi => Only 1 use
    double f = d - ((int)d % 6);
    
    double l = pixelV * (1.0 - pixelS);
    double m = pixelV * (1.0 - f * pixelS);
    double n = pixelV * (1.0 - (1.0 - f) * pixelS);
    
    // Calculate RGB values
    double r, g, b;
    switch ((int)pixelH / 60){
        case 0:
            r = pixelV;
            g = n;
            b = l;
            break;
        case 1:
            r = m;
            g = pixelV;
            b = l;
            break;
        case 2:
            r = l;
            g = pixelV;
            b = n;
            break;
        case 3:
            r = l;
            g = m;
            b = pixelV;
            break;
        case 4:
            r = n;
            g = l;
            b = pixelV;
            break;
        case 5:
        default:
            r = pixelV;
            g = l;
            b = m;
            break;
    }
    
    //Note : out[].x = R | out[].y = G | out[].z = B
    // [0..1] to [0..255]
    out[tid].x = (char)(r * 255);
    out[tid].y = (char)(g * 255);
    out[tid].z = (char)(b * 255);
}
void Labwork::labwork8_GPU() {

    // GRAYSCALING
    //======================
    
    // Preparing var
    //----------------------
    //Calculate number of pixels
    int pixelCount = inputImage->width * inputImage->height;
    //Kernel param
    dim3 blockSize = dim3(32, 32);
    dim3 gridSize = dim3((inputImage->width + (blockSize.x-1))/blockSize.x, 
        (inputImage->height  + (blockSize.y-1))/blockSize.y);
    
    //Kernel var
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    Hsv hsvArray;
    
    // Malloc arrays inside the structure
    cudaMalloc((void**)&hsvArray.h, pixelCount * sizeof(double));
    cudaMalloc((void**)&hsvArray.s, pixelCount * sizeof(double));
    cudaMalloc((void**)&hsvArray.v, pixelCount * sizeof(double));
    
    uchar3 *devInput; 
    cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
    cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);

    // Processing
    //----------------------
    // Start GPU processing (KERNEL)
    RGB2HSV<<<gridSize, blockSize>>>(devInput, hsvArray, inputImage->width, inputImage->height);
    // No need to clean devInput
    HSV2RGB<<<gridSize, blockSize>>>(hsvArray, devInput, inputImage->width, inputImage->height);
    
    // Get final image
    cudaMemcpy(outputImage, devInput, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);

    // Cleaning
    //----------------------
    cudaFree(devInput);
    cudaFree(hsvArray.h);
    cudaFree(hsvArray.s);
    cudaFree(hsvArray.v);

    //======================
    // !GRAYSCALING
    
}

typedef struct {
  unsigned int histogram[256];
} arrayOfHistograms;
__global__ void localGatherHisto(int *input, int imgWidth, arrayOfHistograms *arrayOfHisto) {
    // One thread(histo) / row
    unsigned int localHisto[256] = {0};
    
    //Calculate row histo
    for(int i = 0; i < imgWidth; i++)
        localHisto[ input[ blockIdx.y*imgWidth + i] ]++;
    
    //Store to SoA histo
    memcpy(arrayOfHisto[blockIdx.y].histogram, localHisto, sizeof(int)*256);
    
//    if( blockIdx.y == 0)        
//        printf("\t[localGatherHisto] FINISHED %dpx/%drow\n", imgWidth, gridDim.y);
    
}
__global__ void reduceHistogram(arrayOfHistograms *arrayOfHisto, int *globalHisto, int imgHeight){
    // [Optimization] Final histogram in shared memory    
    __shared__ unsigned int sharedHisto[256];
    // [Optimization] in local memory
    arrayOfHistograms *localArrayOfHisto = arrayOfHisto;
    
    // Init shared histo
    sharedHisto[threadIdx.x] = 0;
    
    __syncthreads();
    
    // Get the sum of particular histogram index
    //from all the previous blocks (during gather process)
    for (int row = 0; row < imgHeight; row++)            
        sharedHisto[threadIdx.x] += localArrayOfHisto[row].histogram[threadIdx.x];
        
    __syncthreads();
    
    // Copy output histogram
    if(threadIdx.x == 0)
        memcpy(globalHisto, sharedHisto, sizeof(int)*256);
        
    
//    if( blockIdx.y == 0)        
//        printf("\t[reduceHistogram] FINISHED\n");
}
__global__ void intensityProbability(float *proba, int *globalHisto, int totalPx){
    proba[threadIdx.x] = globalHisto[threadIdx.x]/totalPx;
}
void Labwork::labwork9_GPU() {

    /*  +---------+
        |   9.a   |
        +---------+ */

    // GRAYSCALING
    //======================
    
    // Preparing var
    //----------------------
    //Calculate number of pixels
    int pixelCount = inputImage->width * inputImage->height;
    uchar3 *devInput;
    int *devHisto;

    //Allocate CUDA memory    
    cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
    cudaMalloc(&devHisto, pixelCount * sizeof(int));
    // Copy CUDA Memory from CPU to GPU
    cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);
    
    //Create 32x32 Blocks
    dim3 blockSize = dim3(32, 32);
    dim3 gridSize = dim3((inputImage->width + (blockSize.x-1))/blockSize.x, 
        (inputImage->height  + (blockSize.y-1))/blockSize.y);

    // Processing
    //----------------------
    // Start GPU processing (KERNEL)
    grayscale2D<<<gridSize, blockSize>>>(devInput, devHisto, inputImage->width, inputImage->height);

    // Cleaning
    //----------------------
    //cudaFree(&devInput);

    //======================
    // !GRAYSCALING

    // HISTOGRAM
    //======================
    
    // Preparing var
    //----------------------
    arrayOfHistograms *arrayOfHisto;
    int *histoOfImage;

    // One histo[256] per row
    cudaMalloc(&arrayOfHisto, inputImage->height * sizeof(arrayOfHistograms));
    cudaMalloc(&histoOfImage, 256 * sizeof(int));
    
    // Processing
    //----------------------    
    localGatherHisto<<<dim3(1, inputImage->height, 1), dim3(1,1,1)>>>(devHisto, inputImage->width, arrayOfHisto);
    
    // SoA to AoS
    reduceHistogram<<<dim3(1,1,1), dim3(256,1,1)>>>(arrayOfHisto, histoOfImage, inputImage->height);

    // Cleaning
    //----------------------
    cudaFree(&arrayOfHisto);
    cudaFree(&devHisto);

    //======================
    // !HISTOGRAM
    
}

void Labwork::labwork10_GPU() {

}

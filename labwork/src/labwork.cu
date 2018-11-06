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

void Labwork::labwork6_GPU() {

}

__global__ void grayscale2D(uchar3 *input, uchar3 *output, int *histo, int imgWidth, int imgHeight) {
    //Calculate tid
    unsigned int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidx >= imgWidth || tidy >= imgHeight) return;

    int localtid =  tidx + (tidy * imgWidth);

    //Process pixel
    unsigned int g = ((int)input[localtid].x + (int)input[localtid].y + (int)input[localtid].z) / 3;
    output[localtid].z = output[localtid].y = output[localtid].x = (char)g;
    histo[localtid] = g;
}
__global__ void reduceMax(int *in, int *out) {
    // dynamic shared memory size, allocated in host
    extern __shared__ int cache[];
    // cache the block content
    unsigned int localtid = threadIdx.x;
    unsigned int tid = threadIdx.x+blockIdx.x*2*blockDim.x;
    cache[localtid] = max(in[tid], in[tid + blockDim.x]);

    __syncthreads();

    // reduction in cache
    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (localtid < s && cache[localtid + s] < 256) { // Debug pointer
            cache[localtid] = max(cache[localtid], cache[localtid + s]);
        }
        __syncthreads();
    }

    // only first thread writes back
    if (localtid == 0) {
        out[blockIdx.x] = cache[0];
    }
}
void Labwork::labwork7_GPU() {
    // GRAYSCALING
    //======================
    
    // Preparing var
    //----------------------
    //Calculate number of pixels
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    uchar3 *devInput;
    uchar3 *devGray;
    int *devHisto;

    //Allocate CUDA memory    
    cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
    cudaMalloc(&devGray, pixelCount * sizeof(uchar3));
    cudaMalloc(&devHisto, pixelCount * sizeof(int));
    // Copy CUDA Memory from CPU to GPU
    cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);

    // Processing
    //----------------------
    // Start GPU processing (KERNEL)
    //Create 32x32 Blocks
    dim3 blockSize = dim3(32, 32);
    dim3 gridSize = dim3((inputImage->width + (blockSize.x-1))/blockSize.x, 
        (inputImage->height  + (blockSize.y-1))/blockSize.y);
    grayscale2D<<<gridSize, blockSize>>>(devInput, devGray, devHisto, inputImage->width, inputImage->height);
    // Copy CUDA Memory from GPU to CPU
    cudaMemcpy(outputImage, devGray, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);

    // Cleaning
    //----------------------
    // Free CUDA Memory
    cudaFree(&devInput);
    cudaFree(&devGray);

    //======================
    // !GRAYSCALING

    // REDUCE
    //======================

    // Prep
    //----------------------
    int dimBlockR = 1024;
    int dimGridR = ceil(pixelCount / dimBlockR);

    int *devMax;



    int *temp = static_cast<int *>(malloc(pixelCount * sizeof(int)));    
    cudaMemcpy(temp, devHisto, pixelCount * sizeof(int), cudaMemcpyDeviceToHost);
    int test1 = -100;
    int test2 = 100;
    for (int i = 0; i < pixelCount; i++){
        //  printf("\t[%d / %d] %d - %d\n", i, dimBlockR, temp[i], &temp[i]);
        test1 = max(test1, temp[i]);
        test2 = min(test2, temp[i]);
        
        //if (outputImage[i]+128 > 160) printf("SAUCISSE %d\n", temp[i]);
    }
    printf("testMax %d\n", test1);
    printf("testMin %d\n", test2);

    //Allocate CUDA memory    
    cudaMalloc(&devMax, pixelCount * sizeof(int));
    // Processing
    //----------------------    
    // Get max value
    while(dimBlockR < (dimBlockR*dimGridR)/2){
        reduceMax<<<dimGridR, dimBlockR, dimBlockR * sizeof(int)>>>(devHisto, devMax);
        dimGridR /= 2;

        int *temp = static_cast<int *>(malloc(dimGridR * sizeof(int)));
        
        cudaMemcpy(temp, devMax, dimGridR * sizeof(int), cudaMemcpyDeviceToHost);
        
        cudaFree(&devHisto);
        cudaMalloc(&devHisto, dimGridR * sizeof(int));
        
        cudaFree(&devMax);
        cudaMalloc(&devMax, dimGridR * sizeof(int));
        
        cudaMemcpy(devHisto, temp, dimGridR * sizeof(int), cudaMemcpyHostToDevice);
    }
    
    // Copy final max reduce to host var
    int *hostMax = static_cast<int *>(malloc(dimGridR*sizeof(int))); 
    cudaMemcpy(hostMax, devHisto, dimGridR*sizeof(int), cudaMemcpyDeviceToHost);

    //printf("%d\n", &outputImage);
    //    printf("%d\n", *devMax.size());
    for (int i = 0; i < dimGridR; i++){
        printf("FINAL :\t[%d / %d] %d\n", i, dimGridR, hostMax[i]);
    }

    // Cleaning
    //----------------------
    // Free CUDA Memory
    cudaFree(&devMax);
    cudaFree(&devHisto);

    //======================
    // !REDUCE
    
}

void Labwork::labwork8_GPU() {

}

void Labwork::labwork9_GPU() {

}

void Labwork::labwork10_GPU() {

}
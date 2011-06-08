#include <string.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <float.h>

#define BLOCK_SIZE 256

__global__ void euclidian_distance_transform_kernel(
  const unsigned char* img, float* dist, int w, int h)
{
  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  const int N = w*h;
  
  if (i >= N)
  {   
    return; 
  }
  
  int cx = i % w;
  int cy = i / w;
  
  float minv = INFINITY;
  
  if (img[i] > 0)
  {
    minv = 0.0f;
  }
  else
  {
    for (int j = 0; j < N; j++)  
    {
        if (img[j] > 0)
        {
          int x = j % w;
          int y = j / w;
          float d = sqrtf( powf(float(x-cx), 2.0f) + powf(float(y-cy), 2.0f) );
          if (d < minv) minv = d;
        }
    }
  }

  dist[i] = minv;
}

void euclidian_distance_transform(unsigned char* img, float* dist, int w, int h) {

    cudaError_t err;
    unsigned char *d_img;
    cudaMalloc((void**) &d_img, w*h*sizeof(unsigned char));    
    cudaMemcpy(d_img, img, w*h*sizeof(unsigned char), cudaMemcpyHostToDevice);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA ERROR: %s\n", cudaGetErrorString(err));
    }
    
    float* d_dist;
    cudaMalloc((void**) &d_dist, w*h*sizeof(float));
    //cudaMemset(d_dist, 0, w*h*sizeof(float));
    
    dim3 block (BLOCK_SIZE,1,1);
    
    int gx = (w*h+BLOCK_SIZE-1)/BLOCK_SIZE;    
    dim3 grid(gx,1);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA ERROR: %s\n", cudaGetErrorString(err));
    }

    euclidian_distance_transform_kernel <<<grid, block>>> (d_img, d_dist, w, h);
    cudaThreadSynchronize();
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA ERROR: %s\n", cudaGetErrorString(err));
    }
    
    cudaMemcpy(dist, d_dist, w*h*sizeof(float), cudaMemcpyDeviceToHost);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA ERROR: %s\n", cudaGetErrorString(err));
    }
    
    cudaFree(d_img);
    cudaFree(d_dist);
}



int main()
{
    char line[256];
    int w,h;
    int i;
    int v;
    
    FILE* f = fopen("img.pgm", "r");
    fgets(line, sizeof(line), f);
    fgets(line, sizeof(line), f);
    fgets(line, sizeof(line), f);
    sscanf(line, "%d %d", &w, &h);
    fgets(line, sizeof(line), f);
    
    printf("%d %d\n", w, h);
    
    unsigned char* img = (unsigned char*)malloc(sizeof(unsigned char)*w*h);
    float* dist = (float*)malloc(sizeof(float)*w*h);
    
    for (i=0; i<w*h; i++)
    {
        fgets(line, sizeof(line), f);
        sscanf(line, "%d", &v);
        img[i] = (v > 0)? 255 : 0;
        //if (img[i]==255) printf("wp: %d %d\n", i%w, i/w);
    }
    
    fclose(f);
    
    printf("start\n");
    cudaEvent_t start,stop;
    float time=0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);      
    cudaEventRecord(start,0);
    
    euclidian_distance_transform(img, dist, w, h);
    
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time,start,stop);
    
    printf("end\n");
    printf("time: %f\n", time);
    
    FILE* f2 = fopen("output.pgm", "w");
    fprintf(f2, "P2\n");
    fprintf(f2, "#\n");
    fprintf(f2, "%d %d\n", w, h);
    fprintf(f2, "255\n");
    
    float max = 0.0f;
    for (i=0; i<w*h; i++)
    {
       max = (dist[i] > max)? dist[i] : max;
    }
    printf("max: %f\n", max);
    
    for (i=0; i<w*h; i++)
    {
        fprintf(f2, "%d\n", ((int)floor((255.0f*dist[i])/max)));
    }
    
    fclose(f2);
       
    free(img);
    free(dist);
    
    return 0;
}

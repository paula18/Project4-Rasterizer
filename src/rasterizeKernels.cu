// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <thrust/random.h>
#include <thrust/device_ptr.h>
#include <thrust/remove.h>
#include "rasterizeKernels.h"
#include "rasterizeTools.h"




glm::vec3* framebuffer;
fragment* depthbuffer;
float* device_vbo;
float* device_cbo;
int* device_ibo;
float* device_nbo;
float* device_vboWindow; 
triangle* primitives;

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 

//Handy dandy little hashing function that provides seeds for random number generation
__host__ __device__ unsigned int hash(unsigned int a){
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

//Writes a given fragment to a fragment buffer at a given location
__host__ __device__ void writeToDepthbuffer(int x, int y, fragment frag, fragment* depthbuffer, glm::vec2 resolution){
  if(x<resolution.x && y<resolution.y){
    int index = (y*resolution.x) + x;
    depthbuffer[index] = frag;
  }
}

//Reads a fragment from a given location in a fragment buffer
__host__ __device__ fragment getFromDepthbuffer(int x, int y, fragment* depthbuffer, glm::vec2 resolution){
  if(x<resolution.x && y<resolution.y){
    int index = (y*resolution.x) + x;
    return depthbuffer[index];
  }else{
    fragment f;
    return f;
  }
}

//Writes a given pixel to a pixel buffer at a given location
__host__ __device__ void writeToFramebuffer(int x, int y, glm::vec3 value, glm::vec3* framebuffer, glm::vec2 resolution){
  if(x<resolution.x && y<resolution.y){
    int index = (y*resolution.x) + x;
    framebuffer[index] = value;
  }
}

//Reads a pixel from a pixel buffer at a given location
__host__ __device__ glm::vec3 getFromFramebuffer(int x, int y, glm::vec3* framebuffer, glm::vec2 resolution){
  if(x<resolution.x && y<resolution.y){
    int index = (y*resolution.x) + x;
    return framebuffer[index];
  }else{
    return glm::vec3(0,0,0);
  }
}

//Kernel that clears a given pixel buffer with a given color
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image, glm::vec3 color){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      image[index] = color;
    }
}

//Kernel that clears a given fragment buffer with a given fragment
__global__ void clearDepthBuffer(glm::vec2 resolution, fragment* buffer, fragment frag){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      fragment f = frag;
      f.position.x = x;
      f.position.y = y;
      buffer[index] = f;
    }
}

//Kernel that writes the image to the OpenGL PBO directly. 
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;      
      color.x = image[index].x*255.0;
      color.y = image[index].y*255.0;
      color.z = image[index].z*255.0;

      if(color.x>255){
        color.x = 255;
      }

      if(color.y>255){
        color.y = 255;
      }

      if(color.z>255){
        color.z = 255;
      }
      
      // Each thread writes one pixel location in the texture (textel)
      PBOpos[index].w = 0;
      PBOpos[index].x = color.x;     
      PBOpos[index].y = color.y;
      PBOpos[index].z = color.z;
  }
}

__global__ void vertexShadeKernel(glm::vec2 resolution, float* vbo, int vbosize, float* vboWindow, glm::mat4 mModel, glm::mat4 mView, glm::mat4 mProj)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index<vbosize/3)
	{
		//Calculate transformation matrices
		glm::mat4 m = mProj * mView * mModel;
		

		//Calculate position after transfrmation
		glm::vec4 pModel = glm::vec4(vbo[3 * index], vbo[3 * index + 1], vbo[3 * index + 2], 1);
		glm::vec4 pClip = m * pModel; 

		//Calculate point in NDC coord
		glm::vec3 pNDC = glm::vec3( pClip.x / pClip.w, pClip.y / pClip.w, pClip.z / pClip.w);  

		//Calculate point in window coord
		float xWindow = ( pNDC.x + 1) * resolution.x * 0.5;
		float yWindow = ( pNDC.y + 1 ) * resolution.y * 0.5;

		glm::vec3 pWindow = glm::vec3( xWindow, yWindow, pNDC.z);

		//put result in vbo array
		vboWindow[3 * index] = pWindow.x; 
		vboWindow[3 * index + 1] = pWindow.y; 
		vboWindow[3 * index + 2] = pWindow.z;
	}
}

__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float* nbo, int nbosize, triangle* primitives, 
	float *vboWindow, int colorMode, glm::vec3 cameraPosition, glm::mat4 mView)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int primitivesCount = ibosize/3;
	if(index<primitivesCount)
	{
		int id0 = ibo[3 * index];
		int id1 = ibo[3 * index + 1]; 
		int id2 = ibo[3 * index + 2];

		
		//Vertices in window
		primitives[index].p0 = glm::vec3(vboWindow[3 * id0], vboWindow[3 * id0 + 1], vboWindow[3 * id0 + 2]);
		primitives[index].p1 = glm::vec3(vboWindow[3 * id1], vboWindow[3 * id1 + 1], vboWindow[3 * id1 + 2]);
		primitives[index].p2 = glm::vec3(vboWindow[3 * id2], vboWindow[3 * id2 + 1], vboWindow[3 * id2 + 2]);


		//Vertices in world
		primitives[index].pworld0 = glm::vec3(vbo[3 * id0], vbo[3 * id0 + 1], vbo[3 * id0 + 2]);
		primitives[index].pworld1 = glm::vec3(vbo[3 * id1], vbo[3 * id1 + 1], vbo[3 * id1 + 2]);
		primitives[index].pworld2 = glm::vec3(vbo[3 * id2], vbo[3 * id2 + 1], vbo[3 * id2 + 2]);

#if(BACK_FACE_CULLING)
		//Back face culling 
		glm::vec4 pv0 = mView * glm::vec4(primitives[index].pworld0, 1.0f);
		glm::vec4 pv1 = mView * glm::vec4(primitives[index].pworld1, 1.0f);
		glm::vec4 pv2 = mView * glm::vec4(primitives[index].pworld2, 1.0f);

		glm::vec4 p10 = pv1 - pv0;
		glm::vec4 p20 = pv2 - pv0;

		glm::vec3 backFace = glm::cross(glm::vec3(p10.x, p10.y, p10.z), glm::vec3(p20.x, p20.y, p20.z ));
		glm::vec3 view = glm::vec3(glm::vec3(pv0) - cameraPosition);
		
		if(backFace.z < 1e-6)
			primitives[index].draw = false; 
		else
			primitives[index].draw = true; 

#else
		primitives[index].draw = true; 

#endif
		
		switch (colorMode)
		{
		case 3:
			//Color
			primitives[index].c0 = glm::vec3(1.0f, 0.0f, 0.0f); 
			primitives[index].c1 = glm::vec3(0.0f, 1.0f, 0.0f);
			primitives[index].c2 = glm::vec3(0.0f, 0.0f, 1.0f);
			break;
	/*	case 4:
			if (primitives[index].draw == true)
			{
				primitives[index].c0 = glm::vec3(1.0f, 0.0, 0.0); 
				primitives[index].c1 = glm::vec3(1.0f, 0.0, 0.0);
				primitives[index].c2 = glm::vec3(1.0f, 0.0, 0.0);
			}
			else
			{
				primitives[index].c0 = glm::vec3(1.0f, 0.0, 1.0f);
				primitives[index].c1 = glm::vec3(1.0f, 0.0, 1.0f);
				primitives[index].c2 = glm::vec3(1.0f, 0.0, 1.0f);
				primitives[index].draw = true;
			}
			break;*/
		default:
			primitives[index].c0 = glm::vec3(cbo[3 * id0], cbo[3 * id0 + 1], cbo[3 * id0 + 2]); 
			primitives[index].c1 = glm::vec3(cbo[3 * id1], cbo[3 * id1 + 1], cbo[3 * id1 + 2]);
			primitives[index].c2 = glm::vec3(cbo[3 * id2], cbo[3 * id2 + 1], cbo[3 * id2 + 2]);

		}
		//Normals
		primitives[index].n0 = glm::vec3(nbo[3 * id0], nbo[3 * id0 + 1], nbo[3 * id0 + 2]);
		primitives[index].n1 = glm::vec3(nbo[3 * id1], nbo[3 * id1 + 1], nbo[3 * id1 + 2]);
		primitives[index].n2 = glm::vec3(nbo[3 * id2], nbo[3 * id2 + 1], nbo[3 * id2 + 2]);

		
	}
}

//TODO: Implement a rasterization method, such as scanline.
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index < primitivesCount)
	{
		triangle t  = primitives[index];
		if(t.draw == false) 
			return;
		
		glm::vec3 minpoint, maxpoint;
		
		getAABBForTriangle(t, minpoint, maxpoint);
		
		minpoint.x = minpoint.x > 0 ? minpoint.x : 0;
		minpoint.y = minpoint.y > 0 ? minpoint.y : 0;
		maxpoint.x = maxpoint.x < resolution.x ? maxpoint.x : resolution.x;
		maxpoint.y = maxpoint.y < resolution.y ? maxpoint.y : resolution.y;
		

		for(int i = minpoint.x; i < maxpoint.x; i++)
		{
			for(int j = minpoint.y; j < maxpoint.y; j++)
			{

				glm::vec3 baryCoord = calculateBarycentricCoordinate(t, glm::vec2(i, j));
				
				if (isBarycentricCoordInBounds(baryCoord))
				{				
					float z = getZAtCoordinate(baryCoord, t);		
					int pixelID = j * resolution.x + i;

					if(z > depthbuffer[pixelID].zdepth)
					{
						depthbuffer[pixelID].color = t.c0 * baryCoord.x + t.c1 * baryCoord.y + t.c2 * baryCoord.z;
						depthbuffer[pixelID].normal = glm::normalize(baryCoord.x * t.n0 + baryCoord.y * t.n1 + baryCoord.z * t.n2);
						depthbuffer[pixelID].position = t.pworld0 * baryCoord.x + t.pworld1 * baryCoord.y + t.pworld2 * baryCoord.z;
						depthbuffer[pixelID].zdepth = z; 
					}
				}
			}
		}
	}
}

//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution, int colorMode, glm::vec3 cameraPosition)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);
	if(x<=resolution.x && y<=resolution.y)
	{
		fragment f = depthbuffer[index];

		//glm::vec3 lightColor = glm::vec3(0.4f, 0.6f, 0.8f); 
		//BUNNY 
		//glm::vec3 lightPos = glm::vec3 (4.0f, 12.0f, 3.0f); 
		///glm::vec3 lightColor = glm::vec3(0.5, 0.7, 0.8); 
		//glm::vec3 lightColor = glm::vec3(0.5*glm::abs(f.normal.x), 0.7*glm::abs(f.normal.y), 0.2*glm::abs(f.normal.z)); 
		//BUDDA
		//glm::vec3 lightPos = glm::vec3 (0.0f, 0.0f, 10.0f);
		//glm::vec3 lightColor = glm::vec3(0.9, 0.5, 0.1); 

		//DRAGON 
		glm::vec3 lightPos = glm::vec3 (0.0f, 0.0f, 10.0f);
		glm::vec3 lightColor = glm::vec3 (f.normal);


		glm::vec3 lightDir = glm::normalize(lightPos - f.position);
		glm::vec3 ambient = glm::vec3(1.0f) * f.color;
		switch (colorMode)
		{
			case 0: 
				float diffuse = glm::dot(lightDir, f.normal); 
			
				//f.color = f.color;
				f.color = 0.1f * ambient + diffuse * lightColor;
				
				break;
			case 1:
				if (colorMode == NORMAL)
				f.color = glm::abs(f.normal);
				
				break;

			case 2:
				float kspec = 3.0f; 
				glm::vec3 r = glm::normalize(glm::reflect(-lightDir, f.normal)); 
				glm::vec3 v = glm::normalize(cameraPosition - f.position); 
			
				float dot = max(glm::dot(r, v), 0.0f); 
				if (dot > 1.0f) dot = 1.0f; 
			
				float klight = glm::dot(lightDir, f.normal);
				float specular = pow(dot, kspec); 
				
				float c0 = min(0.1f * ambient.x + klight * lightColor.x + specular, 1.0f); 
				float c1 = min(0.1f * ambient.y + klight * lightColor.y + specular, 1.0f); 
				float c2 = min(0.1f * ambient.z + klight * lightColor.z + specular, 1.0f); 

				f.color = glm::vec3(c0, c1, c2); 
				break;
				
		}
		depthbuffer[index] = f;
	}
}

//Writes fragment colors to the framebuffer
__global__ void render(glm::vec2 resolution, fragment* depthbuffer, glm::vec3* framebuffer, int aliasing)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);

	if(x<=resolution.x && y<=resolution.y)
	{
		if (aliasing == ON)
		{
			float a = 1;
			glm::vec3 value = glm::vec3(0.0f);
			int num = 0; 
			for(int i = -a; i <= a; ++i)
			{
				for(int j = -a; j <= a; ++j)
				{
					int xvalue = x + i; 
					int yvalue = y + j;
					if( xvalue < 0 || xvalue >= resolution.x || yvalue < 0 || yvalue >= resolution.y)
						continue;
					else
					{
						int id = xvalue + resolution.x * yvalue; 
						value += depthbuffer[id].color; 
						num++;
					}
				}
			}
			framebuffer[index] = value * (1/(float)num);
		}
		
		else		
			framebuffer[index] = depthbuffer[index].color;

	}
}

struct notVisible
{
	__device__ bool operator()(const triangle t) 
	{
		return !t.draw;
	}
};

// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, 
	float* nbo, int nbosize, glm::mat4 mModel, glm::mat4 mView, glm::mat4 mProj, glm::vec3 cameraPosition, int color, int aliasing){

  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(resolution.x)/float(tileSize)), (int)ceil(float(resolution.y)/float(tileSize)));

  //set up framebuffer
  framebuffer = NULL;
  cudaMalloc((void**)&framebuffer, (int)resolution.x*(int)resolution.y*sizeof(glm::vec3));
  
  //set up depthbuffer
  depthbuffer = NULL;
  cudaMalloc((void**)&depthbuffer, (int)resolution.x*(int)resolution.y*sizeof(fragment));

  //kernel launches to black out accumulated/unaccumlated pixel buffers and clear our scattering states
  clearImage<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, framebuffer, glm::vec3(0,0,0));
  
  fragment frag;
  frag.color = glm::vec3(0,0,0);
  frag.normal = glm::vec3(0,0,0);
  frag.position = glm::vec3(0,0,-10000);
  frag.zdepth = -FLT_MAX; 
  clearDepthBuffer<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer,frag);

  //------------------------------
  //memory stuff
  //------------------------------
  primitives = NULL;
  cudaMalloc((void**)&primitives, (ibosize/3)*sizeof(triangle));

  device_ibo = NULL;
  cudaMalloc((void**)&device_ibo, ibosize*sizeof(int));
  cudaMemcpy( device_ibo, ibo, ibosize*sizeof(int), cudaMemcpyHostToDevice);

  device_vbo = NULL;
  cudaMalloc((void**)&device_vbo, vbosize*sizeof(float));
  cudaMemcpy( device_vbo, vbo, vbosize*sizeof(float), cudaMemcpyHostToDevice);

  device_cbo = NULL;
  cudaMalloc((void**)&device_cbo, cbosize*sizeof(float));
  cudaMemcpy( device_cbo, cbo, cbosize*sizeof(float), cudaMemcpyHostToDevice);

  device_nbo = NULL; 
  cudaMalloc((void**)&device_nbo, nbosize*sizeof(float)); 
  cudaMemcpy( device_nbo, nbo, nbosize*sizeof(float), cudaMemcpyHostToDevice);

  device_vboWindow = NULL; 
  cudaMalloc((void**)&device_vboWindow, vbosize*sizeof(float));

  tileSize = 32;
  int primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));

  //------------------------------
  //vertex shader
  //------------------------------

	vertexShadeKernel<<<primitiveBlocks, tileSize>>>(resolution, device_vbo, vbosize, device_vboWindow, mModel, mView, mProj);

	cudaDeviceSynchronize();
  //------------------------------
  //primitive assembly
  //------------------------------
	primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
	primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_cbo, cbosize, device_ibo, ibosize, device_nbo, nbosize, primitives, device_vboWindow, color, cameraPosition, mView);

	cudaDeviceSynchronize();
  //------------------------------
  //rasterization
  //------------------------------
	

#if(BACK_FACE_CULLING)
	thrust::device_ptr<triangle> primitivesArray = thrust::device_pointer_cast(primitives);
	int numberOfPrimitives = ceil((float)ibosize/3);
	int newNumberPrimitives = thrust::remove_if(primitivesArray, primitivesArray + numberOfPrimitives, notVisible()) - primitivesArray;
	int rasterBlocks = ceil(((float)newNumberPrimitives)/((float)tileSize));

	rasterizationKernel<<<rasterBlocks, tileSize>>>(primitives, newNumberPrimitives, depthbuffer, resolution);
#else
	rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution);

#endif

	
  cudaDeviceSynchronize();
  //------------------------------
  //fragment shader
  //------------------------------

	fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution, color, cameraPosition);

	cudaDeviceSynchronize();
  //------------------------------
  //write fragments to framebuffer
  //------------------------------
	render<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer, framebuffer, aliasing);
	sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, resolution, framebuffer);

	cudaDeviceSynchronize();

	kernelCleanup();

	checkCUDAError("Kernel failed!");
}

void kernelCleanup()
{
	cudaFree( primitives );
	cudaFree( device_vbo );
	cudaFree( device_cbo );
	cudaFree( device_ibo );
	cudaFree( device_nbo ); 
	cudaFree( device_vboWindow);
	cudaFree( framebuffer );
	cudaFree( depthbuffer );
}


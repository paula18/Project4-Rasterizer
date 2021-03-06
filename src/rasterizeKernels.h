// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#ifndef RASTERIZEKERNEL_H
#define RASTERIZEKERNEL_H

#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>
#include "glm/glm.hpp"

enum colorMode{DIFFUSE, NORMAL, SPEC, COLOR_INTER, BACK_FACE_COLOR};
enum aliasing {ON, OFF};


#define BACK_FACE_CULLING 1

void kernelCleanup();
void cudaRasterizeCore(uchar4* pos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, 
					float* nbo, int nbosize, glm::mat4 mModel, glm::mat4 mView, glm::mat4 mProj, glm::vec3 cameraPosition, int color, int aliasing);

#endif //RASTERIZEKERNEL_H

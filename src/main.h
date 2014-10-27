// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#ifndef MAIN_H
#define MAIN_H

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <fstream>
#include <glm/glm.hpp>
#include <glslUtil/glslUtility.hpp>
#include <iostream>
#include <objUtil/objloader.h>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <time.h>

#include <glm/gtc/matrix_transform.hpp>
#include "glm/gtx/rotate_vector.hpp"


#include "rasterizeKernels.h"
#include "utilities.h"

using namespace std;

//-------------------------------
//------------GL STUFF-----------
//-------------------------------
int frame;
int fpstracker;
double seconds;
int fps = 0;
int totalTime = 0; 
int iterations = 0;
GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
const char *attributeLocations[] = { "Position", "Tex" };
GLuint pbo = (GLuint)NULL;
GLuint displayImage;
uchar4 *dptr;

GLFWwindow *window;

obj* mesh;

float* vbo;
int vbosize;
float* cbo;
int cbosize;
int* ibo;
int ibosize;
float* nbo; 
int nbosize; 

//-------------------------------
//----------CAMERA STUFF-----------
//-------------------------------
int lastX, lastY;

float eyeDistance = 10.0f;
float head = 45.0f, pitch = 45.0f;
glm::vec3 cameraPosition(0, 0, eyeDistance); 

float upVectorY = glm::cos(glm::radians(head)) > 0.0f ? -1.0f : 1.0f; 
glm::vec3 upVector = glm::vec3(0.0f, upVectorY, 0.0f); 

glm::vec3 cameraTarget = glm::vec3(0.0f, 0.0f, 0.0f);

float speed = 0.05f; 
float sensitivity = 0.1f; 

float fov = 30;     // The horizontal Field of View, in degrees : the amount of "zoom". 
						//Think "camera lens". Usually between 90° (extra wide) and 30° (quite zoomed in)

colorMode color = DIFFUSE; 
aliasing aliasing = OFF; 

//-------------------------------
//----------CUDA STUFF-----------
//-------------------------------

int width = 800; int height = 800;

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv);

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

void runCuda();

#ifdef __APPLE__
	void display();
#else
	void display();
	void keyboard(unsigned char key, int x, int y);
#endif

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------
bool init(int argc, char* argv[]);
void initPBO();
void initCuda();
void initTextures();
void initVAO();
GLuint initShader();


//-------------------------------
//---------CLEANUP STUFF---------
//-------------------------------

void cleanupCuda();
void deletePBO(GLuint* pbo);
void deleteTexture(GLuint* tex);

//------------------------------
//-------GLFW CALLBACKS---------
//------------------------------
void mainLoop();
void errorCallback(int error, const char *description);
void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
void mouseCallback(GLFWwindow* window, int key, int action, int mods);
void mouseScroll(GLFWwindow* window,double x,double y);

void turnCamera(float& m_pitch);

#endif
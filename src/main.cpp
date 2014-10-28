// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include "main.h"

#define TURN_TABLE 0
//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv){

  bool loadedScene = false;
  for(int i=1; i<argc; i++){
    string header; string data;
    istringstream liness(argv[i]);
    getline(liness, header, '='); getline(liness, data, '=');
    if(strcmp(header.c_str(), "mesh")==0){
      //renderScene = new scene(data);
      mesh = new obj();
      objLoader* loader = new objLoader(data, mesh);
      mesh->buildVBOs();
      delete loader;
      loadedScene = true;
    }
  }

  if(!loadedScene){
    cout << "Usage: mesh=[obj file]" << endl;
    return 0;
  }

  frame = 0;
  seconds = time (NULL);
  fpstracker = 0;

  // Launch CUDA/GL
  if (init(argc, argv)) {
    // GLFW main loop
    mainLoop();
  }

  return 0;
}

void mainLoop() {
  while(!glfwWindowShouldClose(window)){
    glfwPollEvents();
    runCuda();

    time_t seconds2 = time (NULL);

    if(seconds2-seconds >= 1){

        fps = fpstracker/(seconds2-seconds);
        fpstracker = 0;
        seconds = seconds2;
    }

    string title = "CIS565 Rasterizer | " + utilityCore::convertIntToString((int)fps) + " FPS";
		glfwSetWindowTitle(window, title.c_str());
    
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glClear(GL_COLOR_BUFFER_BIT);   

    // VAO, shader program, and texture already bound
    glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);
    glfwSwapBuffers(window);
  }
  glfwDestroyWindow(window);
  glfwTerminate();
}

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

void runCuda()
{
  // Map OpenGL buffer object for writing from CUDA on a single GPU
  // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer
	
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float time; 
	cudaEventRecord(start, 0);
	
	
	dptr=NULL;

	vbo = mesh->getVBO();
	vbosize = mesh->getVBOsize();

	float newcbo[] = {0.0, 1.0, 0.0, 
                    0.0, 0.0, 1.0, 
                    1.0, 0.0, 0.0};
	cbo = mesh->getCBO();
	cbosize = mesh->getCBOsize(); 

	ibo = mesh->getIBO();
	ibosize = mesh->getIBOsize();

	nbo = mesh->getNBO();
	nbosize = mesh->getNBOsize();

	glm::mat4 mModel(1.0f); 


#if (TURN_TABLE)
	//head = frame % 360;
	pitch = frame % 360;
#endif
	float r_head = glm::radians(head), r_pitch = glm::radians(pitch);
	cameraPosition.x = cameraTarget.x + eyeDistance * glm::cos(r_head) * glm::cos(r_pitch);
	cameraPosition.y = cameraTarget.y + eyeDistance * glm::sin(r_head);
	cameraPosition.z = cameraTarget.z + eyeDistance * glm::cos(r_head) * glm::sin(r_pitch);


	//Change up vectr depending on model 
	float upVectorY = glm::cos(r_head) > 0.0f ? -1.0f : -1.0f; 
	upVector = glm::vec3(0.0f, upVectorY, 0.0f); 

	
	glm::mat4 mView = glm::lookAt(cameraPosition, cameraTarget, upVector); 

		
	float aspectRatio = width / height;					// Aspect Ratio. Depends on the size of your window. 
													//Notice that 4/3 == 800/600 == 1280/960, sounds familiar ?
	float nearClippingPlane =  0.1f;        // Near clipping plane. Keep as big as possible, or you'll get precision issues.
	float farClippingPlane = 100.0f;       // Far clipping plane. Keep as little as possible.

	glm::mat4 mProj = glm::perspective(fov, aspectRatio, nearClippingPlane, farClippingPlane); 

	cudaGLMapBufferObject((void**)&dptr, pbo);
	cudaRasterizeCore(dptr, glm::vec2(width, height), frame, vbo, vbosize, cbo, cbosize, ibo, ibosize, nbo, nbosize,
									mModel, mView, mProj, cameraPosition, color, aliasing);
	cudaGLUnmapBufferObject(pbo);

	vbo = NULL;
	cbo = NULL;
	ibo = NULL;
	nbo = NULL; 

	frame++;
	fpstracker++;

	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop); 

	cudaEventElapsedTime(&time, start, stop); 

	totalTime += time;

	std::cout << "Iterations: " << iterations << " Time: " << time << std::endl;

	if (iterations == 100)
	{
		std::cout << "Iterations: " << iterations << " Time: " << time << " Average Time: " << totalTime / iterations << std::endl;
	}
	iterations++; 

}
  
//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

bool init(int argc, char* argv[]) {
  glfwSetErrorCallback(errorCallback);

  if (!glfwInit()) {
      return false;
  }

  width = 800;
  height = 800;
  window = glfwCreateWindow(width, height, "CIS 565 Pathtracer", NULL, NULL);
  if (!window){
      glfwTerminate();
      return false;
  }
  glfwMakeContextCurrent(window);
  glfwSetKeyCallback(window, keyCallback);
glfwSetMouseButtonCallback(window, mouseCallback);
  glfwSetScrollCallback(window, mouseScroll);
 
  // Set up GL context
  glewExperimental = GL_TRUE;
  if(glewInit()!=GLEW_OK){
    return false;
  }

  // Initialize other stuff
  initVAO();
  initTextures();
  initCuda();
  initPBO();
  
  GLuint passthroughProgram;
  passthroughProgram = initShader();

  glUseProgram(passthroughProgram);
  glActiveTexture(GL_TEXTURE0);

  return true;
}

void initPBO(){
  // set up vertex data parameter
  int num_texels = width*height;
  int num_values = num_texels * 4;
  int size_tex_data = sizeof(GLubyte) * num_values;
    
  // Generate a buffer ID called a PBO (Pixel Buffer Object)
  glGenBuffers(1, &pbo);

  // Make this the current UNPACK buffer (OpenGL is state-based)
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

  // Allocate data for the buffer. 4-channel 8-bit image
  glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
  cudaGLRegisterBufferObject(pbo);

}

void initCuda(){
  // Use device with highest Gflops/s
  cudaGLSetGLDevice(0);

  // Clean up on program exit
  atexit(cleanupCuda);
}

void initTextures(){
    glGenTextures(1, &displayImage);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA,
        GL_UNSIGNED_BYTE, NULL);
}

void initVAO(void){
    GLfloat vertices[] =
    { 
        -1.0f, -1.0f, 
         1.0f, -1.0f, 
         1.0f,  1.0f, 
        -1.0f,  1.0f, 
    };

    GLfloat texcoords[] = 
    { 
        1.0f, 1.0f,
        0.0f, 1.0f,
        0.0f, 0.0f,
        1.0f, 0.0f
    };

    GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

    GLuint vertexBufferObjID[3];
    glGenBuffers(3, vertexBufferObjID);
    
    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0); 
    glEnableVertexAttribArray(positionLocation);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(texcoordsLocation);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}


GLuint initShader() {
  const char *attribLocations[] = { "Position", "Tex" };
  GLuint program = glslUtility::createDefaultProgram(attribLocations, 2);
  GLint location;
  
  glUseProgram(program);
  if ((location = glGetUniformLocation(program, "u_image")) != -1)
  {
    glUniform1i(location, 0);
  }

  return program;
}

//-------------------------------
//---------CLEANUP STUFF---------
//-------------------------------

void cleanupCuda(){
  if(pbo) deletePBO(&pbo);
  if(displayImage) deleteTexture(&displayImage);
}

void deletePBO(GLuint* pbo){
  if (pbo) {
    // unregister this buffer object with CUDA
    cudaGLUnregisterBufferObject(*pbo);
    
    glBindBuffer(GL_ARRAY_BUFFER, *pbo);
    glDeleteBuffers(1, pbo);
    
    *pbo = (GLuint)NULL;
  }
}

void deleteTexture(GLuint* tex){
    glDeleteTextures(1, tex);
    *tex = (GLuint)NULL;
}
 
void shut_down(int return_code){
  kernelCleanup();
  cudaDeviceReset();
  #ifdef __APPLE__
  glfwTerminate();
  #endif
  exit(return_code);
}

//------------------------------
//-------GLFW CALLBACKS---------
//------------------------------

void errorCallback(int error, const char* description){
    fputs(description, stderr);
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods){
    if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS){
        glfwSetWindowShouldClose(window, GL_TRUE);
    }

	else if(key == GLFW_KEY_D && action == GLFW_PRESS)
		color = DIFFUSE;
	
	else if(key == GLFW_KEY_N && action == GLFW_PRESS)
		color = NORMAL;
	
	else if(key == GLFW_KEY_S && action == GLFW_PRESS)
		color = SPEC;
	
	else if(key == GLFW_KEY_C && action == GLFW_PRESS)
		color = COLOR_INTER;
	
	else if(key == GLFW_KEY_B && action == GLFW_PRESS)
		color = BACK_FACE_COLOR;

	else if(key == GLFW_KEY_A && action == GLFW_PRESS)
	{
		if (aliasing == ON) aliasing = OFF; 
		else aliasing = ON; 
	}
	else if(key == GLFW_KEY_R && action == GLFW_PRESS)
	{
		iterations = 0; 
		totalTime = 0; 
	}

	
}

void mouseCallback(GLFWwindow* window, int key, int action, int mods)
{

	double mouseXPos, mouseYPos;    //Position in screen coordinates
	glfwGetCursorPos (window, &mouseXPos, &mouseYPos);

	float deltaX = float(mouseXPos - lastX); 
	float deltaY = float(mouseYPos - lastY); 

	float xWindow = ( deltaX + 1) * width * 0.5;
	float yWindow = ( deltaY + 1 ) * height * 0.5;

	if (key == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS)
	{
	
		head += deltaY * sensitivity;  
		pitch += deltaX * sensitivity;  

		if (head > 89.0f)
			head = 89.0f; 
		if (head < -89.0f) 
			head = -89.0f; 
	
	}

	else if (key == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
	{
		float upVectorY = glm::cos(glm::radians(head)) > 0.0f ? -1.0f : 1.0f; 
		upVector = glm::vec3(0.0f, upVectorY, 0.0f); 
		
		glm::vec3 vdir(glm::normalize(cameraTarget -  cameraPosition));
		glm::vec3 u(glm::normalize(glm::cross(vdir, upVector)));
		glm::vec3 v(glm::normalize(glm::cross(u, vdir)));

		glm::vec3 a =  0.01f * (deltaY * v - deltaX * u);
		//cameraTarget += a;
		//cameraPosition += a;
	}
	

	
	lastX = mouseXPos; 
	lastY = mouseYPos; 


}

//Zoom in and Zoom out
void mouseScroll(GLFWwindow* window,double x,double y)
{
	
	eyeDistance += y * speed; 


}

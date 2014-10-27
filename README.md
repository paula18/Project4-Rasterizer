-------------------------------------------------------------------------------
CIS565: Project 4: CUDA Rasterizer
-------------------------------------------------------------------------------
Fall 2014
-------------------------------------------------------------------------------

![alt tag](https://github.com/paula18/Project4-Rasterizer/blob/master/dragon1.PNG)

![alt tag](https://github.com/paula18/Project4-Rasterizer/blob/master/buda1.PNG)


-------------------------------------------------------------------------------
ALGORITHM OVERVIEW:
-------------------------------------------------------------------------------

In this project I implemented a CUDA rasterizer. This implementation is based on the standard grapics pipeline.

![alt tag](https://github.com/paula18/Project4-Rasterizer/blob/master/graphicsPipeline.PNG)


The specific steps that I have implemented are: 

* Vertex shading
* Primitive assembly
* Rasterization 
* Fragment shading
* Rendering with anti-aliasing

-------------------------------------------------------------------------------
Vertex Shading:
-------------------------------------------------------------------------------
In the vertex shader, the vertices are transformed from the object space to window coordinates. Each vertex in the VBO array is multiplied by the model, view and projection matrices to transform it to NDC coordinates. Then, each vertex in the NDC space is divided by the w coordinate to account for the perspective transformation and finally transformed to window coordinates defined by the resolution of the screen. 
After these steps, the transformed vertices are stores in the vboWindow array, while the object space vertices are keep in the VBO array. 
The steps to perform the transformation are the following: 


![alt tag](https://github.com/paula18/Project4-Rasterizer/blob/master/vertexShader.PNG)

-------------------------------------------------------------------------------
Primitive Assembly:
-------------------------------------------------------------------------------
The primitive assembly kernel groups the vertices transformed in the previous step to form the triangles. Here, the vertices, normals and colors are obtained from the VBO, NBO and CBO arrays respectively and are assigned to the primitives according to their indeces. 
A single triangle stores its three vertices in world space, its three vertices in screen space, three normals corresponding to each vertex and three colors, each for each vertex as well. 
Back face culling is performed at this stage. To check if a face should be drawn or not, we test if the vertices are positioned in a clockwise or counterclockwise direction. If they are in a counterclockwise direction it means that the face is not visible and therefore should not be drawn. After this testing, each triangle is marked as visible or not visible. Before the rasterization step, we use thrust stream compaction to compact the primitves array to those that are visible only. 
At the beginning I did not implement this feature and therefore my triangles were "z fighting". The following picture shows the difference between having back face culling ON and OFF. The faces are colored according to their normal. As you can see when back face culling is turned off, the faces are fighting to be drawn and the resulting render is wrong.

![alt tag](https://github.com/paula18/Project4-Rasterizer/blob/master/bFComparison.PNG)

-------------------------------------------------------------------------------
Rasterization:
-------------------------------------------------------------------------------
Rasterization is the main step of the pipeline. I started by implementing the scan line algorithm using an edge list array and a active edge array that is updated each time a scaline intersects with an edge. However, I was not getting the results I expecting and I found an easier way of implementing the rasterization step using barycentric coordinates. 
First, we determine the bounding box of each traingle. Then, we loop for each point in the box and test if the point lies inside the triangle. If it does, we perform a depth comparison, and interpolate the normals, positions, color using baycentric coordinates to assign them to the depthbuffer. 
This was my first render with no fragment shader: a simple white triangle. 

![alt tag](https://github.com/paula18/Project4-Rasterizer/blob/master/triangleNoFragment.PNG)


-------------------------------------------------------------------------------
Fragment Shading:
-------------------------------------------------------------------------------
The fragment shader uses the light position, color and camera position to compute diffuse and specular shading following these models.

![alt tag](https://github.com/paula18/Project4-Rasterizer/blob/master/shadingModels.PNG)

The resulting colors are added to an ambient lighting that is calculated using the color of the fragment and the light color. 


-------------------------------------------------------------------------------
Rendering:
-------------------------------------------------------------------------------
In the rendering step takes care of sending the data from the depthbuffer to the framebuffer. I added anti-aliasing by supersampling each pixel. We look at the colors of the neighboring pixels and compute the average of these. 
The picture below shows the difference between rendering with anti-aliasing and rendering without. As the picture shows, when we turn on anti-aliasing the lines look smoother and the "jumps" are not as visible. 

![alt tag](https://github.com/paula18/Project4-Rasterizer/blob/master/aliasingComparison.PNG)


-------------------------------------------------------------------------------
Color and Normal Interpolation:
-------------------------------------------------------------------------------
Color and normal interpolation is performed using baricentric coordinates. The next image each vertex of the triangle is assigned a different color (red, green and blue) and the interior of the triangle is interpolated. 

![alt tag](https://github.com/paula18/Project4-Rasterizer/blob/master/triColorInter.PNG)

The following image is an example of a mesh in normals viewing mode and color interpolation view mode. 

![alt tag](https://github.com/paula18/Project4-Rasterizer/blob/master/bunnyColorAndNormals.PNG)
-------------------------------------------------------------------------------
Mouse and Keyboard Interaction:
-------------------------------------------------------------------------------

* D - Render with diffuse lighting. 
* N - View triangle normals. 
* S - Render with specular lighting.
* C - View color interpolation. 
* A - Turn on/off anti-aliasing

* Mouse wheel - zoom in/out
* Mouse right button - rotate camera



-------------------------------------------------------------------------------
PERFORMANCE EVALUATION
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
Back Face Culling
-------------------------------------------------------------------------------
The way I implemented back face culling has not effect of the performance. Whether this feature is turned ON or OFF does not affect the rate of the rasterizer. I tested this feature on the bunny, the cow and the Buda. The FPS stay the same. 

-------------------------------------------------------------------------------
Anti-Aliasing
-------------------------------------------------------------------------------

Anti-Aliasing affect the performance by a small amount. Even though the change in FPS is not very noticeable, we can see that when this feature is turned ON, the performance decreases. For instance, a render of the bunny with no anti-aliasing renders at 26 FPS, while turning this feature ON decreases it to 22 FPS.

-------------------------------------------------------------------------------
Eye Distance
-------------------------------------------------------------------------------
I also checked what impact the distance from the camera to the model has on the performance. As I expected, the closer the object is to the camera, the slowest the performance. The parallelization is done by primitive (triangles in our case) and therefore the closer the triangles are to the viewing point more work is done per primitive and therefore the running speed decreases.  


-------------------------------------------------------------------------------
VIDEO
-------------------------------------------------------------------------------


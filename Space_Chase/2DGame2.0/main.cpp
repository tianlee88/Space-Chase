
#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define GL_SILENCE_DEPRECATION

#include <vector>
#include <cmath>

#include <chrono>
#include <ctime>


#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>        // must be downloaded
#include <GL/freeglut.h>    // must be downloaded unless you have an Apple
#include <GL/glut.h>
#endif

#include <string>

const unsigned int windowWidth = 1024, windowHeight = 1024;

int TAB = 9;

int SPACE = 32;

int curSelectedObject = 0;

// OpenGL major and minor versions
int majorVersion = 3, minorVersion = 0;

//Bools tellilng when keys a and d are pressed down
//bool aPressed, dPressed = false;
bool keyboardState[256];


// row-major matrix 4x4
struct mat4
{
    float m[4][4];
public:
    mat4() {}
    mat4(float m00, float m01, float m02, float m03,
         float m10, float m11, float m12, float m13,
         float m20, float m21, float m22, float m23,
         float m30, float m31, float m32, float m33)
    {
        m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
        m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
        m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
        m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
    }
    
    mat4 operator*(const mat4& right)
    {
        mat4 result;
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                result.m[i][j] = 0;
                for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
            }
        }
        return result;
    }
    operator float*() { return &m[0][0]; }
};


// 3D point in homogeneous coordinates
struct vec4
{
    float v[4];
    
    vec4(float x = 0, float y = 0, float z = 0, float w = 1)
    {
        v[0] = x; v[1] = y; v[2] = z; v[3] = w;
    }
    
    vec4 operator*(const mat4& mat)
    {
        vec4 result;
        for (int j = 0; j < 4; j++)
        {
            result.v[j] = 0;
            for (int i = 0; i < 4; i++) result.v[j] += v[i] * mat.m[i][j];
        }
        return result;
    }
    
    vec4 operator+(const vec4& vec)
    {
        vec4 result(v[0] + vec.v[0], v[1] + vec.v[1], v[2] + vec.v[2], v[3] + vec.v[3]);
        return result;
    }
};


// 2D point in Cartesian coordinates
struct vec2
{
    float x, y;
    
    vec2(float x = 0.0, float y = 0.0) : x(x), y(y) {}
    
    vec2 operator+(const vec2& v)
    {        return vec2(x + v.x, y + v.y);
    }
};


class Camera{
    vec2 center;
    vec2 scalingFactors;
    
public:
    
    Camera(vec2 centerInput, vec2 scalingFactorInput){
        center = centerInput;
        scalingFactors = scalingFactorInput;
    };
    
    mat4 getViewTransformationMatrix(){
        
        mat4 T = mat4(1, 0, 0, 0,
                      0, 1, 0, 0,
                      0, 0, 1, 0,
                      -center.x, -center.y, 0, 1);
        
        
        mat4 S = mat4(1/scalingFactors.x, 0, 0 ,0,
                      0, 1/scalingFactors.y, 0, 0,
                      0, 0, 1, 0,
                      0, 0, 0, 1);
        
        mat4 V = S * T;
        
        return V;
    }
    
    void move(vec2 avatarPosition){
        center= avatarPosition;
        glutPostRedisplay();
    }
};

//global variables
bool shootCoolDownBool = false;
double shootCoolDownTimer = 0;
int playerLives = 3;
vec2 avatarPosition;
enum objectType {avatar, asteroid, projectile,enemyProjectile,life,explosion,enemy,scoreObject};
Camera camera(vec2(0,0), vec2(1,1));



//Telling the linker where to find the stbi_load function
extern "C" unsigned char* stbi_load(char const *filename, int *x, int *y, int *comp, int req_comp);

//Texture class
class Texture {
    unsigned int textureId;
public:
    Texture(const std::string& inputFileName){
        unsigned char* data;
        int width; int height; int nComponents = 4;
        
        data = stbi_load(inputFileName.c_str(), &width, &height, &nComponents, 0);
        
        if(data == NULL) { return; }
        
        glGenTextures(1, &textureId);
        glBindTexture(GL_TEXTURE_2D, textureId);
        
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        
        delete data;
    }
    
    void Bind()
    {
        glBindTexture(GL_TEXTURE_2D, textureId);
    }
};


//ABSTRACT CLASS
//responsible for compiling vertex/fragment shaders
class Shader{
    
protected:
    // handle of the shader program
    unsigned int shaderProgram;
    
    void getErrorInfo(unsigned int handle)
    {
        int logLen;
        glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
        if (logLen > 0)
        {
            char * log = new char[logLen];
            int written;
            glGetShaderInfoLog(handle, logLen, &written, log);
            printf("Shader log:\n%s", log);
            delete log;
        }
    }
    
    // check if shader could be compiled
    void checkShader(unsigned int shader, char * message)
    {
        int OK;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
        if (!OK)
        {
            printf("%s!\n", message);
            getErrorInfo(shader);
        }
    }
    
    // check if shader could be linked
    void checkLinking(unsigned int program)
    {
        int OK;
        glGetProgramiv(program, GL_LINK_STATUS, &OK);
        if (!OK)
        {
            printf("Failed to link shader program!\n");
            getErrorInfo(program);
        }
    }
    
public:
    //constructor
    Shader(){
        shaderProgram = 0;
    }
    
    //Compile the shader based on arguments from the subClasses
    void CompileShader(const char* vertexSource, const char* fragmentSource){
        
        // create vertex shader from string
        unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
        if (!vertexShader) { printf("Error in vertex shader creation\n"); exit(1); }
        
        glShaderSource(vertexShader, 1, &vertexSource, NULL);
        glCompileShader(vertexShader);
        checkShader(vertexShader, "Vertex shader error");
        
        // create fragment shader from string
        unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        if (!fragmentShader) { printf("Error in fragment shader creation\n"); exit(1); }
        
        glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
        glCompileShader(fragmentShader);
        checkShader(fragmentShader, "Fragment shader error");
        
        // attach shaders to a single program
        shaderProgram = glCreateProgram();
        if (!shaderProgram) { printf("Error in shader program creation\n"); exit(1); }
        
        glAttachShader(shaderProgram, vertexShader);
        glAttachShader(shaderProgram, fragmentShader);
    }
    
    void LinkShader(){
        // program packaging
        glLinkProgram(shaderProgram);
        checkLinking(shaderProgram);
    }
    
    //Destructor
    //in Java, we don't need explicit deallocation, but this is C++, so we must do our own garbage collection
    ~Shader(){
        glDeleteProgram(shaderProgram);
    }
    
    //Activivates the Shader Program
    void Run(){
        // make this program run
        glUseProgram(shaderProgram);
    }
    
    //Upload the transformation matrix
    virtual void UploadM(mat4 M){};
    
    virtual void UploadSamplerID(){};
    
  
    virtual void UploadSubTextureID(int t){};
    
};


class TexturedShader : public Shader
{
public:
    TexturedShader()
    {
        const char *vertexSource = R"(
#version 410
        precision highp float;
        
        in vec2 vertexPosition;
        in vec2 vertexTexCoord;
        uniform mat4 M;
        out vec2 texCoord;
        
        void main()
        {
            texCoord = vertexTexCoord;
            gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1) * M;
        }
        )";
        const char *fragmentSource = R"(
#version 410
        precision highp float;
        uniform sampler2D samplerUnit;
        in vec2 texCoord;
        out vec4 fragmentColor;
        
        void main()
        {
            fragmentColor = texture(samplerUnit, texCoord);
        }
        )";
        
        
        CompileShader(vertexSource, fragmentSource);
        
        // connect Attrib Array to input variables of the vertex shader
        glBindAttribLocation(shaderProgram, 0, "vertexPosition"); // vertexPosition gets values from Attrib Array 0
        glBindAttribLocation(shaderProgram, 1, "vertexTexCoord"); // vertexPosition gets values from Attrib Array 0

        // connect the fragmentColor to the frame buffer memory
        glBindFragDataLocation(shaderProgram, 0, "fragmentColor"); // fragmentColor goes to the frame buffer memory
        
        LinkShader();
    }
    
    
    //Upload the transformation matrix
    void UploadM(mat4 M){
        int location = glGetUniformLocation(shaderProgram, "M");
        if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, M);
        else printf("uniform M cannot be set\n");
    }
    
    void UploadSamplerID()
    {
        int samplerUnit = 0;
        int location = glGetUniformLocation(shaderProgram, "samplerUnit");
        if (location >= 0){
            glUniform1i(location, samplerUnit);
            glActiveTexture(GL_TEXTURE0 + samplerUnit);
        }
        else printf("uniform Sampler Id cannot be set\n");
    }
 
    void  UploadSubTextureID(int t){}
};


class AnimatedTexturedShader : public Shader{
    
public:
    AnimatedTexturedShader()
    {
        const char *vertexSource = R"(
#version 410
        precision highp float;

        in vec2 vertexPosition;
        in vec2 vertexTexCoord;
        uniform mat4 M;
        out vec2 texCoord;

        void main()
        {
            texCoord = vertexTexCoord;
            gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1) * M;
        }
        )";
        const char *fragmentSource = R"(
#version 410
        precision highp float;
        uniform sampler2D samplerUnit;
        uniform int subTextureID;
        in vec2 texCoord;
        out vec4 fragmentColor;
        
        void main()
        {
            int i = subTextureID % 6;
            int j = subTextureID / 6;
            fragmentColor = texture(samplerUnit, (vec2(i, j) + texCoord) / 6.0);
        }
        )";
        

        CompileShader(vertexSource, fragmentSource);

        // connect Attrib Array to input variables of the vertex shader
        glBindAttribLocation(shaderProgram, 0, "vertexPosition"); // vertexPosition gets values from Attrib Array 0
        glBindAttribLocation(shaderProgram, 1, "vertexTexCoord"); // vertexPosition gets values from Attrib Array 0


        // connect the fragmentColor to the frame buffer memory
        glBindFragDataLocation(shaderProgram, 0, "fragmentColor"); // fragmentColor goes to the frame buffer memory

        LinkShader();
    }

    //Upload the transformation matrix
    void UploadM(mat4 M){
        int location = glGetUniformLocation(shaderProgram, "M");
        if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, M);
        else printf("uniform M cannot be set\n");
    }

    void UploadSamplerID()
    {
        int subTextureId = 1;
        int location = glGetUniformLocation(shaderProgram, "samplerUnit");

            glUniform1i(location, subTextureId);
            glActiveTexture(GL_TEXTURE0 + subTextureId);
    }

    void UploadSubTextureID(int t){
        int location = glGetUniformLocation(shaderProgram, "subTextureID");
            glUniform1i(location, t);
            glActiveTexture(GL_TEXTURE0 + t);
    }
};


//Parent for all geometries
class Geometry{
    
    //Only subclasses will be able to access this variable
protected:
    // vertex array object id
    unsigned int vao;
    
public:
    //Constructor
    Geometry(){
        //Initialize the vao global var
        glGenVertexArrays(1, &vao);    // create a vertex array object
    }
    
    //Draw the geometry - Purely virtual function, an abstract class (because has virtual method)
    //abstract classes cannot be instantiated - we use this as an INTERFACE!
    virtual void Draw() = 0;
    
}; //Don't forget that semicolon!!!

//inherits Geometry class
class Triangle : public Geometry
{

public:
    //Constructor, replaces the Create() method
    Triangle()
    {
        glBindVertexArray(vao);        // make it active
        
        unsigned int vbo;        // vertex buffer object
        glGenBuffers(1, &vbo);        // generate a vertex buffer object
        
        // vertex coordinates: vbo -> Attrib Array 0 -> vertexPosition of the vertex shader
        glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
        static float vertexCoords[] = { 0, 0, 1, 0, 0, 1 };    // vertex data on the CPU
        
        glBufferData(GL_ARRAY_BUFFER,    // copy to the GPU
                     sizeof(vertexCoords),    // size of the vbo in bytes
                     vertexCoords,        // address of the data array on the CPU
                     GL_STATIC_DRAW);    // copy to that part of the memory which is not modified
        
        // map Attribute Array 0 to the currently bound vertex buffer (vbo)
        glEnableVertexAttribArray(0);
        
        // data organization of Attribute Array 0
        glVertexAttribPointer(0,    // Attribute Array 0
                              2, GL_FLOAT,        // components/attribute, component type
                              GL_FALSE,        // not in fixed point format, do not normalized
                              0, NULL);        // stride and offset: it is tightly packed
    }
    
    void Draw()
    {
        glBindVertexArray(vao);    // make the vao and its vbos active playing the role of the data source
        glDrawArrays(GL_TRIANGLES, 0, 3); // draw a single triangle with vertices defined in vao
    }
};


class Quad : public Geometry{
    unsigned int vbo;
    
public:
    Quad(){
        glBindVertexArray(vao);
        
        glGenBuffers(1, &vbo);
        
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        static float vertexCoords[] = { -.5, -.5,   .5, -.5 ,   -.5, .5,  .5, .5};
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
    }
    
    void Draw(){
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }
};


class TexturedQuad : public Quad
{
    unsigned int vboTex;
    
public:
    
    TexturedQuad()
    {
        glBindVertexArray(vao);
        glGenBuffers(1, &vboTex);
        
        glBindBuffer(GL_ARRAY_BUFFER, vboTex);
        static float textureCoords[] = { 0, 0,   1, 0 ,   0, 1,  1, 1};
        glBufferData(GL_ARRAY_BUFFER, sizeof(textureCoords), textureCoords, GL_STATIC_DRAW);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);
        
    }
    
    void Draw()
    {
        glEnable(GL_BLEND); // necessary for transparent pixels
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glDisable(GL_BLEND);
    }
};



class Material{
    Shader* shader;
    Texture* texture;
    float time;
    
public:
    Material(Shader* shader, Texture* texture) :
    shader(shader), texture(texture) {
    }

    
    Texture* getTexture(){
        return texture;
    }
    
    void UploadAttributes(float time)
    {
        if(texture){
            shader->UploadSamplerID();
            shader->UploadSubTextureID(time);
            texture->Bind();
        }
        else{
            printf("texture failure abort abort abort");
        }
    }
};


class Mesh{
    Material* material;
    Geometry* geometry;
    
public:
    
    Mesh(Geometry* geometry, Material* material) : material(material), geometry(geometry){
    }
    
    void Draw(float time){
        material->UploadAttributes(time);
        geometry->Draw();
    }
    
    Geometry* getGeometry(){
        return geometry;
    }
    
    Material* getMaterial(){
        return material;
    }
    
};


class Object{
    
public:
    Shader* shader;
    Mesh* mesh;
    vec2 scaling;
    vec2 position;
    vec2 velocity;
    float orientation;
    float time;
    
    //Constructor
    Object(Shader* shader, Mesh* mesh,vec2 position,vec2 scaling,vec2 velocity, float orientation,float time):
    shader(shader), mesh(mesh), position(position), scaling(scaling),velocity(velocity), orientation(orientation), time(time){}
    
    //Upload attrs to the GPU
    void UploadAttributes(){
        shader->Run();
        mat4 tMatrix = createMatrix();
        shader->UploadM(tMatrix);
    }
    
    mat4 createMatrix(){
        // calculate T, S, R from position, scaling, and orientation
        
        mat4 T = mat4(1, 0, 0, 0,
                      0, 1, 0, 0,
                      0, 0, 1, 0,
                      position.x, position.y, 0, 1);
        
        double rotationDegrees = (orientation/180) * M_PI;
        
        mat4 R = mat4(cos(rotationDegrees), sin(rotationDegrees), 0, 0,
                      -sin(rotationDegrees), cos(rotationDegrees), 0, 0,
                      0, 0, 1, 0,
                      0, 0, 0, 1);
        
        mat4 S = mat4(scaling.x, 0, 0 ,0,
                      0, scaling.y, 0, 0,
                      0, 0, 1, 0,
                      0, 0, 0, 1);
        
        mat4 V = camera.getViewTransformationMatrix();
        
        mat4 M = S * R * T * V;
        
        return M;
        
    }
    
    //Draw the object
    void Draw(){
        UploadAttributes();
        mesh->Draw(time);
    }
    
    Shader* getShader(){
        return shader;
    }
    
    Mesh* getMesh(){
        return mesh;
    }
    
    vec2 getPosition(){
        return position;
    }
    
    vec2 getScaling(){
        return scaling;
    }
    
    float getOrientation(){
        return orientation;
    }
    
    vec2 getOrientationVector(){
        return vec2(cos(orientation*M_PI/180),sin(orientation*M_PI/180));
    }
    
    virtual int getObjectType()=0;
    
    bool checkCollision(Object* other){
        if((pow((other->getPosition().x - position.x),2) + pow((other->getPosition().y - position.y), 2)) < pow((other->getScaling().x + scaling.y),4))
        {
            return true;
        }
        return false;
    }
    
    virtual void moveObject(double dt) =0;
    virtual float getTimeUntilDeath()=0;
    virtual void updateTimeUntilDeath(float dt)=0;
    virtual void setOrientation(float newOrientation)=0;
};

class AvatarObject : public Object{
    
    public:
    AvatarObject(Shader* shader, Mesh* mesh,vec2 position,vec2 scaling,vec2 velocity, float orientation,float time)
    :Object(shader, mesh, position, scaling,velocity, orientation,time){}
    
    int getObjectType(){
        return avatar;
    }
    void moveObject(double dt){
            if(keyboardState['a']){
                orientation += dt * 45 *4.0;
            }
            if (keyboardState['d']){
                orientation += dt * -45*4.0;
            }
            //backwards and forwards depend on the direction that the avatar is facing.
            if (keyboardState['w']){
                velocity.x+= getOrientationVector().x *.0008;
                velocity.y+= getOrientationVector().y *.0008;
            }
            if (keyboardState['s']){
                velocity.x-= getOrientationVector().x*.0008;
                velocity.y-= getOrientationVector().y*.0008;
            }
            position = position + vec2(velocity.x, velocity.y);
        
            //drag logic
            velocity.x*=.975;
            velocity.y*= .975;
        
            glutPostRedisplay();
        }
    
    void updateTimeUntilDeath(float dt){}
    float getTimeUntilDeath(){return 5;}
    void setOrientation(float newOrientation){}
    
};

class HorizontalEnemyObject : public Object{
    
public:
    int direction;
    HorizontalEnemyObject(Shader* shader, Mesh* mesh,vec2 position,vec2 scaling,vec2 velocity, float orientation,float time,int new_direction)
    :Object(shader, mesh, position, scaling,velocity, orientation,time){
        direction = new_direction;
    }
    
    int getObjectType(){
        return enemy;
    }
    
    void setOrientation(float newOrientation){
        vec2 newOrientationVector = vec2(avatarPosition.x-position.x,avatarPosition.y-position.y);
        
        float lengthNewOrientationVector = sqrt(pow(newOrientationVector.x, 2)+pow(newOrientationVector.y, 2));
        
        vec2 newOrientationVectorNormalized = vec2(newOrientationVector.x/lengthNewOrientationVector,newOrientationVector.y/
                                                   lengthNewOrientationVector);
        orientation = (std::atan2(newOrientationVectorNormalized.y,newOrientationVectorNormalized.x)*180/M_PI)+270 ;
    }
    
    void moveObject(double dt){
        updateTimeUntilDeath(dt);
        setOrientation(0);

        position.x = direction *(time/10)+avatarPosition.x - (2*direction);
        position.y = sin(time/2.5)+avatarPosition.y;
        glutPostRedisplay();
    }
    
    void updateTimeUntilDeath(float dt){
        time += dt*2;
    }
    
    float getTimeUntilDeath(){
        return time;
    }
};

class VerticalEnemyObject : public Object{
    
public:
    int direction;
    VerticalEnemyObject(Shader* shader, Mesh* mesh,vec2 position,vec2 scaling,vec2 velocity, float orientation,float time,int new_direction)
    :Object(shader, mesh, position, scaling,velocity, orientation,time){
        direction = new_direction;
    }
    
    int getObjectType(){
        return enemy;
    }
    
    void setOrientation(float newOrientation){
        vec2 newOrientationVector = vec2(avatarPosition.x-position.x,avatarPosition.y-position.y);
        
        float lengthNewOrientationVector = sqrt(pow(newOrientationVector.x, 2)+pow(newOrientationVector.y, 2));
        
        vec2 newOrientationVectorNormalized = vec2(newOrientationVector.x/lengthNewOrientationVector,newOrientationVector.y/
                                                   lengthNewOrientationVector);
        orientation = (std::atan2(newOrientationVectorNormalized.y,newOrientationVectorNormalized.x)*180/M_PI)+270 ;
    }
    
    void moveObject(double dt){
        updateTimeUntilDeath(dt);
        setOrientation(0);
        position.x = sin(time/2.5)+avatarPosition.x;
        position.y = direction *(time/10)+avatarPosition.y - (2*direction);

        glutPostRedisplay();
    }
    
    void updateTimeUntilDeath(float dt){
        time += dt*2;
    }
    
    float getTimeUntilDeath(){
        return time;
    }
};


class ProjectileObject : public Object{
    
public:
    ProjectileObject(Shader* shader, Mesh* mesh,vec2 position,vec2 scaling,vec2 velocity, float orientation, float time)
    :Object(shader, mesh, position, scaling,velocity, orientation, time){}
    
    int getObjectType(){
        return projectile;
    }
    void moveObject(double dt){
        
            updateTimeUntilDeath(dt);
            velocity.x= getOrientationVector().x *.03;
            velocity.y= getOrientationVector().y *.03;
            position = position + vec2(velocity.x, velocity.y);
        }

    void updateTimeUntilDeath(float dt){
        time += dt*5;
    }
    
    float getTimeUntilDeath(){
        return time;
    }
    
    void setOrientation(float newOrientation){}

   
};

class EnemyProjectileObject : public Object{
    
public:
    EnemyProjectileObject(Shader* shader, Mesh* mesh,vec2 position,vec2 scaling,vec2 velocity, float orientation, float time)
    :Object(shader, mesh, position, scaling,velocity, orientation, time){}
    
    int getObjectType(){
        return enemyProjectile;
    }
    
    void setOrientation(float newOrientation){
        vec2 newOrientationVector = vec2(avatarPosition.x-position.x,avatarPosition.y-position.y);
        
        float lengthNewOrientationVector = sqrt(pow(newOrientationVector.x, 2)+pow(newOrientationVector.y, 2));
        
        vec2 newOrientationVectorNormalized = vec2(newOrientationVector.x/lengthNewOrientationVector,newOrientationVector.y/
                                                   lengthNewOrientationVector);
        orientation = (std::atan2(newOrientationVectorNormalized.y,newOrientationVectorNormalized.x)*180/M_PI)+90 ;
    }
    
    void moveObject(double dt){
        updateTimeUntilDeath(dt);
        setOrientation(0);
        vec2 newOrientationVector = vec2(avatarPosition.x-position.x,avatarPosition.y-position.y);
        
        float lengthNewOrientationVector = sqrt(pow(newOrientationVector.x, 2)+pow(newOrientationVector.y, 2));
        
        vec2 newOrientationVectorNormalized = vec2(newOrientationVector.x/lengthNewOrientationVector,newOrientationVector.y/
                                                   lengthNewOrientationVector);
        
        velocity.x= newOrientationVectorNormalized.x *.005;
        velocity.y= newOrientationVectorNormalized.y *.005;
        position = position + vec2(velocity.x, velocity.y);
    }
    
    void updateTimeUntilDeath(float dt){}
    
    float getTimeUntilDeath(){
        return time;
    }
    
    
    
};

class AsteroidObject : public Object{
    
public:
    AsteroidObject(Shader* shader, Mesh* mesh,vec2 position,vec2 scaling,vec2 velocity, float orientation,float time)
    :Object(shader, mesh, position, scaling,velocity, orientation, time){}
    
    int getObjectType(){
        return asteroid;
    }
    
    void moveObject(double dt){
        velocity.x= getOrientationVector().x *.005;
        velocity.y= getOrientationVector().y *.005;
        position = position + vec2(velocity.x, velocity.y);
        
        int randomAngle = (rand() % (180 + 1 - 0));
        
        if(position.x>avatarPosition.x+1.0)
        {

            position.x-=2;
            orientation+= randomAngle;
        }
        if(position.x<avatarPosition.x-1.0)
        {
            position.x+=2;
            orientation+= randomAngle;
        }
        if(position.y>avatarPosition.y+1.0)
        {
            position.y-=2;
            orientation+= randomAngle;
        }
        if(position.y<avatarPosition.y-1.0)
        {
            position.y+=2;
            orientation+= randomAngle;
        }
        
        
    }

    void updateTimeUntilDeath(float dt){}
    float getTimeUntilDeath(){return 5;}
    void setOrientation(float newOrientation){
        orientation = newOrientation;
    }
};

class AvatarLifeObject : public Object{

public:
    int lifeNumber;
    
    AvatarLifeObject(Shader* shader, Mesh* mesh,vec2 position,vec2 scaling,vec2 velocity, float orientation,float time,int life_Number)
    :Object(shader, mesh, position, scaling,velocity, orientation,time){
        lifeNumber = life_Number;
    }
    
    int getObjectType(){
        return life;
    }
    
    void moveObject(double dt){
        if(lifeNumber==1){
            position.x= avatarPosition.x-.9;
            position.y= avatarPosition.y+.75;
        }
        else if(lifeNumber==2){
            position.x= avatarPosition.x-.8;
            position.y= avatarPosition.y+.75;
        }
        else if(lifeNumber==3){
            position.x= avatarPosition.x-.7;
            position.y= avatarPosition.y+.75;
        }
        
        glutPostRedisplay();
    }
    void updateTimeUntilDeath(float dt){}
    float getTimeUntilDeath(){return 5;}
    void setOrientation(float newOrientation){}
};

class ScoreObject : public Object{
    
public:
    int scoreNumber;
    int scoreDigit;

    
    ScoreObject(Shader* shader, Mesh* mesh,vec2 position,vec2 scaling,vec2 velocity, float orientation,float time,int score_number,int score_digit)
    :Object(shader, mesh, position, scaling,velocity, orientation,time){
        scoreNumber = score_number;
        scoreDigit = score_digit;
    }
    
    int getscoreNumber(){
        return scoreNumber;
    }
    
    int getObjectType(){
        return scoreObject;
    }
    
    void moveObject(double dt){
        if(scoreDigit==1){
            position.x= avatarPosition.x+.9;
            position.y= avatarPosition.y+.75;
        }
        else if(scoreDigit==2){
            position.x= avatarPosition.x+.8;
            position.y= avatarPosition.y+.75;
        }
        glutPostRedisplay();
    }
    
    void updateTimeUntilDeath(float dt){}
    float getTimeUntilDeath(){return 5;}
    void setOrientation(float newOrientation){}
    
};

class ExplosionObject : public Object{
    
public:
    float timeUntilDeath;
    
    ExplosionObject(Shader* shader, Mesh* mesh,vec2 position,vec2 scaling,vec2 velocity, float orientation,float time)
    :Object(shader, mesh, position, scaling,velocity, orientation, time){
    }
    
    void updateTimeUntilDeath(float dt){
        time += dt*10;
    }
    
    float getTimeUntilDeath(){
        return time;
    }
    
    int getObjectType(){
        return explosion;
    }
    
    void moveObject(double dt){
        updateTimeUntilDeath(dt);
    }
    void setOrientation(float newOrientation){}

};

class Scene{
    Texture* texture;
    TexturedShader* texturedShader;
    AnimatedTexturedShader* animTexShader;
    std::vector<Material*> materials;
    std::vector<Texture*> textures;
    Geometry* geometry;
    std::vector<Mesh*> meshes;
    std::vector<Object*> objects;
    std::vector<Object*> scoreObjects;

    int numAsteroids = 3;
    int numEnemies = 0;
    int score = 0;
    bool boolGameOver = false;
    bool scoreChanged = false;

public:
   
    Scene(){
        texturedShader = 0;
        animTexShader = 0;
        geometry = 0;
    }

    void Initialize(){
        texturedShader = new TexturedShader();
        animTexShader = new AnimatedTexturedShader();

        textures.push_back(new Texture("/Users/tianlee/Desktop/graphics/2DGame2.0/2DGame2.0/spaceship.png"));
        textures.push_back(new Texture("/Users/tianlee/Desktop/graphics/2DGame2.0/2DGame2.0/asteroid.png"));
        textures.push_back(new Texture("/Users/tianlee/Desktop/graphics/2DGame2.0/2DGame2.0/bullet2.png"));
        textures.push_back(new Texture("/Users/tianlee/Desktop/graphics/2DGame2.0/2DGame2.0/boom.png"));
        textures.push_back(new Texture("/Users/tianlee/Desktop/graphics/2DGame2.0/2DGame2.0/enemy.png"));
        textures.push_back(new Texture("/Users/tianlee/Desktop/graphics/2DGame2.0/2DGame2.0/enemyProjectile.png"));
        textures.push_back(new Texture("/Users/tianlee/Desktop/graphics/2DGame2.0/2DGame2.0/gameOver.png"));
        textures.push_back(new Texture("/Users/tianlee/Desktop/graphics/2DGame2.0/2DGame2.0/pressPToContinue.png"));
        
        //textures for score
        textures.push_back(new Texture("/Users/tianlee/Desktop/graphics/2DGame2.0/2DGame2.0/0.png"));
        textures.push_back(new Texture("/Users/tianlee/Desktop/graphics/2DGame2.0/2DGame2.0/1.png"));
        textures.push_back(new Texture("/Users/tianlee/Desktop/graphics/2DGame2.0/2DGame2.0/2.png"));
        textures.push_back(new Texture("/Users/tianlee/Desktop/graphics/2DGame2.0/2DGame2.0/3.png"));
        textures.push_back(new Texture("/Users/tianlee/Desktop/graphics/2DGame2.0/2DGame2.0/4.png"));
        textures.push_back(new Texture("/Users/tianlee/Desktop/graphics/2DGame2.0/2DGame2.0/5.png"));
        textures.push_back(new Texture("/Users/tianlee/Desktop/graphics/2DGame2.0/2DGame2.0/6.png"));
        textures.push_back(new Texture("/Users/tianlee/Desktop/graphics/2DGame2.0/2DGame2.0/7.png"));
        textures.push_back(new Texture("/Users/tianlee/Desktop/graphics/2DGame2.0/2DGame2.0/8.png"));
        textures.push_back(new Texture("/Users/tianlee/Desktop/graphics/2DGame2.0/2DGame2.0/9.png"));

        materials.push_back(new Material(texturedShader, textures[0])); //material for ship
        materials.push_back(new Material(texturedShader, textures[1])); //material for asteroid
        materials.push_back(new Material(texturedShader, textures[2])); //material for projectile
        materials.push_back(new Material(animTexShader, textures[3])); //explosion
        materials.push_back(new Material(texturedShader, textures[4])); //enemy
        materials.push_back(new Material(texturedShader, textures[5])); //enemyProjectile
        materials.push_back(new Material(texturedShader, textures[6])); //gameOver
        materials.push_back(new Material(texturedShader, textures[7])); //gameOver
        
        // materials for score
        materials.push_back(new Material(texturedShader, textures[8]));
        materials.push_back(new Material(texturedShader, textures[9]));
        materials.push_back(new Material(texturedShader, textures[10]));
        materials.push_back(new Material(texturedShader, textures[11]));
        materials.push_back(new Material(texturedShader, textures[12]));
        materials.push_back(new Material(texturedShader, textures[13]));
        materials.push_back(new Material(texturedShader, textures[14]));
        materials.push_back(new Material(texturedShader, textures[15]));
        materials.push_back(new Material(texturedShader, textures[16]));
        materials.push_back(new Material(texturedShader, textures[17]));


        geometry = new TexturedQuad();

        meshes.push_back(new Mesh(geometry, materials[0]));
        meshes.push_back(new Mesh(geometry, materials[1]));
        meshes.push_back(new Mesh(geometry, materials[2]));
        meshes.push_back(new Mesh(geometry, materials[3]));
        meshes.push_back(new Mesh(geometry, materials[4]));
        meshes.push_back(new Mesh(geometry, materials[5]));
        meshes.push_back(new Mesh(geometry, materials[6]));
        meshes.push_back(new Mesh(geometry, materials[7]));
        
        // meshes for score
        meshes.push_back(new Mesh(geometry, materials[8]));
        meshes.push_back(new Mesh(geometry, materials[9]));
        meshes.push_back(new Mesh(geometry, materials[10]));
        meshes.push_back(new Mesh(geometry, materials[11]));
        meshes.push_back(new Mesh(geometry, materials[12]));
        meshes.push_back(new Mesh(geometry, materials[13]));
        meshes.push_back(new Mesh(geometry, materials[14]));
        meshes.push_back(new Mesh(geometry, materials[15]));
        meshes.push_back(new Mesh(geometry, materials[16]));
        meshes.push_back(new Mesh(geometry, materials[17]));

        //avatar
        objects.push_back(new AvatarObject(texturedShader, meshes[0], vec2(0, -.3), vec2(.1,.1) ,vec2(0,0), 90,5));
        //lives left avatars
        objects.push_back(new AvatarLifeObject(texturedShader, meshes[0], vec2(-.9, .5), vec2(.075,.075) ,vec2(0,0), 90,0,1));
        objects.push_back(new AvatarLifeObject(texturedShader, meshes[0], vec2(-.8, .5), vec2(.075,.075) ,vec2(0,0), 90,0,2));
        objects.push_back(new AvatarLifeObject(texturedShader, meshes[0], vec2(-.7, .5), vec2(.075,.075) ,vec2(0,0), 90,0,3));
        
        //asteroids
        objects.push_back(new AsteroidObject(texturedShader, meshes[1], vec2(.4, 0), vec2(.25,.25), vec2(0,0), 90,1));
        objects.push_back(new AsteroidObject(texturedShader, meshes[1], vec2(0, .4), vec2(.25,.25), vec2(0,0), 0,2));
        objects.push_back(new AsteroidObject(texturedShader, meshes[1], vec2(-.4, 0), vec2(.25,.25), vec2(0,0), 180,3));
        
        //initial score object
        scoreObjects.push_back(new ScoreObject(texturedShader, meshes[8], vec2(0, 0), vec2(.1,.1), vec2(0,0), 0,0,0,1));
    
    }
    
    void updateAvatarPosition(){
        avatarPosition = getAvatarPosition();
    }
    
    vec2 getAvatarPosition(){
        return objects[0]->getPosition();
    }

    void spawnProjectile(){
        if(boolGameOver==false){
            if(shootCoolDownBool==false){
                objects.push_back(new ProjectileObject(texturedShader, meshes[2], objects[0]->getPosition(),vec2(.1,.1),vec2(0,0),objects[0]->getOrientation(),0));
                shootCoolDownBool = true;
                shootCoolDownTimer = .20;
            }
        }
    }
    
    void spawnEnemyProjectile(vec2 enemyPosition){
        if(boolGameOver==false){
            objects.push_back(new EnemyProjectileObject(texturedShader, meshes[5], enemyPosition,
                                                        vec2(.2,.2),vec2(0,0),0,0));
        }
    }

    void spawnExplosion(vec2 position){
        if(boolGameOver==false){
            objects.push_back(new ExplosionObject(animTexShader,meshes[3],position,vec2(.25,.25),vec2(0,0),0,0));
        }
    }
    
    void spawnAsteroids(){
        if(boolGameOver==false){
            if(numAsteroids<4){
                objects.push_back(new AsteroidObject(texturedShader, meshes[1], vec2(getAvatarPosition().x+1,0), vec2(.25,.25), vec2(0,0), 90,5));
                numAsteroids++;
            }
        }
    }
    
    void spawnEnemies(){
        if(boolGameOver==false){
            if(numEnemies<2){
                int randomEnemyInt = 1 + rand() % (( 4 + 1 ) - 1);
                
                if(randomEnemyInt ==1 ){
                    objects.push_back(new VerticalEnemyObject(texturedShader, meshes[4], vec2(getAvatarPosition().x-1,0), vec2(.25,.25), vec2(0,0), 0,5,1));
                    numEnemies++;
                }
                else if(randomEnemyInt ==2 ){
                    objects.push_back(new VerticalEnemyObject(texturedShader, meshes[4], vec2(getAvatarPosition().x-1,0), vec2(.25,.25), vec2(0,0), 0,5,-1));
                    numEnemies++;
                }
                else if(randomEnemyInt ==3 ){
                    objects.push_back(new HorizontalEnemyObject(texturedShader, meshes[4], vec2(getAvatarPosition().x-1,0), vec2(.25,.25), vec2(0,0), 0,5,1));
                    numEnemies++;
                }
                else if(randomEnemyInt ==4 ){
                    objects.push_back(new HorizontalEnemyObject(texturedShader, meshes[4], vec2(getAvatarPosition().x-1,0), vec2(.25,.25), vec2(0,0), 0,5,-1));
                    numEnemies++;
                }
                
            }
        }
    }
    
    void Interact(){
        for(int i =0; i<objects.size(); i++)
        {
            for(int j =i+1; j<objects.size(); j++)
            {
                if(objects[i]->checkCollision(objects[j]))
                {
                    if( objects[i]->getObjectType() == asteroid && objects[j]->getObjectType() == projectile)
                    {
                        spawnExplosion(objects[i]->getPosition());
                        objects.erase(objects.begin()+i);
                        objects.erase(objects.begin()+j-1);
                        numAsteroids--;
                    }
                    else if( objects[i]->getObjectType() == asteroid && objects[j]->getObjectType() == asteroid)
                    {
                        //bounce logic between two asteroids
                        vec2 bounceVectorI = vec2(objects[i]->getPosition().x - objects[j]->getPosition().x,objects[i]->getPosition().y -                          objects[j]->getPosition().y);
                        vec2 bounceVectorJ = vec2(objects[j]->getPosition().x - objects[i]->getPosition().x,objects[j]->getPosition().y -                          objects[i]->getPosition().y);
                        
                        float lengthBouncevecI = sqrt(pow(bounceVectorI.x, 2)+pow(bounceVectorI.y, 2));
                        float lengthBouncevecJ = sqrt(pow(bounceVectorJ.x, 2)+pow(bounceVectorJ.y, 2));
                        
                        vec2 normalizedBounceVectorI = vec2(bounceVectorI.x/lengthBouncevecI,bounceVectorI.y/lengthBouncevecI);
                        vec2 normalizedBounceVectorJ = vec2(bounceVectorJ.x/lengthBouncevecJ,bounceVectorJ.y/lengthBouncevecJ);
                        
                        vec2 resultantVectorI = vec2 ((normalizedBounceVectorI.x + objects[i]->getOrientationVector().x)/2,
                                                      (normalizedBounceVectorI.y+ objects[i]->getOrientationVector().y)/2);
                        vec2 resultantVectorJ = vec2 ((normalizedBounceVectorJ.x + objects[j]->getOrientationVector().x)/2,
                                                      (normalizedBounceVectorJ.y+ objects[j]->getOrientationVector().y)/2);
       
                        objects[i]->setOrientation((std::atan2(resultantVectorI.y,resultantVectorI.x))*180/M_PI);
                        objects[j]->setOrientation((std::atan2(resultantVectorJ.y,resultantVectorJ.x))*180/M_PI);

                    }
                    else if( objects[i]->getObjectType() == asteroid && objects[j]->getObjectType() == enemyProjectile)
                    {
                        spawnExplosion(objects[i]->getPosition());
                        objects.erase(objects.begin()+i);
                        objects.erase(objects.begin()+j-1);
                        numAsteroids--;
                    }
                    else if( objects[i]->getObjectType() == asteroid && objects[j]->getObjectType() == enemy)
                    {
                        spawnExplosion(objects[i]->getPosition());
                        spawnExplosion(objects[j]->getPosition());
                        objects.erase(objects.begin()+i);
                        objects.erase(objects.begin()+j-1);
                        numAsteroids--;
                        numEnemies--;
                    }
                    else if( objects[i]->getObjectType() == enemyProjectile && objects[j]->getObjectType() == projectile)
                    {
                        spawnExplosion(objects[j]->getPosition());
                        objects.erase(objects.begin()+i);
                        objects.erase(objects.begin()+j-1);
                        score++;
                        scoreChanged=true;
                    }
                    else if( objects[i]->getObjectType() == enemy && objects[j]->getObjectType() == projectile)
                    {
                        spawnExplosion(objects[j]->getPosition());
                        objects.erase(objects.begin()+i);
                        objects.erase(objects.begin()+j-1);
                        numEnemies--;
                        score++;
                        scoreChanged=true;
                    }
                   
                    else if( objects[i]->getObjectType() == avatar && objects[j]->getObjectType() == asteroid)
                    {
                        if(playerLives>0){
                            spawnExplosion(objects[j]->getPosition());
                            objects.erase(objects.begin()+playerLives);
                            playerLives--;
                            objects.erase(objects.begin()+j-1);
                            numAsteroids--;
                        }
                        else{
                            objects.erase(objects.begin()+i);
                        }
                    }
                    else if( objects[i]->getObjectType() == avatar && objects[j]->getObjectType() == enemyProjectile)
                    {
                        if(playerLives>0){
                            spawnExplosion(objects[j]->getPosition());
                            objects.erase(objects.begin()+playerLives);
                            playerLives--;
                            objects.erase(objects.begin()+j-1);
                        }
                        else{
                            objects.erase(objects.begin()+i);
                        }
                    }
                }
            }
        }
    }
    
    void checkForDeath(float dt){
        for(int i =0; i<objects.size(); i++)
        {
            if(objects[i]->getTimeUntilDeath()>30){
                if(objects[i]->getObjectType()==enemy){numEnemies--;}
                objects.erase(objects.begin()+i);
            }
            else if(objects[i]->getTimeUntilDeath()<30){
                objects[i]->updateTimeUntilDeath(dt);
            }
        }
    }
    
    void updateScore(){
        if(scoreChanged==true){
            if(score<=9){
                scoreObjects.clear();
                scoreObjects.push_back(new ScoreObject(texturedShader, meshes[score+8], vec2(.9+avatarPosition.x, .75+avatarPosition.y), vec2(.1,.1),
                                                       vec2(0,0), 0,0,score,1));
                scoreChanged=false;
            }
            else{
                scoreObjects.clear();
                scoreObjects.push_back(new ScoreObject(texturedShader, meshes[(score%10)+8],
                                                       vec2(avatarPosition.x +.9, avatarPosition.y +.75),vec2(.1,.1), vec2(0,0), 0,0,score%10,1));
                int secondDigitValue = score/10;
                scoreObjects.push_back(new ScoreObject(texturedShader, meshes[secondDigitValue+8], vec2(avatarPosition.x +.8, avatarPosition.y +.75),
                                                       vec2(.1,.1), vec2(0,0), 0,0,secondDigitValue,2));
                scoreChanged=false;
            }
        }
    }
    
    void gameOver(){
        if(playerLives==0){
            objects.clear();
            objects.push_back(new AvatarObject(texturedShader, meshes[6], vec2(0, 0), vec2(1.5,1.5) ,vec2(0,0), 180,5));
            objects.push_back(new AvatarObject(texturedShader, meshes[7], vec2(0,-.75), vec2(.5,.25) ,vec2(0,0), 180,5));
            playerLives= 3;
            boolGameOver = true;
        }
    }
    
    void restartGame(){
        if(keyboardState['p']){
            objects.clear();
            scoreObjects.clear();
            
            //avatar
            objects.push_back(new AvatarObject(texturedShader, meshes[0], vec2(0, -.3), vec2(.1,.1) ,vec2(0,0), 90,5));
            //lives left avatars
            objects.push_back(new AvatarLifeObject(texturedShader, meshes[0], vec2(-.9, .5), vec2(.075,.075) ,vec2(0,0), 90,0,1));
            objects.push_back(new AvatarLifeObject(texturedShader, meshes[0], vec2(-.8, .5), vec2(.075,.075) ,vec2(0,0), 90,0,2));
            objects.push_back(new AvatarLifeObject(texturedShader, meshes[0], vec2(-.7, .5), vec2(.075,.075) ,vec2(0,0), 90,0,3));
            
            objects.push_back(new AsteroidObject(texturedShader, meshes[1], vec2(1, 0), vec2(.25,.25), vec2(0,0), 90,1));
            objects.push_back(new AsteroidObject(texturedShader, meshes[1], vec2(0, 1), vec2(.25,.25), vec2(0,0), 0,2));
            objects.push_back(new AsteroidObject(texturedShader, meshes[1], vec2(-1, 0), vec2(.25,.25), vec2(0,0), 180,3));
            objects.push_back(new AsteroidObject(texturedShader, meshes[1], vec2(0, -1), vec2(.25,.25), vec2(0,0), 180,3));
            
            scoreObjects.push_back(new ScoreObject(texturedShader, meshes[8], vec2(0, 0), vec2(.1,.1), vec2(0,0), 0,0,0,1));
            
            boolGameOver = false;
            numEnemies=0;
            numAsteroids=4;
            score = 0;
        }
    }
    
    void Draw(){
        for (int i = 0; i < objects.size(); i++){
            objects[i]->Draw();
        }
        for (int j = 0; j < scoreObjects.size(); j++){
            scoreObjects[j]->Draw();
        }
    }
    
    void moveObjects(double dt,double elapsedTime){
        updateAvatarPosition();
        for(int i= 0 ; i < objects.size();i++){
            objects[i]->moveObject(dt);
            if(objects[i]->getObjectType() == enemy){
                if( fmod(elapsedTime,5.0)>0 && fmod(elapsedTime,5.0)<.02){
                    spawnEnemyProjectile(objects[i]->getPosition());
                }
            }
        }
        
        for (int j = 0; j < scoreObjects.size(); j++){
            scoreObjects[j]->moveObject(0);
        }
    }
     
    //Destructor
    ~Scene(){
        for(int i = 0; i < materials.size(); i++) delete materials[i];
        delete geometry;
        for(int i = 0; i < meshes.size(); i++) delete meshes[i];
        for(int i = 0; i < objects.size(); i++) delete objects[i];
        if(texture) delete texture;
        if(texturedShader) delete texturedShader;
    }
};



Scene* gScene = 0;

void onReshape(int winWidth0, int winHeight0){
}

// initialization, create an OpenGL context
void onInitialization()
{
    glViewport(0, 0, windowWidth, windowHeight);
    
    for(int i = 0; i < 256; i++){
        keyboardState[i] = false;
    }
    
    gScene = new Scene();
    gScene->Initialize();
}

void onExit()
{
    delete gScene;
    printf("exit");
}

// window has become invalid: redraw
void onDisplay()
{
    glClearColor(0, 0, 0, 0); // background color
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
    
    gScene->Draw();
    
    glutSwapBuffers(); // exchange the two buffers
}

//Animation
void onIdle() {
    // time elapsed since program started, in seconds
    double t = glutGet(GLUT_ELAPSED_TIME) * 0.001;
    //printf("%f", t);
    // variable to remember last time idle was called
    static double lastTime = 0.0;
    // time difference between calls: time step
    double dt = t - lastTime;
    // store time
    lastTime = t;
    
    //shooting cool down logic
    if(shootCoolDownBool==true){
        shootCoolDownTimer-=dt;
    }
    if(shootCoolDownTimer<0){
        shootCoolDownBool = false;
    }
   
    camera.move(gScene->getAvatarPosition());
    gScene->moveObjects(dt,t);
    gScene->spawnAsteroids();
    gScene->spawnEnemies();
    gScene->gameOver();
    gScene->restartGame();
    gScene->Interact();
    gScene->checkForDeath(dt);
    gScene->updateScore();
 
    glutPostRedisplay();
}

bool spacePressed = false;

void onKeyboard(unsigned char key, int i, int j){
    
    keyboardState[key] = true;
    if(keyboardState[TAB]){
        curSelectedObject++;
    }
    
    if(!spacePressed && keyboardState[SPACE]){
        gScene->spawnProjectile();
        spacePressed = true;
    }
}

void onKeyboardUp(unsigned char key, int i, int j){
    keyboardState[key] = false;
    if(key == ' '){
        spacePressed = false;
    }
}


int main(int argc, char * argv[])
{
    glutInit(&argc, argv);
#if !defined(__APPLE__)
    glutInitContextVersion(majorVersion, minorVersion);
#endif
    glutInitWindowSize(windowWidth, windowHeight);     // application window is initially of resolution 512x512
    glutInitWindowPosition(50, 50);            // relative location of the application window
#if defined(__APPLE__)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_2_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
    glutCreateWindow("Triangle Rendering");
    
#if !defined(__APPLE__)
    glewExperimental = true;
    glewInit();
#endif
    printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
    printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
    printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
    glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
    glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
    printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
    printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
    
    onInitialization();
    
    glutDisplayFunc(onDisplay); // register event handlers
    glutKeyboardFunc(onKeyboard);
    glutKeyboardUpFunc(onKeyboardUp);
    glutIdleFunc(onIdle);
    glutReshapeFunc(onReshape);
    
    glutMainLoop();
    onExit();
    return 1;
};

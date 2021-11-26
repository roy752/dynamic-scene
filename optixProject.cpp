#define STB_IMAGE_IMPLEMENTATION
#include <glad/glad.h>  // this and tiny_obj_loader.h Needs to be included before gl_interop
#include "loadObj.h"
#include "stb/stb_image.h"
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include "Freeimage/FreeImage.h"
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <sampleConfig.h>

#include <sutil/Aabb.h>
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Camera.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>
#include <sutil/Matrix.h>
#include <sutil/Trackball.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>
#include <optix_stack_size.h>

#include <GLFW/glfw3.h>
#include "JSONFileManager.h"


#include "optixProject.h"
#include "vertices.h"
#include "motionHelper.hpp"

#include <cstdlib>
#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <set>
#include<unordered_map>

//////////////////////////////////////////////////////////////
//        run parameter predefines
//        
//        USAGE:
// 
//        MODEL
//                      BEN
//						SPONZA
//						
//        BUILD_OPTION
//                      WW: WHOLE_LOAD_WHOLE_BUILD 
//                      SW: SEPERATE_LOAD_WHOLE_BUILD
//                      SS: SEPERATE_LOAD_SEPERATE_BUILD (미구현)
//        
//////////////////////////////////////////////////////////////

#define MODEL BEN
#define BUILD_OPTION WW

//////////////////////////////////////////////////////////////
//        GLFW callback variables						      

bool resize_dirty = false;
bool minimized    = false;



// Camera state
bool             camera_changed = true;
sutil::Camera    camera;
sutil::Trackball trackball;



// Mouse state
int32_t mouse_button = -1;

//        GLFW callback variables end
//////////////////////////////////////////////////////////////








//////////////////////////////////////////////////////////////                                                          
//         user-define global variables                                                                                       

enum MODEL_SELECT{BEN, SPONZA};
enum BUILD_OPTION_SELECT{WW,SW,SS};
enum ROTATE_DIRECTION{CW=-1, CCW=1};
enum DIRECTION{PLUS_X, MINUS_Z, MINUS_X, PLUS_Z};

int32_t width = 1280;
int32_t height = 1440;
int32_t maxTraceDepth = 3; //물체 반사를 몇번까지 튕길것인가? (실제 shader에서 사용하는 변수)
int32_t traceDepthLimit = 5; //물체를 최대 몇번까지 튕길 것으로 정할 것인가? (pipeline에 들어가는 변수, 실제 shader에서 optixTrace의 recursion 횟수가 이 limit을 넘어갈 수 없음)
int frameCount = 0;
int frame = 1;
Params params;
Params* d_params;

float3 eye = { 1.0f, 1.0f, 0.0f };
float3 dir = { -0.01f, -0.01f, 0.0f };
float3 up = { 0.0f, 1.0f, 0.0f };
float fovy=94.0f;
std::vector<BasicLight> lights;

std::chrono::duration<double> state_update_time(0.0);
std::chrono::duration<double> render_time(0.0);
std::chrono::duration<double> display_time(0.0);
std::chrono::steady_clock::time_point frame_change_time;

float fnear=1.0f;


sutil::Matrix4x4 projectionMatrix = 
{ 0.999391f, 0.000000f, 0.000000f, 0.000000f,
0.000000f, 0.930073f, 0.000000f, 0.000000f,
0.034900f, -0.069927f, -1.000100f, -1.000000f,
0.000000f, 0.000000f, -1.000100f, 0.000000f };
                                          
//         user-define global variables end                                                                                   
//////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////                                                          
//         GLFW Callback functions                                                                                             


static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);

	if (action == GLFW_PRESS)
	{
		mouse_button = button;
		trackball.startTracking(static_cast<int>(xpos), static_cast<int>(ypos));
	}
	else
	{
		mouse_button = -1;
	}
}
static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
	if (mouse_button == GLFW_MOUSE_BUTTON_LEFT)
	{
		trackball.setViewMode(sutil::Trackball::LookAtFixed);
		trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), width, height);
		camera_changed = true;
	}
	else if (mouse_button == GLFW_MOUSE_BUTTON_RIGHT)
	{
		trackball.setViewMode(sutil::Trackball::EyeFixed);
		trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), width, height);
		camera_changed = true;
	}
}
static void windowSizeCallback(GLFWwindow* window, int32_t res_x, int32_t res_y)
{
	// Keep rendering at the current resolution when the window is minimized.
	if (minimized)
		return;

	// Output dimensions must be at least 1 in both x and y.
	sutil::ensureMinimumSize(res_x, res_y);

	//width = res_x;
	//height = res_y;
	camera_changed = true;
	resize_dirty = true;
}
static void windowIconifyCallback(GLFWwindow* window, int32_t iconified)
{
	minimized = (iconified > 0);
}
static void keyCallback(GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/)
{
	if (action == GLFW_PRESS)
	{
		if (key == GLFW_KEY_Q ||
			key == GLFW_KEY_ESCAPE)
		{
			glfwSetWindowShouldClose(window, true);
		}
		if (key == GLFW_KEY_1) params.maxTraceDepth++;
		if (key == GLFW_KEY_2) params.maxTraceDepth--;
		
	}
	else if (key == GLFW_KEY_G)
	{
		// toggle UI draw
	}
}
static void scrollCallback(GLFWwindow* window, double xscroll, double yscroll)
{
	if (trackball.wheelEvent((int)yscroll))
		camera_changed = true;
}

                                                  
//         GLFW Callback functions end                                                                                         
//////////////////////////////////////////////////////////////



//////////////////////////////////////////////////////////////                                                          
//         user-define functions                               


template <typename T>
struct SbtRecord
{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	T data;
};
typedef SbtRecord<RayGenData>     RayGenSbtRecord;
typedef SbtRecord<MissData>       MissSbtRecord;
typedef SbtRecord<HitGroupData>   HitGroupSbtRecord;

class IndexHash
{
public:
	std::size_t operator()(const tinyobj::index_t& k) const
	{
		return std::hash<int>()(k.vertex_index) ^ ((std::hash<int>()(k.normal_index) << 1) >> 1) ^ (std::hash<int>()(k.texcoord_index) << 1);
	}
};
class CudaBuffer
{
public:
	CudaBuffer(size_t count = 0) { alloc(count); }
	~CudaBuffer() { free(); }
	
	void alloc(size_t count)
	{
		free();
		m_allocCount = m_count = count;
		if (m_count)
		{
			CUDA_CHECK(cudaMalloc(&m_ptr, m_allocCount));
		}
	}
	void allocIfRequired(size_t count)
	{
		if (count <= m_allocCount)
		{
			m_count = count;
			return;
		}
		alloc(count);
	}
	
	CUdeviceptr get() const { return reinterpret_cast<CUdeviceptr>(m_ptr); }
	CUdeviceptr get(size_t index) const { return ((CUdeviceptr)m_ptr + index); }
	void set(CUdeviceptr nowPtr, int sizes) { m_ptr = (void*)nowPtr, m_count = m_allocCount = sizes; }
	void        free()
	{
		m_count = 0;
		m_allocCount = 0;
		CUDA_CHECK(cudaFree(m_ptr));
		m_ptr = nullptr;
	}
	CUdeviceptr release()
	{
		m_count = 0;
		m_allocCount = 0;
		CUdeviceptr current = reinterpret_cast<CUdeviceptr>(m_ptr);
		m_ptr = nullptr;
		return current;
	}
	template<typename T>
	void upload(const T* data)
	{
		CUDA_CHECK(cudaMemcpy(m_ptr, data, m_count, cudaMemcpyHostToDevice));
	}

	template<typename T>
	void alloc_and_upload(const std::vector<T>& vt)
	{
		alloc(vt.size() * sizeof(T));
		upload((const T*)vt.data());
	}
	template<typename T>
	void alloc_and_upload(const T* data)
	{
		alloc(sizeof(T));
		upload(data);
	}

	template<typename T>
	void download(T* data, size_t count) const
	{
		assert(count <= m_count);
		CUDA_CHECK(cudaMemcpy(data, m_ptr, count, cudaMemcpyDeviceToHost));
	}
	template<typename T>
	void download_and_free(T* data, size_t count)
	{
		download(data, count);
		free();
	}
	template<typename T>
	void downloadSub(size_t count, size_t offset, T* data) const
	{
		assert(count + offset < m_allocCount);
		CUDA_CHECK(cudaMemcpy(data, m_ptr + offset, count * sizeof(T), cudaMemcpyDeviceToHost));
	}
	size_t sizeInBytes() const { return m_count; }
	size_t reservedCount() const { return m_allocCount; }
	
private:
	size_t m_count = 0;
	size_t m_allocCount = 0;
	void* m_ptr = nullptr;
};
void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
	std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
		<< message << "\n";
}

class Scene
{
public:

	
	struct TriangleMesh 
	{
		std::vector<float3> vertex;
		std::vector<float3> normal;
		std::vector<float2> texcoord;
		std::vector<int3> vertexIndex;
		std::vector<int3> texcoordIndex;
		std::vector<int3> normalIndex;
		std::vector<int> materialID;
		OptixTraversableHandle gas_handle;
	};
	struct Texture {
		~Texture()
		{
			if (pixel) delete[] pixel;
		}

		uint32_t* pixel{ nullptr };
		int2      resolution{ -1 };
	};

	OptixPipeline				   pipeline()				  const { return m_pipeline; }
	const OptixShaderBindingTable* sbt(int idx)				  const { return &(m_sbt[idx]); }
	OptixTraversableHandle		   traversableHandle(int idx) const	{ return m_meshes[idx].gas_handle; }
	int							   numOfMesh()				  const{ return m_meshes.size(); }

	void loadSceneSeperateMesh(const std::string& filePath)
	{
		std::cout << "...loading obj..." << std::endl;
		LoadObj obj;
		obj.loadObj(filePath);

		auto& attrib = obj.getAttrib();
		auto& shapes = obj.getShapes();
		auto& materials = obj.getMaterials();

		TriangleMesh m_mesh;

		int numOfShape = (int)shapes.size();

		int numOfVertex = attrib.vertices.size() / 3;
		int numOfNormal = attrib.normals.size() / 3;
		int numOfTexcoord = attrib.texcoords.size() / 2;
		std::cout << numOfVertex << ' ' << numOfNormal << ' ' << numOfTexcoord << std::endl;
		const float3* vertices = (float3*)(attrib.vertices.data());
		const float3* normal = (float3*)(attrib.normals.data());
		const float2* texcoords = (float2*)(attrib.texcoords.data());
		m_mesh.vertex.insert(m_mesh.vertex.end(), vertices, vertices + numOfVertex);
		m_mesh.normal.insert(m_mesh.normal.end(), normal, normal + numOfNormal);
		m_mesh.texcoord.insert(m_mesh.texcoord.end(), texcoords, texcoords + numOfTexcoord);

		for (int shapeID = 0; shapeID < numOfShape; shapeID++)
		{
			tinyobj::shape_t& shape = shapes[shapeID];
			auto nowVertexIndex = (int3*)(shape.mesh.vertex_indices.data());
			auto nowNormalIndex = (int3*)(shape.mesh.normal_indices.data());
			auto nowTexcoordIndex = (int3*)(shape.mesh.texcoord_indices.data());

			assert(shape.mesh.vertex_indices.size() == shape.mesh.normal_indices.size() && shape.mesh.normal_indices.size() == shape.mesh.texcoord_indices.size());

			if (shape.mesh.vertex_indices.empty()) continue;

			int numOfPrimitives = shape.mesh.material_ids.size();
			m_mesh.vertexIndex.insert(m_mesh.vertexIndex.end(), nowVertexIndex, nowVertexIndex + numOfPrimitives);
			m_mesh.normalIndex.insert(m_mesh.normalIndex.end(), nowNormalIndex, nowNormalIndex + numOfPrimitives);
			m_mesh.texcoordIndex.insert(m_mesh.texcoordIndex.end(), nowTexcoordIndex, nowTexcoordIndex + numOfPrimitives);
			m_mesh.materialID.insert(m_mesh.materialID.end(), shape.mesh.material_ids.begin(), shape.mesh.material_ids.end());
		}

		

		if (!m_loadTextureFlag)
		{
			std::vector<Texture*> m_textures;

			std::map<std::string, int>      knownTextures;
			std::string baseDir = filePath.substr(0, filePath.find_last_of('/'));


			int numOfMaterial = materials.size();
			m_materials.resize(numOfMaterial);
			for (int materialID = 0; materialID < numOfMaterial; ++materialID)
			{
				m_materials[materialID].ambient = { materials[materialID].ambient[0],materials[materialID].ambient[1],materials[materialID].ambient[2] };
				m_materials[materialID].diffuse = { materials[materialID].diffuse[0],materials[materialID].diffuse[1],materials[materialID].diffuse[2] };
				m_materials[materialID].specular = { materials[materialID].specular[0],materials[materialID].specular[1],materials[materialID].specular[2] };
				m_materials[materialID].transmittance = { materials[materialID].transmittance[0],materials[materialID].transmittance[1],materials[materialID].transmittance[2] };
				m_materials[materialID].emission = { materials[materialID].emission[0],materials[materialID].emission[1],materials[materialID].emission[2] };
				m_materials[materialID].shininess = materials[materialID].shininess;
				m_materials[materialID].ior = materials[materialID].ior;
				m_materials[materialID].dissolve = 1.0f - materials[materialID].dissolve; //dissolve랑 transperency랑 반대.
				m_materials[materialID].illum = materials[materialID].illum;
				m_materials[materialID].metallic = materials[materialID].metallic;

				m_materials[materialID].ambientTextureID = loadTextureppm(
					knownTextures,
					materials[materialID].ambient_texname,
					baseDir, m_textures);

				m_materials[materialID].diffuseTextureID = loadTextureppm(
					knownTextures,
					materials[materialID].diffuse_texname,
					baseDir, m_textures);

				m_materials[materialID].specularTextureID = loadTextureppm(
					knownTextures,
					materials[materialID].specular_texname,
					baseDir, m_textures);

				m_materials[materialID].specularHighlightTextureID = loadTextureppm(
					knownTextures,
					materials[materialID].specular_highlight_texname,
					baseDir, m_textures);

				m_materials[materialID].bumpTextureID = loadTextureppm(
					knownTextures,
					materials[materialID].bump_texname,
					baseDir, m_textures);

				m_materials[materialID].displacementTextureID = loadTextureppm(
					knownTextures,
					materials[materialID].displacement_texname,
					baseDir, m_textures);

				m_materials[materialID].alphaTextureID = loadTextureppm(
					knownTextures,
					materials[materialID].alpha_texname,
					baseDir, m_textures);

				m_materials[materialID].reflectionTextureID = loadTextureppm(
					knownTextures,
					materials[materialID].reflection_texname,
					baseDir, m_textures);
			}

			std::cout << "...creating textures..." << std::endl;
			createTextures(m_textures);
			std::cout << "create textures success." << std::endl;
			for (auto texture : m_textures) free(texture);
		}
		m_loadTextureFlag = true;
		m_meshes.push_back(m_mesh);

		std::cout << "load obj success." << std::endl;
	}
	void createContext()
	{
		std::cout << "...creating OptiX context..." << '\n';
		CUDA_CHECK(cudaFree(nullptr));

		CUcontext cuCtx = nullptr;
		OPTIX_CHECK(optixInit());
		OptixDeviceContextOptions options = {};
		options.logCallbackFunction = &context_log_cb;
		options.logCallbackLevel = 4;
		OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &m_context));
		std::cout << "create OptiX context success." << '\n';
		
	}
	virtual void buildAccel(OptixDeviceContext &parameterContext)
	{
		std::cout << "...building single mesh single gas..." << std::endl;

		int numOfMesh = m_meshes.size();
		vertexBuffer.resize(numOfMesh);
		vertexIndexBuffer.resize(numOfMesh);
		normalBuffer.resize(numOfMesh);
		normalIndexBuffer.resize(numOfMesh);
		texcoordBuffer.resize(numOfMesh);
		texcoordIndexBuffer.resize(numOfMesh);
		materialIdBuffer.resize(numOfMesh);
		materialBuffer.alloc_and_upload(m_materials);
		
		if (!m_textures.empty()) textureBuffer.alloc_and_upload(m_textures);

		for (int meshID = 0; meshID < numOfMesh; ++meshID)
		{
			vertexBuffer[meshID].alloc_and_upload(m_meshes[meshID].vertex);
			vertexIndexBuffer[meshID].alloc_and_upload(m_meshes[meshID].vertexIndex);
			if (m_meshes[meshID].normal.size())
			{
				normalBuffer[meshID].alloc_and_upload(m_meshes[meshID].normal);
				normalIndexBuffer[meshID].alloc_and_upload(m_meshes[meshID].normalIndex);
			}

			if (m_meshes[meshID].texcoord.size())
			{
				texcoordBuffer[meshID].alloc_and_upload(m_meshes[meshID].texcoord);
				texcoordIndexBuffer[meshID].alloc_and_upload(m_meshes[meshID].texcoordIndex);
			}
			materialIdBuffer[meshID].alloc_and_upload(m_meshes[meshID].materialID);

			CUdeviceptr vertexBufferPointer = vertexBuffer[meshID].get();
			uint32_t triangleInputFlags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
			OptixBuildInput triangleInput = {};

			triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

			triangleInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
			triangleInput.triangleArray.vertexStrideInBytes = sizeof(float3);
			triangleInput.triangleArray.numVertices = (int)m_meshes[meshID].vertex.size();
			triangleInput.triangleArray.vertexBuffers = &vertexBufferPointer;

			triangleInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
			triangleInput.triangleArray.indexStrideInBytes = sizeof(int3);
			triangleInput.triangleArray.numIndexTriplets = (int)m_meshes[meshID].vertexIndex.size();
			triangleInput.triangleArray.indexBuffer = vertexIndexBuffer[meshID].get();

			triangleInput.triangleArray.flags = triangleInputFlags;
			triangleInput.triangleArray.numSbtRecords = 1;

			OptixAccelBuildOptions accelOptions = {};

			accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE
				| OPTIX_BUILD_FLAG_ALLOW_COMPACTION
				| OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;

			accelOptions.motionOptions.numKeys = 1;
			accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

			//buildInput 을 바탕으로 GAS build에 필요한 buffer 크기 계산
			OptixAccelBufferSizes gasBufferSizes;
			OPTIX_CHECK(optixAccelComputeMemoryUsage(
				parameterContext,
				&accelOptions,
				&triangleInput,
				1,				// num_build_inputs
				&gasBufferSizes
			));


			// GAS build 동안 compactedOutputBuffer 계산해주는 속성 추가
			CudaBuffer compactedSizeBuffer;
			compactedSizeBuffer.alloc(sizeof(uint64_t));

			OptixAccelEmitDesc emitDesc;
			emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
			emitDesc.result = compactedSizeBuffer.get();

			// tempBuffer,outputBuffer 준비
			CudaBuffer tempBuffer;
			tempBuffer.alloc(gasBufferSizes.tempSizeInBytes);

			CudaBuffer outputBuffer;
			outputBuffer.alloc(gasBufferSizes.outputSizeInBytes);


			// 메인 스테이지. AccelBuild() 실행하여 GAS 빌드
			OPTIX_CHECK(optixAccelBuild(
				parameterContext,
				0,
				&accelOptions,
				&triangleInput,
				1,
				tempBuffer.get(),
				tempBuffer.sizeInBytes(),
				outputBuffer.get(),
				outputBuffer.sizeInBytes(),
				&m_meshes[meshID].gas_handle,
				&emitDesc,
				1
			));
			CUDA_SYNC_CHECK();


			// compaction 가능한지 체크
			uint64_t compactedSize;
			compactedSizeBuffer.download_and_free(&compactedSize, sizeof(uint64_t));


			if (compactedSize < gasBufferSizes.outputSizeInBytes) // 압축을 할 수 있다면(이득이 있다면)
			{
				CudaBuffer compactedBuffer;
				compactedBuffer.alloc(compactedSize);

				OPTIX_CHECK(optixAccelCompact(
					parameterContext,
					0,
					m_meshes[meshID].gas_handle,
					compactedBuffer.get(),
					compactedSize,
					&m_meshes[meshID].gas_handle
				));

				CUDA_SYNC_CHECK();

				compactedBuffer.release();
			}
			else //압축이 필요없는 경우: 최종 outputBuffer는 build의 outputBuffer 그대로
			{
				outputBuffer.release();
			}
		}
		std::cout << "build gas success." << std::endl;
	}
	void buildSingleAccel(OptixDeviceContext& parameterContext)
	{
		//std::cout << "...building single mesh single gas..." << std::endl;

		if (m_outputBuffer.sizeInBytes()) m_outputBuffer.free();

		m_matrixFrameCount++;
		m_matrixFrameCount %= m_matrixFrame;

		int numOfMesh = m_meshes.size();
		if (vertexBuffer.size() != numOfMesh)
		{
			vertexBuffer.resize(numOfMesh);
			vertexIndexBuffer.resize(numOfMesh);
			normalBuffer.resize(numOfMesh);
			normalIndexBuffer.resize(numOfMesh);
			texcoordBuffer.resize(numOfMesh);
			texcoordIndexBuffer.resize(numOfMesh);
			materialIdBuffer.resize(numOfMesh);
			materialBuffer.alloc_and_upload(m_materials);

			if (!m_textures.empty()) textureBuffer.alloc_and_upload(m_textures);
		}

		int meshID = frameCount;
		if (vertexBuffer[meshID].sizeInBytes())
		{
			vertexBuffer[meshID].free();
			vertexIndexBuffer[meshID].free();
			normalBuffer[meshID].free();
			normalIndexBuffer[meshID].free();
			texcoordBuffer[meshID].free();
			texcoordIndexBuffer[meshID].free();
			materialIdBuffer[meshID].free();
		}
		
		vertexBuffer[meshID].alloc_and_upload(m_meshes[meshID].vertex);
		vertexIndexBuffer[meshID].alloc_and_upload(m_meshes[meshID].vertexIndex);
		if (m_meshes[meshID].normal.size())
		{
			normalBuffer[meshID].alloc_and_upload(m_meshes[meshID].normal);
			normalIndexBuffer[meshID].alloc_and_upload(m_meshes[meshID].normalIndex);
		}

		if (m_meshes[meshID].texcoord.size())
		{
			texcoordBuffer[meshID].alloc_and_upload(m_meshes[meshID].texcoord);
			texcoordIndexBuffer[meshID].alloc_and_upload(m_meshes[meshID].texcoordIndex);
		}
		materialIdBuffer[meshID].alloc_and_upload(m_meshes[meshID].materialID);

		CUdeviceptr vertexBufferPointer = vertexBuffer[meshID].get();
		uint32_t triangleInputFlags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
		OptixBuildInput triangleInput = {};

		triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

		triangleInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
		triangleInput.triangleArray.vertexStrideInBytes = sizeof(float3);
		triangleInput.triangleArray.numVertices = (int)m_meshes[meshID].vertex.size();
		triangleInput.triangleArray.vertexBuffers = &vertexBufferPointer;

		triangleInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
		triangleInput.triangleArray.indexStrideInBytes = sizeof(int3);
		triangleInput.triangleArray.numIndexTriplets = (int)m_meshes[meshID].vertexIndex.size();
		triangleInput.triangleArray.indexBuffer = vertexIndexBuffer[meshID].get();

		triangleInput.triangleArray.flags = triangleInputFlags;
		triangleInput.triangleArray.numSbtRecords = 1;

		OptixAccelBuildOptions accelOptions = {};

		accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE
			| OPTIX_BUILD_FLAG_ALLOW_COMPACTION
			| OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;

		accelOptions.motionOptions.numKeys = 1;
		accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

		//buildInput 을 바탕으로 GAS build에 필요한 buffer 크기 계산
		OptixAccelBufferSizes gasBufferSizes;
		OPTIX_CHECK(optixAccelComputeMemoryUsage(
			parameterContext,
			&accelOptions,
			&triangleInput,
			1,				// num_build_inputs
			&gasBufferSizes
		));


		// GAS build 동안 compactedOutputBuffer 계산해주는 속성 추가
		CudaBuffer compactedSizeBuffer;
		compactedSizeBuffer.alloc(sizeof(uint64_t));

		OptixAccelEmitDesc emitDesc;
		emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
		emitDesc.result = compactedSizeBuffer.get();

		// tempBuffer,outputBuffer 준비
		CudaBuffer tempBuffer;
		tempBuffer.alloc(gasBufferSizes.tempSizeInBytes);

		CudaBuffer outputBuffer;
		outputBuffer.alloc(gasBufferSizes.outputSizeInBytes);


		// 메인 스테이지. AccelBuild() 실행하여 GAS 빌드
		OPTIX_CHECK(optixAccelBuild(
			parameterContext,
			0,
			&accelOptions,
			&triangleInput,
			1,
			tempBuffer.get(),
			tempBuffer.sizeInBytes(),
			outputBuffer.get(),
			outputBuffer.sizeInBytes(),
			&m_meshes[meshID].gas_handle,
			&emitDesc,
			1
		));
		CUDA_SYNC_CHECK();


		// compaction 가능한지 체크
		uint64_t compactedSize;
		compactedSizeBuffer.download_and_free(&compactedSize, sizeof(uint64_t));


		if (compactedSize < gasBufferSizes.outputSizeInBytes) // 압축을 할 수 있다면(이득이 있다면)
		{
			CudaBuffer compactedBuffer;
			compactedBuffer.alloc(compactedSize);

			OPTIX_CHECK(optixAccelCompact(
				parameterContext,
				0,
				m_meshes[meshID].gas_handle,
				compactedBuffer.get(),
				compactedSize,
				&m_meshes[meshID].gas_handle
			));

			CUDA_SYNC_CHECK();
			int sizes = compactedBuffer.sizeInBytes();
			m_outputBuffer.set(compactedBuffer.release(),sizes);
		}
		else //압축이 필요없는 경우: 최종 outputBuffer는 build의 outputBuffer 그대로
		{
			int sizes = outputBuffer.sizeInBytes();
			m_outputBuffer.set(outputBuffer.release(),sizes);
		}
		
		//std::cout << "build gas success." << std::endl;
	}
	/*
	void buildNaiveSeperateAccel()
	{
		std::cout << "...building naive ias..." << std::endl;
		const int numMeshes = (int)m_meshes.size();
		vertexBuffer.resize(numMeshes);
		normalBuffer.resize(numMeshes);
		texcoordBuffer.resize(numMeshes);
		indexBuffer.resize(numMeshes);

		

		OptixAccelBuildOptions accelOptions = {};
		accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE
			| OPTIX_BUILD_FLAG_ALLOW_COMPACTION
			;
		accelOptions.motionOptions.numKeys = 0;
		accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;


		// ==================================================================
		// triangle inputs
		// ==================================================================
		
		std::vector<CUdeviceptr> d_vertices(numMeshes);
		std::vector<CUdeviceptr> d_indices(numMeshes);
		
	   
		
		for (int meshID = 0; meshID < numMeshes; meshID++)
		{
			
			

			OptixBuildInput triangleInput;
			memset(&triangleInput, 0, sizeof(OptixBuildInput));
			unsigned int triangleInputFlags = 0u;
			CUDABuffer tempBuffer;
			CUDABuffer unCompactedOutputBuffer;
			CUDABuffer compactedOutputSizeBuffer;
			CUDABuffer compactedOutputBuffer;
			uint64_t compactedSize;
			memset(&tempBuffer, 0, sizeof(CUDABuffer));
			memset(&unCompactedOutputBuffer, 0, sizeof(CUDABuffer));
			memset(&compactedOutputSizeBuffer, 0, sizeof(CUDABuffer));
			memset(&compactedOutputBuffer, 0, sizeof(CUDABuffer));

			compactedOutputSizeBuffer.alloc(sizeof(uint64_t));

			OptixAccelEmitDesc emitDesc;
			emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
			emitDesc.result = compactedOutputSizeBuffer.d_pointer();

			// upload the model to the device: the builder
			TriangleSeperateMesh& mesh = *m_meshes[meshID];

			std::cout <<"now triangle number is :" <<mesh.index.size() << std::endl;
			int mx=INT_MIN, mn = INT_MAX;
			for (auto now : mesh.index) mx=max(max(mx, now.x),max(now.y,now.z)), mn=min(min(mn,now.x),min(now.y,now.z));
			std::cout << "min max is : " << mx << ' ' << mn << std::endl;
			
			std::cout << "now vertex number is :"<<mesh.vertex.size() << std::endl;
			//for (auto now : mesh.vertex) std::cout << "now vertex is : " << now.x << " " << now.y << " " << now.z << std::endl;
			
			
			vertexBuffer[meshID].alloc_and_upload(mesh.vertex);
		   
			indexBuffer[meshID].alloc_and_upload(mesh.index);
			if (!mesh.normal.empty())
				normalBuffer[meshID].alloc_and_upload(mesh.normal);
			if (!mesh.texcoord.empty())
				texcoordBuffer[meshID].alloc_and_upload(mesh.texcoord);

			triangleInput = {};
			triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

			// create local variables, because we need a *pointer* to the
			// device pointers
			CUdeviceptr vertexPointer = vertexBuffer[meshID].d_pointer();
			d_indices[meshID] = indexBuffer[meshID].d_pointer();

			

			triangleInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
			triangleInput.triangleArray.vertexStrideInBytes = sizeof(float3);
			triangleInput.triangleArray.numVertices = (int)(mesh.vertex.size());
			triangleInput.triangleArray.vertexBuffers = &vertexPointer;

			triangleInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
			triangleInput.triangleArray.indexStrideInBytes = sizeof(int3);
			triangleInput.triangleArray.numIndexTriplets = (int)(mesh.index.size());
			triangleInput.triangleArray.indexBuffer = indexBuffer[meshID].d_pointer();

			// in this example we have one SBT entry, and no per-primitive
			// materials:
			triangleInput.triangleArray.flags = &triangleInputFlags;
			triangleInput.triangleArray.numSbtRecords = 1;
			triangleInput.triangleArray.sbtIndexOffsetBuffer = 0;
			triangleInput.triangleArray.sbtIndexOffsetSizeInBytes = 0;
			triangleInput.triangleArray.sbtIndexOffsetStrideInBytes = 0;

			OptixAccelBufferSizes gasBufferSize;
			memset(&gasBufferSize, 0, sizeof(OptixAccelBufferSizes));
		   
			OPTIX_CHECK(optixAccelComputeMemoryUsage
			(m_context,
				&accelOptions,
				&triangleInput,
				1,  // num_build_inputs
				&gasBufferSize
			));
		   

			
			tempBuffer.alloc(gasBufferSize.tempSizeInBytes+ 10000000);

			unCompactedOutputBuffer.alloc(gasBufferSize.outputSizeInBytes+10000000);


			


			std::cout << meshID << "번 메쉬 빌드 중. 크기는 " << gasBufferSize.outputSizeInBytes << std::endl;

			std::cout << "tempBuffer 의 크기는 " << tempBuffer.sizeInBytes<<std::endl;
			std::cout << "tempBuffer의 포인터는 " << tempBuffer.d_pointer() << std::endl;
			std::cout << "outputBuffer의 크기는 " << unCompactedOutputBuffer.sizeInBytes << std::endl;
			std::cout << "outputBuffer의 포인터는" << unCompactedOutputBuffer.d_pointer() << std::endl;

			OPTIX_CHECK(optixAccelBuild(m_context,
				0,
				&accelOptions,
				&triangleInput,
				1,
				tempBuffer.d_pointer(),
				tempBuffer.sizeInBytes+ 10000000,

				unCompactedOutputBuffer.d_pointer(),
				unCompactedOutputBuffer.sizeInBytes+ 10000000,

				&(mesh.gas_handle),

				&emitDesc, 1
			));
			CUDA_SYNC_CHECK();


			
			compactedOutputSizeBuffer.download(&compactedSize, 1);

			

			mesh.compactedOutputBuffer.alloc(compactedSize);
			OPTIX_CHECK(optixAccelCompact(m_context,
				0,
				mesh.gas_handle,
				mesh.compactedOutputBuffer.d_pointer(),
				mesh.compactedOutputBuffer.sizeInBytes,
				&(mesh.gas_handle)));
			CUDA_SYNC_CHECK();
			
			



		}
		
		/// <summary>
		/// 까지가 gas 각각 build.
		/// //////////////////////////////////////////////////////
		/// </summary>
		

		std::vector<OptixInstance> optix_instances(numMeshes);
		sutil::Matrix4x4 forIdentity;
		forIdentity = forIdentity.identity();
		

		unsigned int sbt_offset = 0;
		for (size_t i = 0; i < numMeshes; ++i)
		{
			auto&  mesh = *(m_meshes[i]);
			auto& optix_instance = optix_instances[i];
			memset(&optix_instance, 0, sizeof(OptixInstance));

			optix_instance.flags = OPTIX_INSTANCE_FLAG_NONE|OPTIX_INSTANCE_FLAG_DISABLE_TRANSFORM;
			optix_instance.instanceId = static_cast<unsigned int>(i);
			optix_instance.sbtOffset = sbt_offset;
			optix_instance.visibilityMask = 1;
			optix_instance.traversableHandle = mesh.gas_handle;

			memcpy(optix_instance.transform, forIdentity.getData(), sizeof(float) * 12);

			sbt_offset += static_cast<unsigned int>(RAY_TYPE_COUNT);  // one sbt record per GAS build input per RAY_TYPE
		}

		const size_t instances_size_in_bytes = sizeof(OptixInstance) * numMeshes;
		CUdeviceptr  d_instances;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_instances), instances_size_in_bytes));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_instances),
			optix_instances.data(),
			instances_size_in_bytes,
			cudaMemcpyHostToDevice
		));

		OptixBuildInput instance_input = {};
		instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
		instance_input.instanceArray.instances = d_instances;
		instance_input.instanceArray.numInstances = static_cast<unsigned int>(numMeshes);

		OptixAccelBufferSizes ias_buffer_sizes;
		OPTIX_CHECK(optixAccelComputeMemoryUsage(
			m_context,
			&accelOptions,
			&instance_input,
			1, // num build inputs
			&ias_buffer_sizes
		));

		

		CUdeviceptr d_temp_buffer;
		CUDA_CHECK(cudaMalloc(
			reinterpret_cast<void**>(&d_temp_buffer),
			ias_buffer_sizes.tempSizeInBytes
		));
		
		CUDA_CHECK(cudaMalloc(
			reinterpret_cast<void**>(&m_d_ias_output_buffer),
			ias_buffer_sizes.outputSizeInBytes
		));
		OPTIX_CHECK(optixAccelBuild(
			m_context,
			nullptr,                  // CUDA stream
			&accelOptions,
			&instance_input,
			1,                  // num build inputs
			d_temp_buffer,
			ias_buffer_sizes.tempSizeInBytes,
			m_d_ias_output_buffer,
			ias_buffer_sizes.outputSizeInBytes,
			&m_gas_handle,
			nullptr,            // emitted property list
			0                   // num emitted properties
		));

		
		


		// ==================================================================
		// execute build (main stage)
		// ==================================================================

	   
		// ==================================================================
		// perform compaction
		// ==================================================================
		
		

		// ==================================================================
		// aaaaaand .... clean up
		// ==================================================================
		

		std::cout << "build gas success." << std::endl;
	}
	
	void buildAdvancedSeperateAccel()
	{
	////////////////////////////////////////////////////////////////////////////////////////
	// 
	// Ploblem:
	// compacted GAS build 에 필요한 메모리 크기는 
	// 일반 GAS 를 build 하기 전에는 알 수 없다.
	// 따라서, GAS의 compaction은
	// 
	// ======================================================
	// 1.일반 GAS를 먼저 빌드하고,
	// 2.이때 구해지는 메모리 size 만큼 memory allocate
	// ======================================================
	// 
	// 와 같은 방식으로 이루어진다.
	// 이때 device-host 간 동기화 지점(synchronization point)이 생긴다. 
	// 이는 퍼포먼스를 크게 해칠 가능성이 있다.
	// 이러한 문제는 실제 build와 compaction이 매우 빠른, 작은 GAS들에게 특히 치명적이다.
	// 
	// ====================================================================================
	// 한번에 한 GAS를 build 하는 naive한 알고리즘은 다음과 같다:
	// 1. 일반 gas build에 필요한 메모리 크기를 계산한다.(computeMemoryUsage() 함수)
	// 2. build buffer에 그 크기만큼 allocate 한다.
	// 3. GAS를 build 한다.
	// 4. compacted buffer size를 계산한다.
	// 5. compacted buffer size가 build buffer size 보다 작으면(즉 compaction이 의미있을 경우)
	// compacted buffer에 그 크기만큼 allocate 한다.
	// 6. build buffer에서 compacted buffer로 compaction을 수행한다.
	// ====================================================================================
	//
	// 
	// 
	// 개선된 알고리즘의 아이디어:
	// 여러 GAS의 building과 compaction process를 묶어서(batch) 처리한다.
	// 묶어서 처리하면 host-device 간 동기화 지점을 줄일 수 있다. 
	// 이론적으로는, 동기화 지점의 개수를 GAS의 개수에서 batch의 개수만큼 떨어뜨릴 수 있다.
	// 
	// GAS의 batch 선택에 고려해야할 사항은 다음과 같다:
	// a) GAS를 batch 했을때 peak memory consumption.
	// b) compacted GAS를 포함해서, output buffer에 필요한 메모리 크기.
	// 
	// b를 고려한다면, 메모리 크기를 가능한 한 작게 유지해야한다.
	// 즉, output에 필요한 총 메모리 크기는 compacted GAS의 합과 같아야한다.
	// 따라서, compacted GAS에 필요한 메모리 크기보다 크게 buffer에 allocating 하는 것을 피해야 한다.
	// 
	// 또한 peak memory consumption이 곧 algorithm의 효율이다.
	// build에 발생하는 peak memory consumption 의 lower bound는 process의 output이다. 즉 compacted GAS의 size이다.
	// 
	//
	// 개선된 알고리즘은 compacted GAS의 size를 예측하는데, 이는 compaction ratio라는 변수에 기반한다.
	// compaction ratio는 size of compacted GAS/size of build output of GAS 이다.
	// 이 예측의 유효성은 그러므로 compaction ratio를 얼마나 잘 때려맞추냐에 달려있다.
	// 알고리즘은 fixed compaction ratio를 일단 사용하기로 한다.
	// 
	// 다른 전략을 세울 수도 있다:
	// - compaction ratio 를 update 한다. 이미 처리된 GAS에 대한 통계를 내 remaining batch를 예측한다.
	// - GAS의 type에 따라 다른 compaction ratio를 사용한다.(예를 들어, motion vs static). GAS의 type에 따라 compaction ratio가 많이 차이난다.
	// 더 나아가, compaction을 skip하게 할 수도 있다.(compaction ratio를 1.0으로 두어서)
	// 
	// 
	// GAS들의 batch를 선택하기 전, 모든 GAS들을 build size 를 기준으로 정렬한다.
	// 큰 GAS 들을 작은 GAS들 보다 먼저 처리한다. 이는 peak memory consumption이 minimal memory consumption과 최대한 가깝게 하기 위함이다. 
	// 이는 또한 batching의 이점 또한 살릴 수 있다. 큰 GAS 보다 작은 GAS들이 batching으로 더 큰 이득을 본다.
	// minimum batch size는 GAS 한개 부터다. 
	//
	//
	// 목표:
	// 총 필요한 output size(이는 또한 minimal peak memory consumption)를 계산한다.
	// GAS를 batch한다. memory bound 안에 맞아떨어지는 만큼.
	// 
	// 가정:
	// 각 GAS의 input은 device memory에 있고 이후에도 필요하다.
	// 그렇지 않으면 peak memory consumption에 영향을 준다.
	// 예를 들어, GAS build 직전에 input data를 device에 업로드하고 바로 풀어주는 행위를 하면 안된다. 
	// 
	// 나아가, peak memory consumption은 추가적으로 외부환경에 많은 영향을 받는다.
	// GAS를 build 하고도 어쨌든 큰 memory 가 필요할 것으로 예상된다면(예를 들어, 현재 device에 texture data가 있어야할 경우),
	// peak memory consumption이 결국은 커질 것이고 GAS build 는 이미 큰 메모리를 사용할 수도 있다.
	// 
	// TODO:
	// - compaction ratio 예측 또는 update.
	// - compaction 불가능한 GAS들의 처리.
	// - GAS input data upload / freeing 예쁘게 만들기.
	// - 추가적인 limit? 
	// 
	//////////////////////////////////////////////////////////////////////////



		std::cout << "...building IAS..." << std::endl;


	const int numMeshes = (int)m_meshes.size();
	vertexBuffer.resize(numMeshes);
	normalBuffer.resize(numMeshes);
	texcoordBuffer.resize(numMeshes);
	indexBuffer.resize(numMeshes);




		// 초기의 Compaction ratio.
		constexpr double initialCompactionRatio = 0.5;

		// GAS를 만들면 나중에 trace할 것이다.
		// 그때 memory consumption은 적어도 compacted GAS + some CUDA stack space 정도 될것이다.
		// 250MB 정도 추가해서 CUDA stack space requirement 와 대강 match 시킨다.
		constexpr size_t additionalAvailableMemory = 250 * 1024 * 1024;

		OptixAccelBuildOptions accel_options = {};
		accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE|OPTIX_BUILD_FLAG_ALLOW_COMPACTION|OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
		accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;


		struct GASInfo {
			OptixBuildInput buildInput;
			OptixAccelBufferSizes gas_buffer_sizes;
			TriangleSeperateMesh *mesh;
		};
		
		std::multimap<size_t, GASInfo> gases;
		size_t totalTempOutputSize = 0;

		for (size_t i = 0; i < m_meshes.size(); ++i)
		{
			TriangleSeperateMesh& mesh = *m_meshes[i];
	
			vertexBuffer[i].alloc_and_upload(mesh.vertex);
			indexBuffer[i].alloc_and_upload(mesh.index);
			if (!mesh.normal.empty())
			normalBuffer[i].alloc_and_upload(mesh.normal);
			if (!mesh.texcoord.empty())
			texcoordBuffer[i].alloc_and_upload(mesh.texcoord);
	 
	 
	 
	 
			OptixBuildInput buildInput;
			memset(&buildInput, 0, sizeof(OptixBuildInput));
			buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
			buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
			buildInput.triangleArray.vertexStrideInBytes = sizeof(float3);
			buildInput.triangleArray.numVertices = mesh.vertex.size();
			auto VertexInDeviceMemoryPointer = vertexBuffer[i].d_pointer();
			buildInput.triangleArray.vertexBuffers = &VertexInDeviceMemoryPointer;
		   
			buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
			buildInput.triangleArray.indexStrideInBytes = sizeof(int3);
			buildInput.triangleArray.numIndexTriplets = mesh.index.size();
			buildInput.triangleArray.indexBuffer = indexBuffer[i].d_pointer();
			buildInput.triangleArray.flags = 0;
			buildInput.triangleArray.numSbtRecords = 1;
			
			unsigned int buildFlag = 0;
			buildInput.triangleArray.flags = &buildFlag;


			OptixAccelBufferSizes gas_buffer_sizes;
			OPTIX_CHECK(optixAccelComputeMemoryUsage(m_context, &accel_options, &buildInput,
				1, &gas_buffer_sizes));

			totalTempOutputSize += gas_buffer_sizes.outputSizeInBytes;
			GASInfo g = { std::move(buildInput), gas_buffer_sizes, &mesh };
			gases.emplace(gas_buffer_sizes.outputSizeInBytes, g);
		}

		////////////////////////////////////////////////////////////////////////////////////////




		size_t totalTempOutputProcessedSize = 0;
		size_t usedCompactedOutputSize = 0;
		double compactionRatio = initialCompactionRatio;

		CudaBuffer<char> d_temp;
		CudaBuffer<char> d_temp_output;
		CudaBuffer<size_t> d_temp_compactedSizes;

		OptixAccelEmitDesc emitProperty = {};
		emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;



		while (!gases.empty())
		{

			//
			// compaction을 사용했을때 예상되는 총 output size이다.
			// minimum peak memory consumption을 의미하며, 실제 GAS를 빌드하기 전에는 모른다.
			// 실제 memory constraints result가 minimal peak memory consumption과 매우 가까울때만 동작한다.
			// 
			
			size_t remainingEstimatedTotalOutputSize =
				(size_t)((totalTempOutputSize - totalTempOutputProcessedSize) * compactionRatio);
			size_t availableMemPoolSize = remainingEstimatedTotalOutputSize + additionalAvailableMemory;
			// We need to fit the following things into availableMemPoolSize:
			// - temporary buffer for building a GAS (only during build, can be cleared before compaction)
			// - build output buffer of a GAS
			// - size (actual number) of a compacted GAS as output of a build
			// - compacted GAS

			//
			// availableMemPoolSize에 아래 사항을 맞춰야한다:
			//
			// - GAS building 에 필요한 temp buffer(build 때만 필요하므로 compaction 전에 비워도 된다)
			// - build output buffer
			// - compacted GAS 의 실제 개수
			// - compacted GAS



			size_t batchNGASes = 0;
			size_t batchBuildOutputRequirement = 0;
			size_t batchBuildMaxTempRequirement = 0;
			size_t batchBuildCompactedRequirement = 0;
			for (auto it = gases.rbegin(); it != gases.rend(); it++)
			{
				batchBuildOutputRequirement += it->second.gas_buffer_sizes.outputSizeInBytes; //이 메쉬를 빌드하는데 필요한 buffer size를 더한다.
				batchBuildCompactedRequirement += (size_t)(it->second.gas_buffer_sizes.outputSizeInBytes * compactionRatio); //compacted GAS 에 필요한 buffer size를 더한다.
				// roughly account for the storage of the compacted size, although that goes into a separate buffer
				batchBuildOutputRequirement += 8ull;
				// make sure that all further output pointers are 256 byte aligned
				batchBuildOutputRequirement = roundUp<size_t>(batchBuildOutputRequirement, 256ull);
				// temp buffer is shared for all builds in the batch
				batchBuildMaxTempRequirement = std::max(batchBuildMaxTempRequirement, it->second.gas_buffer_sizes.tempSizeInBytes);
				batchNGASes++;
				// 만약 지금까지 batch한 메쉬들의 build buffer size + temp buffer size + compacted buffer size 가 available 보다 크다면 탈출.
				if ((batchBuildOutputRequirement + batchBuildMaxTempRequirement + batchBuildCompactedRequirement) > availableMemPoolSize)
					break;
			}

			// d_temp may still be available from a previous batch, but is freed later if it is "too big"
			d_temp.allocIfRequired(batchBuildMaxTempRequirement);

			// trash existing buffer if it is more than 10% bigger than what we need
			// if it is roughly the same, we keep it
			if (d_temp_output.byteSize() > batchBuildOutputRequirement * 1.1)
				d_temp_output.free();
			d_temp_output.allocIfRequired(batchBuildOutputRequirement);

			// this buffer is assumed to be very small
			// trash d_temp_compactedSizes if it is at least 20MB in size and at least double the size than required for the next run
			if (d_temp_compactedSizes.reservedCount() > batchNGASes * 2 && d_temp_compactedSizes.byteSize() > 20 * 1024 * 1024)
				d_temp_compactedSizes.free();
			d_temp_compactedSizes.allocIfRequired(batchNGASes);
			std::cout << "Well.." << std::endl;

			auto it = gases.rbegin();
			for (size_t i = 0, tempOutputAlignmentOffset = 0; i < batchNGASes; ++i)
			{
				emitProperty.result = d_temp_compactedSizes.get(i);
				GASInfo& info = it->second;

				OPTIX_CHECK(optixAccelBuild(m_context, 0,   // CUDA stream
					&accel_options,
					&info.buildInput,
					1u,
					d_temp.get(),
					d_temp.byteSize(),
					d_temp_output.get(tempOutputAlignmentOffset),
					info.gas_buffer_sizes.outputSizeInBytes,
					&info.mesh->gas_handle,
					&emitProperty,  // emitted property list
					1               // num emitted properties
				));

				tempOutputAlignmentOffset += roundUp<size_t>(info.gas_buffer_sizes.outputSizeInBytes, 256ull);
				it++;
			}

			// trash d_temp if it is at least 20MB in size
			if (d_temp.byteSize() > 20 * 1024 * 1024)
				d_temp.free();

			// download all compacted sizes to allocate final output buffers for these GASes
			std::vector<size_t> h_compactedSizes(batchNGASes);
			d_temp_compactedSizes.download(h_compactedSizes.data());

			//////////////////////////////////////////////////////////////////////////
			// TODO:
			// Now we know the actual memory requirement of the compacted GASes.
			// Based on that we could shrink the batch if the compaction ratio is bad and we need to strictly fit into the/any available memory pool.
			bool canCompact = false;
			it = gases.rbegin();
			for (size_t i = 0; i < batchNGASes; ++i)
			{
				GASInfo& info = it->second;
				if (info.gas_buffer_sizes.outputSizeInBytes > h_compactedSizes[i])
				{
					canCompact = true;
					break;
				}
				it++;
			}

			// sum of size of compacted GASes
			size_t batchCompactedSize = 0;

			if (canCompact)
			{
				//////////////////////////////////////////////////////////////////////////
				// "batch allocate" the compacted buffers
				it = gases.rbegin();
				for (size_t i = 0; i < batchNGASes; ++i)
				{
					GASInfo& info = it->second;
					batchCompactedSize += h_compactedSizes[i];
					//CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&info.mesh->compactedOutputBuffer), h_compactedSizes[i]));
					info.mesh->compactedOutputBuffer.alloc(h_compactedSizes[i]);
					totalTempOutputProcessedSize += info.gas_buffer_sizes.outputSizeInBytes;
					it++;
				}

				it = gases.rbegin();
				for (size_t i = 0; i < batchNGASes; ++i)
				{
					GASInfo& info = it->second;
					OPTIX_CHECK(optixAccelCompact(m_context, 0, info.mesh->gas_handle, info.mesh->compactedOutputBuffer.d_pointer(),
						h_compactedSizes[i], &info.mesh->gas_handle));
					it++;
				}
			}
			else
			{
				it = gases.rbegin();
				for (size_t i = 0, tempOutputAlignmentOffset = 0; i < batchNGASes; ++i)
				{
					GASInfo& info = it->second;
					info.mesh->compactedOutputBuffer.d_ptr = (void*)d_temp_output.get(tempOutputAlignmentOffset);
					batchCompactedSize += h_compactedSizes[i];
					totalTempOutputProcessedSize += info.gas_buffer_sizes.outputSizeInBytes;

					tempOutputAlignmentOffset += roundUp<size_t>(info.gas_buffer_sizes.outputSizeInBytes, 256ull);
					it++;
				}
				d_temp_output.release();
			}

			usedCompactedOutputSize += batchCompactedSize;

			gases.erase(it.base(), gases.end());
		}
		sutil::Matrix4x4 forIdentity;
		forIdentity = forIdentity.identity();
		std::cout << "TADA!" << std::endl;

		const size_t num_instances = m_meshes.size();

		std::vector<OptixInstance> optix_instances(num_instances);

		unsigned int sbt_offset = 0;
		for (size_t i = 0; i < m_meshes.size(); ++i)
		{
			auto  mesh = m_meshes[i];
			auto& optix_instance = optix_instances[i];
			memset(&optix_instance, 0, sizeof(OptixInstance));

			optix_instance.flags = OPTIX_INSTANCE_FLAG_NONE;
			optix_instance.instanceId = static_cast<unsigned int>(i);
			optix_instance.sbtOffset = sbt_offset;
			optix_instance.visibilityMask = 1;
			optix_instance.traversableHandle = mesh->gas_handle;
			memcpy(optix_instance.transform, forIdentity.getData(), sizeof(float) * 12);

			sbt_offset += RAY_TYPE_COUNT;  // one sbt record per GAS build input per RAY_TYPE
		}

		const size_t instances_size_in_bytes = sizeof(OptixInstance) * num_instances;
		CUdeviceptr  d_instances;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_instances), instances_size_in_bytes));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_instances),
			optix_instances.data(),
			instances_size_in_bytes,
			cudaMemcpyHostToDevice
		));

		OptixBuildInput instance_input = {};
		instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
		instance_input.instanceArray.instances = d_instances;
		instance_input.instanceArray.numInstances = static_cast<unsigned int>(num_instances);

		

		OptixAccelBufferSizes ias_buffer_sizes;
		OPTIX_CHECK(optixAccelComputeMemoryUsage(
			m_context,
			&accel_options,
			&instance_input,
			1, // num build inputs
			&ias_buffer_sizes
		));

		CUdeviceptr d_temp_buffer;
		CUDA_CHECK(cudaMalloc(
			reinterpret_cast<void**>(&d_temp_buffer),
			ias_buffer_sizes.tempSizeInBytes
		));
		CUDA_CHECK(cudaMalloc(
			reinterpret_cast<void**>(&m_d_ias_output_buffer),
			ias_buffer_sizes.outputSizeInBytes
		));

		OPTIX_CHECK(optixAccelBuild(
			m_context,
			nullptr,                  // CUDA stream
			&accel_options,
			&instance_input,
			1,                  // num build inputs
			d_temp_buffer,
			ias_buffer_sizes.tempSizeInBytes,
			m_d_ias_output_buffer,
			ias_buffer_sizes.outputSizeInBytes,
			&m_gas_handle,
			nullptr,            // emitted property list
			0                   // num emitted properties
		));





		std::cout << "build IAS success." << std::endl;
	}
	*/
	
	virtual void createModule()
	{
		std::cout << "Creating module..." << "\n";
		OptixModuleCompileOptions module_compile_options = {};
		module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
		module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

		m_pipeline_compile_options = {};
		m_pipeline_compile_options.usesMotionBlur = false;
		m_pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
		m_pipeline_compile_options.numPayloadValues = 5;
		m_pipeline_compile_options.numAttributeValues = 3; // TODO
		m_pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE; // should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
		m_pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

		size_t      inputSize = 0;
		const char* input = sutil::getInputData("optixProject", "optixProject", "shaderWholeMesh.cu", inputSize);

		m_module = {};
		char log[2048];
		size_t sizeof_log = sizeof(log);
		OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
			m_context,
			&module_compile_options,
			&m_pipeline_compile_options,
			input,
			inputSize,
			log,
			&sizeof_log,
			&m_module
		));
		std::cout << "create module success." << "\n";
	}
	void createProgramGroups() 
	{
		std::cout << "...creating Program groups..." << std::endl;
		OptixProgramGroupOptions program_group_options = {};

		char log[2048];
		size_t sizeof_log = sizeof(log);

		//
		//Ray generation
		//

		{
			OptixProgramGroupDesc raygen_prog_group_desc = {};
			raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
			raygen_prog_group_desc.raygen.module = m_module;
			raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";


			OPTIX_CHECK_LOG(optixProgramGroupCreate(
				m_context,
				&raygen_prog_group_desc,
				1,                             // num program groups
				&program_group_options,
				log,
				&sizeof_log,
				&m_raygen_prog_group
			)
			);
		}

		//
		// Miss
		//
		{
			OptixProgramGroupDesc miss_prog_group_desc = {};
			miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
			miss_prog_group_desc.miss.module = m_module;
			miss_prog_group_desc.miss.entryFunctionName = "__miss__radiance";
			sizeof_log = sizeof(log);
			OPTIX_CHECK_LOG(optixProgramGroupCreate(
				m_context,
				&miss_prog_group_desc,
				1,   // num program groups
				&program_group_options,
				log,
				&sizeof_log,
				&m_radiance_miss_prog_group
			)
			);

			memset(&miss_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
			miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
			miss_prog_group_desc.miss.module = nullptr;
			miss_prog_group_desc.miss.entryFunctionName = nullptr;
			sizeof_log = sizeof(log);
			OPTIX_CHECK_LOG(optixProgramGroupCreate(
				m_context,
				&miss_prog_group_desc,
				1,   // num program groups
				&program_group_options,
				log,
				&sizeof_log,
				&m_occlusion_miss_prog_group
			)
			);
		}
		//
		// Hit
		//
		{
			OptixProgramGroupDesc hitgroup_prog_group_desc = {};
			hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
			hitgroup_prog_group_desc.hitgroup.moduleCH = m_module;
			hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
			sizeof_log = sizeof(log);
			OPTIX_CHECK_LOG(optixProgramGroupCreate(
				m_context,
				&hitgroup_prog_group_desc,
				1,   // num program groups
				&program_group_options,
				log,
				&sizeof_log,
				&m_radiance_hitgroup_prog_group
			)
			);

			memset(&hitgroup_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
			hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
			hitgroup_prog_group_desc.hitgroup.moduleAH = m_module;
			hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__occlusion";
			sizeof_log = sizeof(log);
			OPTIX_CHECK(optixProgramGroupCreate(
				m_context,
				&hitgroup_prog_group_desc,
				1,                             // num program groups
				&program_group_options,
				log,
				&sizeof_log,
				&m_occlusion_hitgroup_prog_group
			)
			);




		}
		std::cout << "create Program groups success." << std::endl;
	}
	void createPipeline() 
	{
		std::cout << "...creating Pipeline..." << std::endl;

		OptixProgramGroup program_groups[] =
		{
			m_raygen_prog_group,
			m_radiance_miss_prog_group,
			m_occlusion_miss_prog_group,
			m_radiance_hitgroup_prog_group,
			m_occlusion_hitgroup_prog_group
		};

		OptixPipelineLinkOptions pipeline_link_options = {};
		pipeline_link_options.maxTraceDepth = traceDepthLimit;
		pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

		char log[2048];

		size_t sizeof_log = sizeof(log);

		OPTIX_CHECK_LOG(optixPipelineCreate(
			m_context,
			&m_pipeline_compile_options,
			&pipeline_link_options,
			program_groups,
			sizeof(program_groups) / sizeof(program_groups[0]),
			log,
			&sizeof_log,
			&m_pipeline
		));

		std::cout << "create Pipeline success." << std::endl;
	}
	virtual void createSBT()
	{
		std::cout << "...building SBT records..." << std::endl;

		int numOfMesh = m_meshes.size();
		m_sbt.resize(numOfMesh);
		for (int meshID = 0; meshID < numOfMesh; ++meshID)
		{
			// ------------------------------------------------------------------
			// build raygen sbt
			// ------------------------------------------------------------------
			RayGenSbtRecord raygenSbtRecord = {};
			OPTIX_CHECK(optixSbtRecordPackHeader(m_raygen_prog_group, &raygenSbtRecord));

			CudaBuffer raygenSbt;
			raygenSbt.alloc_and_upload(&raygenSbtRecord);
			m_sbt[meshID].raygenRecord = raygenSbt.release();


			// ------------------------------------------------------------------
			// build miss sbt
			// ------------------------------------------------------------------


			std::vector<MissSbtRecord> missSbtRecord = {};

			MissSbtRecord nowMissSbtRecord;
			nowMissSbtRecord.data.bg_color = { 0.0f, 0.0f, 0.0f };

			OPTIX_CHECK(optixSbtRecordPackHeader(m_radiance_miss_prog_group, &nowMissSbtRecord));
			missSbtRecord.push_back(nowMissSbtRecord);

			OPTIX_CHECK(optixSbtRecordPackHeader(m_occlusion_miss_prog_group, &nowMissSbtRecord));
			missSbtRecord.push_back(nowMissSbtRecord);


			CudaBuffer missSbt;
			missSbt.alloc_and_upload(missSbtRecord);
			m_sbt[meshID].missRecordBase = missSbt.release();
			m_sbt[meshID].missRecordCount = RAY_TYPE_COUNT;
			m_sbt[meshID].missRecordStrideInBytes = static_cast<uint32_t>(sizeof(MissSbtRecord));

			// ------------------------------------------------------------------
			// build hitgroup records
			// ------------------------------------------------------------------

			std::vector<HitGroupSbtRecord> hitgroupSbtRecord;

			HitGroupSbtRecord nowHitgroupSbtRecord = {};

			nowHitgroupSbtRecord.data.vertex = (float3*)(vertexBuffer[meshID].get());
			nowHitgroupSbtRecord.data.vertexIndex = (int3*)(vertexIndexBuffer[meshID].get());
			if (normalBuffer[meshID].sizeInBytes())
			{
				nowHitgroupSbtRecord.data.normal = (float3*)(normalBuffer[meshID].get());
				nowHitgroupSbtRecord.data.normalIndex = (int3*)(normalIndexBuffer[meshID].get());
			}

			if (texcoordBuffer[meshID].sizeInBytes())
			{
				nowHitgroupSbtRecord.data.texcoord = (float2*)(texcoordBuffer[meshID].get());
				nowHitgroupSbtRecord.data.texcoordIndex = (int3*)(texcoordIndexBuffer[meshID].get());
			}

			if (materialIdBuffer[meshID].sizeInBytes())
			{
				nowHitgroupSbtRecord.data.materialIDs = (int*)(materialIdBuffer[meshID].get());
			}
			nowHitgroupSbtRecord.data.materials = (Material*)(materialBuffer.get());
			if (!m_textures.empty())
			{
				nowHitgroupSbtRecord.data.textures = (cudaTextureObject_t*)(textureBuffer.get());
			}
			OPTIX_CHECK(optixSbtRecordPackHeader(m_radiance_hitgroup_prog_group, &nowHitgroupSbtRecord));
			hitgroupSbtRecord.push_back(nowHitgroupSbtRecord);

			OPTIX_CHECK(optixSbtRecordPackHeader(m_occlusion_hitgroup_prog_group, &nowHitgroupSbtRecord));
			hitgroupSbtRecord.push_back(nowHitgroupSbtRecord);


			CudaBuffer hitgroupSbt;
			hitgroupSbt.alloc_and_upload(hitgroupSbtRecord);

			m_sbt[meshID].hitgroupRecordBase = hitgroupSbt.release();
			m_sbt[meshID].hitgroupRecordCount = RAY_TYPE_COUNT;

			m_sbt[meshID].hitgroupRecordStrideInBytes = static_cast<uint32_t>(sizeof(HitGroupSbtRecord));
		}
		std::cout << "build SBT records success." << std::endl;
	}
	virtual void prepareScene() 
	{
		createContext();
		buildAccel(m_context);
		createModule();
		createProgramGroups();
		createPipeline();
		createSBT();
	}

	int loadTexture
	(
		std::map<std::string, int>& knownTextures,
		const std::string& textureName,
		const std::string filePath,
		std::vector<Texture*> &m_textures
	)
	{
		if (textureName == "") return -1;

		if (knownTextures.find(textureName) != knownTextures.end()) return knownTextures[textureName]; 
		
		std::string fileName = textureName;
		for (auto& c : fileName)
			if (c == '\\') c = '/';
		fileName = filePath + "/" + fileName;

		int2 res;
		int   comp;
		
		unsigned char* image = stbi_load(fileName.c_str(),
			&res.x, &res.y, &comp, STBI_rgb_alpha);
		
		int textureID = -1;
		if (image) 
		{
			
			textureID = (int)m_textures.size();
			Texture *texture = new Texture;
			texture->resolution = res;
			texture->pixel = (uint32_t*)image;

			/* iw - actually, it seems that stbi loads the pictures
			   mirrored along the y axis - mirror them here */
			for (int y = 0; y < res.y / 2; y++) {
				uint32_t* line_y = texture->pixel + y * res.x;
				uint32_t* mirrored_y = texture->pixel + (res.y - 1 - y) * res.x;
				int mirror_y = res.y - 1 - y;
				for (int x = 0; x < res.x; x++) {
					std::swap(line_y[x], mirrored_y[x]);
				}
			}

			m_textures.push_back(texture);
			
		}
		else 
		{
			std::cout << "Could not load texture from " << textureName << "!" << std::endl;
		}
		
		knownTextures[textureName] = textureID;
		return textureID;

	}

	int loadTextureppm
	(
		std::map<std::string, int>& knownTextures,
		const std::string& textureName,
		const std::string filePath,
		std::vector<Texture*>& m_textures
	)
	{
		if (textureName == "") return -1;

		if (knownTextures.find(textureName) != knownTextures.end()) return knownTextures[textureName];

		std::string fileName = textureName;
		for (auto& c : fileName)
			if (c == '\\') c = '/';
		fileName = filePath + "/" + fileName;

		int2 res;

		int textureID = m_textures.size();
		
		FREE_IMAGE_FORMAT nowTexType = FreeImage_GetFileType(fileName.c_str(), 0);
		if (nowTexType == FIF_UNKNOWN) { std::cout << "SIBAL" << std::endl; }
		
		FIBITMAP* imagen = FreeImage_Load(nowTexType, fileName.c_str());
		if (!imagen) { std::cout << "SOME SIBAL" << std::endl; }
		
		FIBITMAP* temp2 = FreeImage_ConvertTo32Bits(imagen);
		if (!temp2) { std::cout << "Very SIBAL" << std::endl; }
		FreeImage_Unload(imagen);
		imagen = temp2;
		Texture* texture = new Texture;
		res.x = FreeImage_GetWidth(imagen);
		res.y = FreeImage_GetHeight(imagen);

		texture->resolution = res;
		
		char* tempPxl = new char[4 * res.x * res.y];
		char* tempPxl2 = (char*)FreeImage_GetBits(imagen);
		for (int i = 0; i < res.x * res.y; ++i)
		{
			tempPxl[4 * i] = tempPxl2[4 * i + 2];
			tempPxl[4 * i + 1] = tempPxl2[4 * i + 1];
			tempPxl[4 * i + 2] = tempPxl2[4 * i];
			tempPxl[4 * i + 3] = tempPxl2[4 * i + 3];
		}

		texture->pixel = (uint32_t*)tempPxl;
		m_textures.push_back(texture);

		knownTextures[textureName] = textureID;
		return textureID;

	}

	void createTextures(std::vector<Texture*> &textures)
	{
		int numTextures = (int)textures.size();
		std::vector<cudaArray_t> textureArrays;

		textureArrays.resize(numTextures);
		m_textures.resize(numTextures);

		for (int textureID = 0; textureID < numTextures; textureID++) 
		{
			auto texture = textures[textureID];

			cudaResourceDesc res_desc = {};

			cudaChannelFormatDesc channel_desc;
			int32_t width = texture->resolution.x;
			int32_t height = texture->resolution.y;
			int32_t numComponents = 4;
			int32_t pitch = width * numComponents * sizeof(uint8_t);
			channel_desc = cudaCreateChannelDesc<uchar4>();

			cudaArray_t& pixelArray = textureArrays[textureID];
			CUDA_CHECK(cudaMallocArray(&pixelArray,
				&channel_desc,
				width, height));

			CUDA_CHECK(cudaMemcpy2DToArray(pixelArray,
				/* offset */0, 0,
				texture->pixel,
				pitch, pitch, height,
				cudaMemcpyHostToDevice));

			res_desc.resType = cudaResourceTypeArray;
			res_desc.res.array.array = pixelArray;

			cudaTextureDesc tex_desc = {};
			tex_desc.addressMode[0] = cudaAddressModeWrap;
			tex_desc.addressMode[1] = cudaAddressModeWrap;
			tex_desc.filterMode = cudaFilterModeLinear;
			tex_desc.readMode = cudaReadModeNormalizedFloat;
			tex_desc.normalizedCoords = 1;
			tex_desc.maxAnisotropy = 1;
			tex_desc.maxMipmapLevelClamp = 99;
			tex_desc.minMipmapLevelClamp = 0;
			tex_desc.mipmapFilterMode = cudaFilterModePoint;
			tex_desc.borderColor[0] = 1.0f;
		   // tex_desc.sRGB = 1;

			// Create texture object
			cudaTextureObject_t cuda_tex = 0;
			CUDA_CHECK(cudaCreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr));
			m_textures[textureID] = cuda_tex;
		}
	}

	void createMatrices(ROTATE_DIRECTION rotateDir, DIRECTION startDirection, float3 startPosition, float radius, float workingLength, int numOfMatrices, float scaleFactor)
	{

		m_matrix.resize(numOfMatrices);
		m_matrixFrame = numOfMatrices;
		sutil::Matrix4x4 initialScale = sutil::Matrix4x4::scale({ scaleFactor, scaleFactor, scaleFactor });
		sutil::Matrix4x4 initialRotate = sutil::Matrix4x4::rotate(0.5 * M_PI, { 0.0f, 1.0f, 0.0f });
		sutil::Matrix4x4 initialTranslate = sutil::Matrix4x4::translate({ 0.0f, 0.0f, radius*rotateDir });
		sutil::Matrix4x4 finalRotate = sutil::Matrix4x4::rotate(startDirection * M_PI * 0.5f, { 0.0f, 1.0f, 0.0f });
		sutil::Matrix4x4 finalTranslate = sutil::Matrix4x4::translate(startPosition);
		
		sutil::Matrix4x4 step=sutil::Matrix4x4::identity();
		sutil::Matrix4x4 cornerRotate=sutil::Matrix4x4::identity();
		
		for (int i = 0; i < numOfMatrices; ++i)
		{
			if(i<numOfMatrices/4)
				step = sutil::Matrix4x4::translate({ 2.0f * i * (workingLength) / (numOfMatrices / 4) - workingLength ,0.0f, 0.0f});
			if (i >= numOfMatrices / 4 && i < 2 * numOfMatrices / 4)
				cornerRotate = sutil::Matrix4x4::rotate(((float)(i - numOfMatrices / 4) / (numOfMatrices / 4)) *rotateDir* M_PI, { 0.0f, 1.0f, 0.0f });
			if (i >= 2 * numOfMatrices / 4 && i < 3 * numOfMatrices / 4)
				step = sutil::Matrix4x4::translate({ workingLength - 2.0f * (i - 2 * numOfMatrices / 4) * (workingLength) / (numOfMatrices / 4),0.0f, 0.0f });
			if (i >= 3 * numOfMatrices / 4)
				cornerRotate = sutil::Matrix4x4::rotate(((float)(i - 2 * numOfMatrices / 4) / (numOfMatrices / 4)) * rotateDir*M_PI, { 0.0f,1.0f, 0.0f });
			m_matrix[i] = finalTranslate*finalRotate * step * cornerRotate * initialTranslate * initialRotate * initialScale;
		}
	}
	
	std::vector<TriangleMesh>					m_meshes = {};
	std::vector<cudaTextureObject_t>            m_textures = {};
	std::vector<Material>						m_materials = {};
	std::vector<sutil::Matrix4x4>				m_matrix = {};

	std::vector<CudaBuffer>						vertexBuffer;
	std::vector<CudaBuffer>						vertexIndexBuffer;
	std::vector<CudaBuffer>						normalBuffer;
	std::vector<CudaBuffer>						normalIndexBuffer;
	std::vector<CudaBuffer>						texcoordBuffer;
	std::vector<CudaBuffer>						texcoordIndexBuffer;
	std::vector<CudaBuffer>						materialIdBuffer;
	CudaBuffer									materialBuffer;
	CudaBuffer									textureBuffer;
	CudaBuffer									m_outputBuffer;

	OptixDeviceContext							m_context                            = 0;
	OptixModule									m_module                             = 0;
	OptixPipelineCompileOptions					m_pipeline_compile_options           = {};
	OptixPipeline								m_pipeline                           = 0;
	std::vector<OptixShaderBindingTable>        m_sbt                                = {};

	OptixProgramGroup							m_raygen_prog_group                  = 0;
	OptixProgramGroup							m_radiance_miss_prog_group           = 0;
	OptixProgramGroup							m_occlusion_miss_prog_group          = 0;
	OptixProgramGroup							m_radiance_hitgroup_prog_group       = 0;
	OptixProgramGroup							m_occlusion_hitgroup_prog_group      = 0;

	bool										m_loadTextureFlag						 = false;
	int											m_matrixFrame						= 0;
	int											m_matrixFrameCount					= 0;
};
class CombScene : public Scene
{
	public:
	void buildAccel(Scene& staticScene, std::vector<Scene>& dynamicScene)
	{
		//std::cout << "instancing.." << std::endl;
		
		if (m_outputBuffer.sizeInBytes()) m_outputBuffer.free();

		OptixInstance staticInstance = {};
		std::vector<OptixInstance> dynamicInstance(dynamicScene.size(),staticInstance);

		sutil::Matrix4x4 forIdentity;
		forIdentity = forIdentity.identity();


		// 정적인 scene 먼저 instancing. 이미 gas_handle까지 빌드가 되어있음.
		auto mesh = staticScene.m_meshes[0];

		staticInstance.flags = OPTIX_INSTANCE_FLAG_NONE;
		
		staticInstance.instanceId = static_cast<unsigned int>(0);
		staticInstance.sbtOffset = 0;
		
		staticInstance.visibilityMask = 1;
		staticInstance.traversableHandle = mesh.gas_handle;
		
		memcpy(staticInstance.transform, forIdentity.getData(), sizeof(float) * 12);
		//정적인 scene instancing 완료.


		// 동적 장면 instancing. 빌드부터.

		for (int i = 0; i < dynamicScene.size(); ++i)
		{
			dynamicScene[i].buildSingleAccel(m_context);
			dynamicInstance[i].flags = OPTIX_INSTANCE_FLAG_NONE;
			dynamicInstance[i].instanceId = static_cast<unsigned int>(i+1);
			dynamicInstance[i].sbtOffset = RAY_TYPE_COUNT*(i+1);
			dynamicInstance[i].visibilityMask = 1;
			dynamicInstance[i].traversableHandle = dynamicScene[i].traversableHandle(frameCount);
			memcpy(dynamicInstance[i].transform, dynamicScene[i].m_matrix[dynamicScene[i].m_matrixFrameCount].getData(), sizeof(float) * 12);
		}

		OptixAccelBuildOptions accel_options = {};
		accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
		accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

		
			std::vector<OptixInstance> tempNow;
			tempNow.push_back(staticInstance);
			tempNow.insert(tempNow.end(),dynamicInstance.begin(), dynamicInstance.end());
			

			CudaBuffer instanceBuffer;
			instanceBuffer.alloc_and_upload(tempNow);

			OptixBuildInput instance_input = {};
			instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
			instance_input.instanceArray.instances = instanceBuffer.get();
			instance_input.instanceArray.numInstances = static_cast<unsigned int>(tempNow.size());


			OptixAccelBufferSizes ias_buffer_sizes;

			OPTIX_CHECK(optixAccelComputeMemoryUsage(
				m_context,
				&accel_options,
				&instance_input,
				1, // num build inputs
				&ias_buffer_sizes
			));

			CudaBuffer tempBuffer;
			CudaBuffer outputBuffer;
			tempBuffer.alloc(ias_buffer_sizes.tempSizeInBytes);
			outputBuffer.alloc(ias_buffer_sizes.outputSizeInBytes);
			
			OPTIX_CHECK(optixAccelBuild(
				m_context,
				nullptr,                  // CUDA stream
				&accel_options,
				&instance_input,
				1,                  // num build inputs
				tempBuffer.get(),
				ias_buffer_sizes.tempSizeInBytes,
				outputBuffer.get(),
				ias_buffer_sizes.outputSizeInBytes,
				&(comb_ias_handle),
				nullptr,            // emitted property list
				0                   // num emitted properties
			));
			int sizes = outputBuffer.sizeInBytes();
			m_outputBuffer.set(outputBuffer.release(),sizes);
		
	}
	void createModule()
	{
		std::cout << "Creating module..." << "\n";
		OptixModuleCompileOptions module_compile_options = {};
		module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
		module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

		m_pipeline_compile_options = {};
		m_pipeline_compile_options.usesMotionBlur = false;
		m_pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
		m_pipeline_compile_options.numPayloadValues = 5;
		m_pipeline_compile_options.numAttributeValues = 3; // TODO
		m_pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE; // should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
		m_pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

		size_t      inputSize = 0;
		const char* input = sutil::getInputData("optixProject", "optixProject", "shaderWholeMesh.cu", inputSize);

		m_module = {};
		char log[2048];
		size_t sizeof_log = sizeof(log);
		OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
			m_context,
			&module_compile_options,
			&m_pipeline_compile_options,
			input,
			inputSize,
			log,
			&sizeof_log,
			&m_module
		));
		std::cout << "create module success." << "\n";
	}
	void createSBT(Scene& staticScene, std::vector<Scene>& dynamicScene)
	{
		//std::cout << "...building instance SBT records..." << std::endl;

		if (comb_sbt.hitgroupRecordBase) cudaFree((void*)comb_sbt.hitgroupRecordBase);
		if (comb_sbt.missRecordBase) cudaFree((void*)comb_sbt.missRecordBase);
		if (comb_sbt.raygenRecord) cudaFree((void*)comb_sbt.raygenRecord);
		
		int meshID = frameCount;

		// ------------------------------------------------------------------
		// build raygen sbt
		// ------------------------------------------------------------------
		RayGenSbtRecord raygenSbtRecord = {};
		OPTIX_CHECK(optixSbtRecordPackHeader(m_raygen_prog_group, &raygenSbtRecord));

		CudaBuffer raygenSbt;
		raygenSbt.alloc_and_upload(&raygenSbtRecord);
		comb_sbt.raygenRecord = raygenSbt.release();


		// ------------------------------------------------------------------
		// build miss sbt
		// ------------------------------------------------------------------


		std::vector<MissSbtRecord> missSbtRecord = {};

		MissSbtRecord nowMissSbtRecord;
		nowMissSbtRecord.data.bg_color = { 0.0f, 0.0f, 0.0f };

		OPTIX_CHECK(optixSbtRecordPackHeader(m_radiance_miss_prog_group, &nowMissSbtRecord));
		missSbtRecord.push_back(nowMissSbtRecord);

		OPTIX_CHECK(optixSbtRecordPackHeader(m_occlusion_miss_prog_group, &nowMissSbtRecord));
		missSbtRecord.push_back(nowMissSbtRecord);

		CudaBuffer missSbt;
		missSbt.alloc_and_upload(missSbtRecord);
		comb_sbt.missRecordBase = missSbt.release();
		comb_sbt.missRecordCount = RAY_TYPE_COUNT;
		comb_sbt.missRecordStrideInBytes = static_cast<uint32_t>(sizeof(MissSbtRecord));

		// ------------------------------------------------------------------
		// build hitgroup records
		// ------------------------------------------------------------------

		std::vector<HitGroupSbtRecord> hitgroupSbtRecord;

		HitGroupSbtRecord nowHitgroupSbtRecord = {};


		//sponza first



		nowHitgroupSbtRecord.data.vertex = (float3*)(staticScene.vertexBuffer[0].get());
		nowHitgroupSbtRecord.data.vertexIndex = (int3*)(staticScene.vertexIndexBuffer[0].get());
		if (staticScene.normalBuffer[0].sizeInBytes())
		{
			nowHitgroupSbtRecord.data.normal = (float3*)(staticScene.normalBuffer[0].get());
			nowHitgroupSbtRecord.data.normalIndex = (int3*)(staticScene.normalIndexBuffer[0].get());
		}

		if (staticScene.texcoordBuffer[0].sizeInBytes())
		{
			nowHitgroupSbtRecord.data.texcoord = (float2*)(staticScene.texcoordBuffer[0].get());
			nowHitgroupSbtRecord.data.texcoordIndex = (int3*)(staticScene.texcoordIndexBuffer[0].get());
		}

		if (staticScene.materialIdBuffer[0].sizeInBytes())
		{
			nowHitgroupSbtRecord.data.materialIDs = (int*)(staticScene.materialIdBuffer[0].get());
		}
		nowHitgroupSbtRecord.data.materials = (Material*)(staticScene.materialBuffer.get());
		if (!staticScene.m_textures.empty())
		{
			nowHitgroupSbtRecord.data.textures = (cudaTextureObject_t*)(staticScene.textureBuffer.get());
		}
		OPTIX_CHECK(optixSbtRecordPackHeader(m_radiance_hitgroup_prog_group, &nowHitgroupSbtRecord));
		hitgroupSbtRecord.push_back(nowHitgroupSbtRecord);

		OPTIX_CHECK(optixSbtRecordPackHeader(m_occlusion_hitgroup_prog_group, &nowHitgroupSbtRecord));
		hitgroupSbtRecord.push_back(nowHitgroupSbtRecord);




		// now ben

		memset(&nowHitgroupSbtRecord, 0, sizeof(HitGroupSbtRecord));



		for (int i = 0; i < dynamicScene.size(); ++i)
		{
			nowHitgroupSbtRecord.data.vertex = (float3*)(dynamicScene[i].vertexBuffer[meshID].get());
			nowHitgroupSbtRecord.data.vertexIndex = (int3*)(dynamicScene[i].vertexIndexBuffer[meshID].get());
			if (dynamicScene[i].normalBuffer[meshID].sizeInBytes())
			{
				nowHitgroupSbtRecord.data.normal = (float3*)(dynamicScene[i].normalBuffer[meshID].get());
				nowHitgroupSbtRecord.data.normalIndex = (int3*)(dynamicScene[i].normalIndexBuffer[meshID].get());
			}

			if (dynamicScene[i].texcoordBuffer[meshID].sizeInBytes())
			{
				nowHitgroupSbtRecord.data.texcoord = (float2*)(dynamicScene[i].texcoordBuffer[meshID].get());
				nowHitgroupSbtRecord.data.texcoordIndex = (int3*)(dynamicScene[i].texcoordIndexBuffer[meshID].get());
			}

			if (dynamicScene[i].materialIdBuffer[meshID].sizeInBytes())
			{
				nowHitgroupSbtRecord.data.materialIDs = (int*)(dynamicScene[i].materialIdBuffer[meshID].get());
			}
			nowHitgroupSbtRecord.data.materials = (Material*)(dynamicScene[i].materialBuffer.get());
			if (!dynamicScene[i].m_textures.empty())
			{
				nowHitgroupSbtRecord.data.textures = (cudaTextureObject_t*)(dynamicScene[i].textureBuffer.get());
			}
			OPTIX_CHECK(optixSbtRecordPackHeader(m_radiance_hitgroup_prog_group, &nowHitgroupSbtRecord));
			hitgroupSbtRecord.push_back(nowHitgroupSbtRecord);

			OPTIX_CHECK(optixSbtRecordPackHeader(m_occlusion_hitgroup_prog_group, &nowHitgroupSbtRecord));
			hitgroupSbtRecord.push_back(nowHitgroupSbtRecord);

		}
		


		CudaBuffer hitgroupSbt;
		hitgroupSbt.alloc_and_upload(hitgroupSbtRecord);

		comb_sbt.hitgroupRecordBase = hitgroupSbt.release();
		comb_sbt.hitgroupRecordCount = hitgroupSbtRecord.size();

		comb_sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(sizeof(HitGroupSbtRecord));
		
		//std::cout << "build SBT records success." << std::endl;
	}
	void prepareScene(Scene& staticScene, std::vector<Scene>& dynamicScene)
	{
		buildAccel(staticScene, dynamicScene);
		createModule();
		createProgramGroups();
		createPipeline();
		createSBT(staticScene, dynamicScene);
	}
	void updateScene(Scene& staticScene, std::vector<Scene>& dynamicScene)
	{
		buildAccel(staticScene, dynamicScene);
		createSBT(staticScene, dynamicScene);
	}

	OptixShaderBindingTable comb_sbt = {};
	OptixTraversableHandle  comb_ias_handle;

	const OptixShaderBindingTable* sbt()				  const { return &(comb_sbt); }
	OptixTraversableHandle		   traversableHandle()   const { return comb_ias_handle; }
};
void loadCameraLightParameter(std::string filePath)
{
	JSONFileManager jsonFile;
	jsonFile.loadFile(filePath);


	eye = jsonFile.getCameraPosition(0);
	dir = jsonFile.getCameraView(0);
	up = jsonFile.getCameraUp(0);
	


	int lightCount = jsonFile.getLightCount();

	for (int lightID = 0; lightID < lightCount; ++lightID)
	{
		BasicLight nowLight;
		nowLight.pos = jsonFile.getLightPosition(lightID);
		nowLight.color = jsonFile.getLightColor(lightID);
		lights.push_back(nowLight);
	}

}
void modelLoader(int model, Scene& scene)
{
	std::string fileName;
	switch (model)
	{
		case BEN:
		{
			frame = 30;
			std::string nowFilePath = "../data/ben/ben_";
			std::string postFix = ".obj";
			for (int i = 0; i < frame; ++i)
			{
				char buffer[50];
				sprintf(buffer, "%02d", i);
				std::string nowFileName = buffer;
				
				fileName = sutil::sampleFilePath(nullptr, (nowFilePath+nowFileName+postFix).c_str());
				scene.loadSceneSeperateMesh(fileName);
			}
			break;
		}
		case SPONZA:
		{
			frame = 1;
			fileName = sutil::sampleFilePath(nullptr, "../data/sponza/sponza.obj");
			loadCameraLightParameter("../data/sponza/crytek_sponza.json");
			scene.loadSceneSeperateMesh(fileName);
			break;
		}
		
		default:
		{
			std::cout << "Unknown model. please add your model in function \"modelSelector()\"." << std::endl;
			exit(1);
			break;
		}
	}
}
void setGlfwCallback(GLFWwindow* window)
{
	glfwSetMouseButtonCallback(window, mouseButtonCallback);
	glfwSetCursorPosCallback(window, cursorPosCallback);
	glfwSetWindowSizeCallback(window, windowSizeCallback);
	glfwSetWindowIconifyCallback(window, windowIconifyCallback);
	glfwSetKeyCallback(window, keyCallback);
	glfwSetScrollCallback(window, scrollCallback);
	glfwSetWindowUserPointer(window, &params);
}
void initCameraState()
{
	camera.setAspectRatio(static_cast<float>(width) / static_cast<float>(height));
	camera.setFovY(fovy);
	camera.setEye(eye);
	camera.setLookat(eye + dir * 100.0f);
	camera.setUp(up);


	camera_changed = true;

	trackball.setCamera(&camera);
	trackball.setMoveSpeed(10.0f);
	trackball.setReferenceFrame(make_float3(1.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 1.0f), make_float3(0.0f, 1.0f, 0.0f));
	trackball.setGimbalLock(true);
}
void InitLaunchParams(CombScene &scene)
{
	params.frame_buffer = nullptr;
	params.subframe_index = 0u;
	params.maxTraceDepth = maxTraceDepth;
	

	//modify start
	/*
	BasicLight nowLight;
	nowLight.color = { 1.0f, 1.0f, 1.0f };
	nowLight.pos = { 20.0f, 20.0f, 0.0f };
	lights.push_back(nowLight);
	nowLight.pos = { -5.0f, 20.0f, 10.0f };
	lights.push_back(nowLight);
	*/
	//modify end

	params.lights.count = static_cast<uint32_t>(lights.size());

	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&params.lights.data),
		lights.size() * sizeof(BasicLight)
	));
	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(params.lights.data),
		lights.data(),
		lights.size() * sizeof(BasicLight),
		cudaMemcpyHostToDevice
	));

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(Params)));

	std::cout << "parameter setting..." << std::endl;
	params.handle = scene.traversableHandle();
	params.width = width;
	params.height = height;
	std::cout << "parameter setting success!" << std::endl;
	
}
void handleResize(sutil::CUDAOutputBuffer<uchar4>& output_buffer)
{
	if (!resize_dirty)
		return;
	resize_dirty = false;
}
void handleCameraUpdate(Params &params)
{
	if (!camera_changed)
		return;
	camera_changed = false;

	camera.setAspectRatio(static_cast<float>(width) / static_cast<float>(height));
	params.eye = camera.eye();
	camera.UVWFrame(params.U, params.V, params.W);

	//modify
	params.U=normalize(params.U);
	params.V=normalize(params.V);
	params.W=normalize(params.W);
}
void handleProjection(Params& params)
{
	float RminusL = 2 * fnear / projectionMatrix.getRow(0).x;
	float RplusL = projectionMatrix.getRow(2).x * RminusL;
	float L = (RplusL - RminusL) * 0.5f;
	float R = RplusL - L;
	float cameraPlaneWidth = R - L;
	float CameraPlaneL = L;

	float TminusB = 2 * fnear / (projectionMatrix.getRow(1)).y;
	float TplusB = projectionMatrix.getRow(2).y * TminusB;
	float T = (TplusB + TminusB) * 0.5f;
	float B = TplusB - T;
	float cameraPlaneHeight = T - B;
	float CameraPlaneT = T;

	params.stepX = cameraPlaneWidth / static_cast<float>(params.width);
	params.stepY = cameraPlaneHeight / static_cast<float>(params.height);
	params.startPoint = params.eye + params.W * fnear + params.V * CameraPlaneT + params.U * CameraPlaneL;
}
void updateState(sutil::CUDAOutputBuffer<uchar4>& output_buffer,CombScene& scene, Scene& staticScene,std::vector<Scene>&dynamicScene, Params& params , std::chrono::steady_clock::time_point &start_time, std::chrono::steady_clock::time_point &now_time)
{
	if (camera_changed || resize_dirty)
		params.subframe_index = 0;
	handleCameraUpdate(params);
	handleResize(output_buffer);
	handleProjection(params); //추가함
	if ((std::chrono::duration<double>)(now_time - start_time) > (std::chrono::duration<double>)(1.0f / frame))
	{
		start_time = now_time;
		frameCount = (frameCount + 1) % frame;
		scene.updateScene(staticScene, dynamicScene);
		params.handle = scene.traversableHandle();
	}
}
void launchSubframe(sutil::CUDAOutputBuffer<uchar4>& output_buffer, const CombScene& scene)
{
	uchar4* result_buffer_data = output_buffer.map();
	params.frame_buffer = result_buffer_data;
	CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_params),
		&params,
		sizeof(Params),
		cudaMemcpyHostToDevice,
		0 // stream
	));
	
	OPTIX_CHECK(optixLaunch(
		scene.pipeline(),
		0,             // stream
		reinterpret_cast<CUdeviceptr>(d_params),
		sizeof(Params),
		scene.sbt(),
		width,  // launch width
		height, // launch height
		1       // launch depth
	));
	
	output_buffer.unmap();
	CUDA_SYNC_CHECK();
}
void displaySubframe(sutil::CUDAOutputBuffer<uchar4>& output_buffer, sutil::GLDisplay& gl_display,GLFWwindow* window) 
{
	int framebuf_res_x = 0;   // The display's resolution (could be HDPI res)
	int framebuf_res_y = 0;   //
	glfwGetFramebufferSize(window, &framebuf_res_x, &framebuf_res_y);
	gl_display.display(
		output_buffer.width(),
		output_buffer.height(),
		framebuf_res_x,
		framebuf_res_y,
		output_buffer.getPBO()
	);
}
void cleanup()
{
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(params.lights.data)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_params)));
}

                                                         
//         user-define functions end                                                                                      
//////////////////////////////////////////////////////////////



int main(int argc, char* argv[])
{
	try
	{
		sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;
	
		
		CombScene scene;
		scene.createContext();
		
		Scene sponzaScene;
		modelLoader(SPONZA, sponzaScene);
		sponzaScene.buildAccel(scene.m_context);
		Scene benSceneInstance;
		modelLoader(BEN, benSceneInstance);
		std::vector<Scene> benScene;
		for (int i = 0; i < 4; ++i) benScene.push_back(benSceneInstance);
		benScene[0].createMatrices(
			CW,	// 회전하는 방향(Clockwise, CounterClockWise)
			PLUS_X, //처음 바라보는 방향
			{ -700.0f, 0.0f, 100.0f }, //처음 위치
			30.0f, //회전 반경
			40.0f, //걷는 길이
			120, // matrix 개수
			200.0f //크기 조정
			);
		benScene[1].createMatrices(CCW, PLUS_Z, { -400.0f, 0.0f, 0.0f }, 10.0f, 60.0f, 90, 150.0f);
		benScene[2].createMatrices(CW, MINUS_X, { -300.0f, 0.0f, -150.0f }, 30.0f, 60.0f, 90, 100.0f);
		benScene[3].createMatrices(CCW, MINUS_Z, { -100.0f, 0.0f, 0.0f }, 30.0f, 50.0f, 100, 300.0f);
		

		scene.prepareScene(sponzaScene, benScene);
		std::cout << "prepareScene success!" << std::endl;
		
		
		//

		OPTIX_CHECK(optixInit());
		 
		GLFWwindow* window = sutil::initUI("optixProject", width, height);
		sutil::CUDAOutputBuffer<uchar4> output_buffer(output_buffer_type, width, height);
		sutil::GLDisplay gl_display;
		setGlfwCallback(window);
		initCameraState();
		InitLaunchParams(scene);

		frame_change_time = std::chrono::steady_clock::now();
		do
		{
			auto t0 = std::chrono::steady_clock::now();
			auto frame_change_now = t0;
			
			glfwPollEvents();

			updateState(output_buffer,scene,sponzaScene,benScene, params, frame_change_time ,frame_change_now);

			auto t1 = std::chrono::steady_clock::now();
			state_update_time += t1 - t0;
			t0 = t1;
			
			launchSubframe(output_buffer, scene);
			
			t1 = std::chrono::steady_clock::now();
			render_time += t1 - t0;

			t0 = t1;

			displaySubframe(output_buffer, gl_display, window);
			t1 = std::chrono::steady_clock::now();

			display_time += t1 - t0;

			sutil::displayStats(state_update_time, render_time, display_time);

			glfwSwapBuffers(window);

			++params.subframe_index;
		} while (!glfwWindowShouldClose(window));
		CUDA_SYNC_CHECK();
		
		sutil::cleanupUI(window);
	}
	catch (std::exception& e)
	{
		std::cerr << "Caught exception: " << e.what() << "\n";
		return 1;
	}
	
	return 0;
} 

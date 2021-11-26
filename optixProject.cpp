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
//                      SS: SEPERATE_LOAD_SEPERATE_BUILD (�̱���)
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
int32_t maxTraceDepth = 3; //��ü �ݻ縦 ������� ƨ����ΰ�? (���� shader���� ����ϴ� ����)
int32_t traceDepthLimit = 5; //��ü�� �ִ� ������� ƨ�� ������ ���� ���ΰ�? (pipeline�� ���� ����, ���� shader���� optixTrace�� recursion Ƚ���� �� limit�� �Ѿ �� ����)
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
				m_materials[materialID].dissolve = 1.0f - materials[materialID].dissolve; //dissolve�� transperency�� �ݴ�.
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

			//buildInput �� �������� GAS build�� �ʿ��� buffer ũ�� ���
			OptixAccelBufferSizes gasBufferSizes;
			OPTIX_CHECK(optixAccelComputeMemoryUsage(
				parameterContext,
				&accelOptions,
				&triangleInput,
				1,				// num_build_inputs
				&gasBufferSizes
			));


			// GAS build ���� compactedOutputBuffer ������ִ� �Ӽ� �߰�
			CudaBuffer compactedSizeBuffer;
			compactedSizeBuffer.alloc(sizeof(uint64_t));

			OptixAccelEmitDesc emitDesc;
			emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
			emitDesc.result = compactedSizeBuffer.get();

			// tempBuffer,outputBuffer �غ�
			CudaBuffer tempBuffer;
			tempBuffer.alloc(gasBufferSizes.tempSizeInBytes);

			CudaBuffer outputBuffer;
			outputBuffer.alloc(gasBufferSizes.outputSizeInBytes);


			// ���� ��������. AccelBuild() �����Ͽ� GAS ����
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


			// compaction �������� üũ
			uint64_t compactedSize;
			compactedSizeBuffer.download_and_free(&compactedSize, sizeof(uint64_t));


			if (compactedSize < gasBufferSizes.outputSizeInBytes) // ������ �� �� �ִٸ�(�̵��� �ִٸ�)
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
			else //������ �ʿ���� ���: ���� outputBuffer�� build�� outputBuffer �״��
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

		//buildInput �� �������� GAS build�� �ʿ��� buffer ũ�� ���
		OptixAccelBufferSizes gasBufferSizes;
		OPTIX_CHECK(optixAccelComputeMemoryUsage(
			parameterContext,
			&accelOptions,
			&triangleInput,
			1,				// num_build_inputs
			&gasBufferSizes
		));


		// GAS build ���� compactedOutputBuffer ������ִ� �Ӽ� �߰�
		CudaBuffer compactedSizeBuffer;
		compactedSizeBuffer.alloc(sizeof(uint64_t));

		OptixAccelEmitDesc emitDesc;
		emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
		emitDesc.result = compactedSizeBuffer.get();

		// tempBuffer,outputBuffer �غ�
		CudaBuffer tempBuffer;
		tempBuffer.alloc(gasBufferSizes.tempSizeInBytes);

		CudaBuffer outputBuffer;
		outputBuffer.alloc(gasBufferSizes.outputSizeInBytes);


		// ���� ��������. AccelBuild() �����Ͽ� GAS ����
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


		// compaction �������� üũ
		uint64_t compactedSize;
		compactedSizeBuffer.download_and_free(&compactedSize, sizeof(uint64_t));


		if (compactedSize < gasBufferSizes.outputSizeInBytes) // ������ �� �� �ִٸ�(�̵��� �ִٸ�)
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
		else //������ �ʿ���� ���: ���� outputBuffer�� build�� outputBuffer �״��
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


			


			std::cout << meshID << "�� �޽� ���� ��. ũ��� " << gasBufferSize.outputSizeInBytes << std::endl;

			std::cout << "tempBuffer �� ũ��� " << tempBuffer.sizeInBytes<<std::endl;
			std::cout << "tempBuffer�� �����ʹ� " << tempBuffer.d_pointer() << std::endl;
			std::cout << "outputBuffer�� ũ��� " << unCompactedOutputBuffer.sizeInBytes << std::endl;
			std::cout << "outputBuffer�� �����ʹ�" << unCompactedOutputBuffer.d_pointer() << std::endl;

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
		/// ������ gas ���� build.
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
	// compacted GAS build �� �ʿ��� �޸� ũ��� 
	// �Ϲ� GAS �� build �ϱ� ������ �� �� ����.
	// ����, GAS�� compaction��
	// 
	// ======================================================
	// 1.�Ϲ� GAS�� ���� �����ϰ�,
	// 2.�̶� �������� �޸� size ��ŭ memory allocate
	// ======================================================
	// 
	// �� ���� ������� �̷������.
	// �̶� device-host �� ����ȭ ����(synchronization point)�� �����. 
	// �̴� �����ս��� ũ�� ��ĥ ���ɼ��� �ִ�.
	// �̷��� ������ ���� build�� compaction�� �ſ� ����, ���� GAS�鿡�� Ư�� ġ�����̴�.
	// 
	// ====================================================================================
	// �ѹ��� �� GAS�� build �ϴ� naive�� �˰����� ������ ����:
	// 1. �Ϲ� gas build�� �ʿ��� �޸� ũ�⸦ ����Ѵ�.(computeMemoryUsage() �Լ�)
	// 2. build buffer�� �� ũ�⸸ŭ allocate �Ѵ�.
	// 3. GAS�� build �Ѵ�.
	// 4. compacted buffer size�� ����Ѵ�.
	// 5. compacted buffer size�� build buffer size ���� ������(�� compaction�� �ǹ����� ���)
	// compacted buffer�� �� ũ�⸸ŭ allocate �Ѵ�.
	// 6. build buffer���� compacted buffer�� compaction�� �����Ѵ�.
	// ====================================================================================
	//
	// 
	// 
	// ������ �˰����� ���̵��:
	// ���� GAS�� building�� compaction process�� ���(batch) ó���Ѵ�.
	// ��� ó���ϸ� host-device �� ����ȭ ������ ���� �� �ִ�. 
	// �̷������δ�, ����ȭ ������ ������ GAS�� �������� batch�� ������ŭ ����߸� �� �ִ�.
	// 
	// GAS�� batch ���ÿ� ����ؾ��� ������ ������ ����:
	// a) GAS�� batch ������ peak memory consumption.
	// b) compacted GAS�� �����ؼ�, output buffer�� �ʿ��� �޸� ũ��.
	// 
	// b�� ����Ѵٸ�, �޸� ũ�⸦ ������ �� �۰� �����ؾ��Ѵ�.
	// ��, output�� �ʿ��� �� �޸� ũ��� compacted GAS�� �հ� ���ƾ��Ѵ�.
	// ����, compacted GAS�� �ʿ��� �޸� ũ�⺸�� ũ�� buffer�� allocating �ϴ� ���� ���ؾ� �Ѵ�.
	// 
	// ���� peak memory consumption�� �� algorithm�� ȿ���̴�.
	// build�� �߻��ϴ� peak memory consumption �� lower bound�� process�� output�̴�. �� compacted GAS�� size�̴�.
	// 
	//
	// ������ �˰����� compacted GAS�� size�� �����ϴµ�, �̴� compaction ratio��� ������ ����Ѵ�.
	// compaction ratio�� size of compacted GAS/size of build output of GAS �̴�.
	// �� ������ ��ȿ���� �׷��Ƿ� compaction ratio�� �󸶳� �� �������߳Ŀ� �޷��ִ�.
	// �˰����� fixed compaction ratio�� �ϴ� ����ϱ�� �Ѵ�.
	// 
	// �ٸ� ������ ���� ���� �ִ�:
	// - compaction ratio �� update �Ѵ�. �̹� ó���� GAS�� ���� ��踦 �� remaining batch�� �����Ѵ�.
	// - GAS�� type�� ���� �ٸ� compaction ratio�� ����Ѵ�.(���� ���, motion vs static). GAS�� type�� ���� compaction ratio�� ���� ���̳���.
	// �� ���ư�, compaction�� skip�ϰ� �� ���� �ִ�.(compaction ratio�� 1.0���� �ξ)
	// 
	// 
	// GAS���� batch�� �����ϱ� ��, ��� GAS���� build size �� �������� �����Ѵ�.
	// ū GAS ���� ���� GAS�� ���� ���� ó���Ѵ�. �̴� peak memory consumption�� minimal memory consumption�� �ִ��� ������ �ϱ� �����̴�. 
	// �̴� ���� batching�� ���� ���� �츱 �� �ִ�. ū GAS ���� ���� GAS���� batching���� �� ū �̵��� ����.
	// minimum batch size�� GAS �Ѱ� ���ʹ�. 
	//
	//
	// ��ǥ:
	// �� �ʿ��� output size(�̴� ���� minimal peak memory consumption)�� ����Ѵ�.
	// GAS�� batch�Ѵ�. memory bound �ȿ� �¾ƶ������� ��ŭ.
	// 
	// ����:
	// �� GAS�� input�� device memory�� �ְ� ���Ŀ��� �ʿ��ϴ�.
	// �׷��� ������ peak memory consumption�� ������ �ش�.
	// ���� ���, GAS build ������ input data�� device�� ���ε��ϰ� �ٷ� Ǯ���ִ� ������ �ϸ� �ȵȴ�. 
	// 
	// ���ư�, peak memory consumption�� �߰������� �ܺ�ȯ�濡 ���� ������ �޴´�.
	// GAS�� build �ϰ� ��·�� ū memory �� �ʿ��� ������ ����ȴٸ�(���� ���, ���� device�� texture data�� �־���� ���),
	// peak memory consumption�� �ᱹ�� Ŀ�� ���̰� GAS build �� �̹� ū �޸𸮸� ����� ���� �ִ�.
	// 
	// TODO:
	// - compaction ratio ���� �Ǵ� update.
	// - compaction �Ұ����� GAS���� ó��.
	// - GAS input data upload / freeing ���ڰ� �����.
	// - �߰����� limit? 
	// 
	//////////////////////////////////////////////////////////////////////////



		std::cout << "...building IAS..." << std::endl;


	const int numMeshes = (int)m_meshes.size();
	vertexBuffer.resize(numMeshes);
	normalBuffer.resize(numMeshes);
	texcoordBuffer.resize(numMeshes);
	indexBuffer.resize(numMeshes);




		// �ʱ��� Compaction ratio.
		constexpr double initialCompactionRatio = 0.5;

		// GAS�� ����� ���߿� trace�� ���̴�.
		// �׶� memory consumption�� ��� compacted GAS + some CUDA stack space ���� �ɰ��̴�.
		// 250MB ���� �߰��ؼ� CUDA stack space requirement �� �밭 match ��Ų��.
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
			// compaction�� ��������� ����Ǵ� �� output size�̴�.
			// minimum peak memory consumption�� �ǹ��ϸ�, ���� GAS�� �����ϱ� ������ �𸥴�.
			// ���� memory constraints result�� minimal peak memory consumption�� �ſ� ����ﶧ�� �����Ѵ�.
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
			// availableMemPoolSize�� �Ʒ� ������ ������Ѵ�:
			//
			// - GAS building �� �ʿ��� temp buffer(build ���� �ʿ��ϹǷ� compaction ���� ����� �ȴ�)
			// - build output buffer
			// - compacted GAS �� ���� ����
			// - compacted GAS



			size_t batchNGASes = 0;
			size_t batchBuildOutputRequirement = 0;
			size_t batchBuildMaxTempRequirement = 0;
			size_t batchBuildCompactedRequirement = 0;
			for (auto it = gases.rbegin(); it != gases.rend(); it++)
			{
				batchBuildOutputRequirement += it->second.gas_buffer_sizes.outputSizeInBytes; //�� �޽��� �����ϴµ� �ʿ��� buffer size�� ���Ѵ�.
				batchBuildCompactedRequirement += (size_t)(it->second.gas_buffer_sizes.outputSizeInBytes * compactionRatio); //compacted GAS �� �ʿ��� buffer size�� ���Ѵ�.
				// roughly account for the storage of the compacted size, although that goes into a separate buffer
				batchBuildOutputRequirement += 8ull;
				// make sure that all further output pointers are 256 byte aligned
				batchBuildOutputRequirement = roundUp<size_t>(batchBuildOutputRequirement, 256ull);
				// temp buffer is shared for all builds in the batch
				batchBuildMaxTempRequirement = std::max(batchBuildMaxTempRequirement, it->second.gas_buffer_sizes.tempSizeInBytes);
				batchNGASes++;
				// ���� ���ݱ��� batch�� �޽����� build buffer size + temp buffer size + compacted buffer size �� available ���� ũ�ٸ� Ż��.
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


		// ������ scene ���� instancing. �̹� gas_handle���� ���尡 �Ǿ�����.
		auto mesh = staticScene.m_meshes[0];

		staticInstance.flags = OPTIX_INSTANCE_FLAG_NONE;
		
		staticInstance.instanceId = static_cast<unsigned int>(0);
		staticInstance.sbtOffset = 0;
		
		staticInstance.visibilityMask = 1;
		staticInstance.traversableHandle = mesh.gas_handle;
		
		memcpy(staticInstance.transform, forIdentity.getData(), sizeof(float) * 12);
		//������ scene instancing �Ϸ�.


		// ���� ��� instancing. �������.

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
	handleProjection(params); //�߰���
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
			CW,	// ȸ���ϴ� ����(Clockwise, CounterClockWise)
			PLUS_X, //ó�� �ٶ󺸴� ����
			{ -700.0f, 0.0f, 100.0f }, //ó�� ��ġ
			30.0f, //ȸ�� �ݰ�
			40.0f, //�ȴ� ����
			120, // matrix ����
			200.0f //ũ�� ����
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

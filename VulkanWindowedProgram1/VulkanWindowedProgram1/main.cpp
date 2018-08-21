/*
 * Vulkan Windowed Program
 *
 * Copyright (C) 2016, 2018 Valve Corporation
 * Copyright (C) 2016, 2018 LunarG, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
Vulkan Windowed Project Template
Create and destroy a Vulkan surface on an SDL window.
*/

// Enable the WSI extensions
#if defined(__ANDROID__)
#define VK_USE_PLATFORM_ANDROID_KHR
#elif defined(__linux__)
#define VK_USE_PLATFORM_XLIB_KHR
#elif defined(_WIN32)
#define VK_USE_PLATFORM_WIN32_KHR
#endif

// Tell SDL not to mess with main()
#define SDL_MAIN_HANDLED
//Had to do this due to a conflict with Windows.h
#define NOMINMAX

#include <glm/glm.hpp>
#include <SDL2/SDL.h>
#include <SDL2/SDL_syswm.h>
#include <SDL2/SDL_vulkan.h>
#include <vulkan/vulkan.h>
#include <assert.h>

#include <iostream>
#include <stdexcept>
#include <functional>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <algorithm>
#include <set>

std::vector<const char*> validationLayers = { "VK_LAYER_LUNARG_standard_validation" };

std::vector<const char*> deviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

const int WIDTH = 1280;
const int HEIGHT = 720;

#if defined(_DEBUG)
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pCallback) {
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
	if (func != nullptr) {
		return func(instance, pCreateInfo, pAllocator, pCallback);
	}
	else {
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT callback, const VkAllocationCallbacks* pAllocator) {
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
	if (func != nullptr) {
		func(instance, callback, pAllocator);
	}
}

struct QueueFamilyIndices
{
	int graphicsFamily = -1;
	int presentFamily = -1;

	bool isComplete()
	{
		return graphicsFamily >= 0 && presentFamily >= 0;
	}
};

struct SwapChainSupportDetails
{
	VkSurfaceCapabilitiesKHR capabilities;
	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR> presentModes;
};

class VulkanTriangleApplication
{
public:
	void run()
	{
		if (!initWindow())
		{
			return;
		}
		if (!initVulkan())
		{
			return;
		}
		mainLoop();
		cleanup();
	}

private:
	SDL_Window * window;
	VkInstance instance;
	VkDebugUtilsMessengerEXT callback;
	VkSurfaceKHR surface;
	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	VkDevice device;
	VkQueue graphicsQueue;
	VkQueue presentQueue;
	
	bool initWindow()
	{
		// Create an SDL window that supports Vulkan rendering.
		if (SDL_Init(SDL_INIT_VIDEO) != 0) {
			std::cout << "Could not initialize SDL." << std::endl;
			return false;
		}
		window = SDL_CreateWindow("Vulkan Window", SDL_WINDOWPOS_CENTERED,
			SDL_WINDOWPOS_CENTERED, WIDTH, HEIGHT, SDL_WINDOW_VULKAN);
		if (window == NULL) {
			std::cout << "Could not create SDL window." << std::endl;
			return false;
		}

		return true;
	}

	bool createInstance()
	{
		if (enableValidationLayers && !checkValidationLayerSupport())
		{
			throw std::runtime_error("validation layers requested, but not available!");
		}

		// VkApplicationInfo allows the programmer to specifiy some basic information about the
		// program, which can be useful for layers and tools to provide more debug information.
		VkApplicationInfo appInfo = {};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "Vulkan Practice Program";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName = "My First Engine";
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_0;

		// VkInstanceCreateInfo is where the programmer specifies the layers and/or extensions that
		// are needed.
		VkInstanceCreateInfo instInfo = {};
		instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		instInfo.pApplicationInfo = &appInfo;

		auto sdlExtensions = getRequiredExtensions();
		instInfo.enabledExtensionCount = static_cast<uint32_t>(sdlExtensions.size());
		instInfo.ppEnabledExtensionNames = sdlExtensions.data();

		auto extensions = getRequiredExtensions();

		if (enableValidationLayers)
		{
			instInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			instInfo.ppEnabledLayerNames = validationLayers.data();
		}
		else
		{
			instInfo.enabledLayerCount = 0;
		}

		VkResult result = vkCreateInstance(&instInfo, nullptr, &instance);
		if (result == VK_ERROR_INCOMPATIBLE_DRIVER) {
			std::cout << "Unable to find a compatible Vulkan Driver." << std::endl;
			return false;
		}
		else if (result) {
			std::cout << "Could not create a Vulkan instance (for unknown reasons)." << std::endl;
			return false;
		}

		return true;
	}

	bool checkValidationLayerSupport()
	{
		uint32_t layerCount;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		for (const char* layerName : validationLayers)
		{
			bool layerFound = false;

			for (const auto& layerProperties : availableLayers)
			{
				if (strcmp(layerName, layerProperties.layerName) == 0)
				{
					layerFound = true;
					break;
				}
			}

			if (!layerFound)
			{
				return false;
			}
		}

		return true;
	}

	bool checkDeviceExtensionSupport(VkPhysicalDevice device)
	{
		uint32_t extensionCount;
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

		std::vector<VkExtensionProperties> availableExtensions(extensionCount);
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

		std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

		for (const auto& extension : availableExtensions)
		{
			requiredExtensions.erase(extension.extensionName);
		}
		return requiredExtensions.empty();
	}

	SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device)
	{
		SwapChainSupportDetails details;

		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

		uint32_t formatCount;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

		if (formatCount != 0)
		{
			details.formats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
		}

		uint32_t presentModeCount;
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

		if (presentModeCount != 0)
		{
			details.presentModes.resize(presentModeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
		}

		return details;
	}

	std::vector<const char*> getRequiredExtensions()
	{
		const char** sdlExtensions = new const char *[1];
		uint32_t sdlExtensionCount = 0;
		if (!SDL_Vulkan_GetInstanceExtensions(window, &sdlExtensionCount, NULL)) {
			std::cout << "Could not get the number of required instance extensions from SDL." << std::endl;
			return {};
		}


		if (!SDL_Vulkan_GetInstanceExtensions(window, &sdlExtensionCount, sdlExtensions)) {
			std::cout << "Could not get the names of required instance extensions from SDL." << std::endl;
			return {};
		}

		std::vector<const char*> extensions(sdlExtensions, sdlExtensions + sdlExtensionCount);

		if (enableValidationLayers)
		{
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		}

		return extensions;
	}

	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
		VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
		VkDebugUtilsMessageTypeFlagsEXT messageType,
		const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
		void* pUserData)
	{
		std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

		return VK_FALSE;
	}

	void setupDebugCallback()
	{
		if (!enableValidationLayers) return;

		VkDebugUtilsMessengerCreateInfoEXT createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		createInfo.pfnUserCallback = debugCallback;

		if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &callback) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to set up debug callback!");
		}
	}

	bool isDeviceSuitable(VkPhysicalDevice device)
	{
		/*VkPhysicalDeviceProperties deviceProperties;
		vkGetPhysicalDeviceProperties(device, &deviceProperties);
		VkPhysicalDeviceFeatures deviceFeatures;
		vkGetPhysicalDeviceFeatures(device, &deviceFeatures);
		
		return deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && deviceFeatures.geometryShader;*/
		QueueFamilyIndices indices = findQueueFamilies(device);

		bool extensionsSupported = checkDeviceExtensionSupport(device);

		bool swapChainAdequate = false;
		if (extensionsSupported)
		{
			SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
			swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
		}

		return indices.isComplete() && extensionsSupported && swapChainAdequate;
	}

	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
	{
		if (availableFormats.size() == 1 && availableFormats[0].format == VK_FORMAT_UNDEFINED)
		{
			return { VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
		}

		for (const auto& availableFormat : availableFormats)
		{
			if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
			{
				return availableFormat;
			}
		}

		return availableFormats[0];
	}

	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR> availablePresentModes)
	{
		for (const auto& availablePresentMode : availablePresentModes)
		{
			if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
			{
				return availablePresentMode;
			}
		}

		return VK_PRESENT_MODE_FIFO_KHR;
	}
	
	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
	{
		if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
		{
			return capabilities.currentExtent;
		}
		else
		{
			VkExtent2D actualExtent = { WIDTH, HEIGHT };

			actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
			actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));

			return actualExtent;
		}
	}

	QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device)
	{
		QueueFamilyIndices indices;

		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

		int index = 0;
		for (const auto& queueFamily : queueFamilies)
		{
			if (queueFamily.queueCount > 0 && queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
			{
				indices.graphicsFamily = index;
			}

			VkBool32 presentSupport = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(device, index, surface, &presentSupport);

			if (queueFamily.queueCount > 0 && presentSupport)
			{
				indices.presentFamily = index;
			}

			if (indices.isComplete())
			{
				break;
			}

			index++;
		}

		return indices;
	}

	void pickPhysicalDevice()
	{
		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

		if (deviceCount == 0)
		{
			throw std::runtime_error("failed to find GPUs with Vulkan support!");
		}

		std::vector<VkPhysicalDevice> devices(deviceCount);
		vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

		for (const auto& device : devices)
		{
			if (isDeviceSuitable(device))
			{
				physicalDevice = device;
				break;
			}
		}

		if (physicalDevice == VK_NULL_HANDLE)
		{
			throw std::runtime_error("failed to find a suitable GPU!");
		}
	}

	void createLogicalDevice()
	{
		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
		std::set<int> uniqueQueueFamilies = { indices.graphicsFamily, indices.presentFamily };

		float queuePriority = 1.0f;

		for (int queueFamily : uniqueQueueFamilies)
		{
			VkDeviceQueueCreateInfo queueCreateInfo = {};
			queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueCreateInfo.queueFamilyIndex = queueFamily;
			queueCreateInfo.queueCount = 1;
			queueCreateInfo.pQueuePriorities = &queuePriority;
			queueCreateInfos.push_back(queueCreateInfo);
		}

		VkPhysicalDeviceFeatures deviceFeatures = {};

		VkDeviceCreateInfo createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		createInfo.pQueueCreateInfos = queueCreateInfos.data();
		createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());

		createInfo.pEnabledFeatures = &deviceFeatures;

		createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
		createInfo.ppEnabledExtensionNames = deviceExtensions.data();

		if (enableValidationLayers) {
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();
		}
		else {
			createInfo.enabledLayerCount = 0;
		}

		if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create logical device.");
		}

		vkGetDeviceQueue(device, indices.graphicsFamily, 0, &graphicsQueue);
		vkGetDeviceQueue(device, indices.presentFamily, 0, &presentQueue);
	}

	void createSurface()
	{
		if (SDL_Vulkan_CreateSurface(window, instance, &surface) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create window surface.");
		}
	}

	bool initVulkan()
	{
		if (!createInstance())
		{
			return false;
		}
		setupDebugCallback();
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();

		return true;
	}

	void mainLoop()
	{
		bool stillRunning = true;
		while (stillRunning) {

			SDL_Event event;
			while (SDL_PollEvent(&event)) {

				switch (event.type) {

				case SDL_QUIT:
					stillRunning = false;
					break;

				default:
					// Do nothing.
					break;
				}
			}

			SDL_Delay(10);
		}
	}

	void cleanup()
	{
		vkDestroyDevice(device, nullptr);
		if (enableValidationLayers)
		{
			DestroyDebugUtilsMessengerEXT(instance, callback, nullptr);
		}
		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyInstance(instance, nullptr);
		SDL_DestroyWindow(window);
		SDL_Quit();
	}
};

int main()
{

	VulkanTriangleApplication app;

	try
	{
		app.run();
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return  EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
    

    

 //   // Use validation layers if this is a debug build
 //   


 //   // Create the Vulkan instance.
 //   // Create a Vulkan surface for rendering
 //   VkSurfaceKHR surface;
 //   if(!SDL_Vulkan_CreateSurface(window, instance, &surface)) {
 //       std::cout << "Could not create a Vulkan surface." << std::endl;
 //       return 1;
 //   }

 //   // This is where most initializtion for a program should be performed
	//uint32_t gpuCount = 0;
	//result = vkEnumeratePhysicalDevices(instance, &gpuCount, NULL);
	//if (gpuCount == 0)
	//{
	//	std::cout << "There are no GPUs available." << std::endl;
	//}

	//std::vector<VkPhysicalDevice> gpus(gpuCount);
	//result = vkEnumeratePhysicalDevices(instance, &gpuCount, gpus.data());

	//uint32_t familyCount;
	//vkGetPhysicalDeviceQueueFamilyProperties(gpus[0], &familyCount, NULL);

	//std::vector<VkQueueFamilyProperties> queueFamilyProps(familyCount);
	//vkGetPhysicalDeviceQueueFamilyProperties(gpus[0], &familyCount, queueFamilyProps.data());

	//if (familyCount == 0)
	//{
	//	std::cout << "There are no Family Properties available." << std::endl;
	//}
	//
	//bool found = false;
	//int familyIndex = 0;
	//for (unsigned int index = 0; index < familyCount; index++)
	//{
	//	if (queueFamilyProps[index].queueFlags & VK_QUEUE_GRAPHICS_BIT)
	//	{
	//		familyIndex = index;
	//		found = true;
	//		break;
	//	}
	//}

	//float queuePriorities[1] = { 0.0 };
	//VkDeviceQueueCreateInfo queueInfo = {};
	//queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	//queueInfo.pNext = NULL;
	//queueInfo.queueCount = 1;
	//queueInfo.pQueuePriorities = queuePriorities;

	//VkDeviceCreateInfo deviceInfo = {};
	//deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	//deviceInfo.pNext = NULL;
	//deviceInfo.queueCreateInfoCount = 1;
	//deviceInfo.pQueueCreateInfos = &queueInfo;
	//deviceInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
	//deviceInfo.ppEnabledExtensionNames = ;
	//deviceInfo.enabledLayerCount = 0;
	//deviceInfo.ppEnabledLayerNames = NULL;
	//deviceInfo.pEnabledFeatures = NULL;

	//VkDevice device;
	//result = vkCreateDevice(gpus[0], &deviceInfo, NULL, &device);
	//assert(result == VK_SUCCESS);

	//VkCommandPoolCreateInfo poolInfo = {};
	//poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	//poolInfo.pNext = NULL;
	//poolInfo.queueFamilyIndex = familyIndex;
	//poolInfo.flags = 0;

	//VkCommandPool commandPool;
	//result = vkCreateCommandPool(device, &poolInfo, NULL, &commandPool);
	//assert(result == VK_SUCCESS);

	//VkCommandBufferAllocateInfo commandBufferInfo = {};
	//commandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	//commandBufferInfo.pNext = NULL;
	//commandBufferInfo.commandPool = commandPool;
	//commandBufferInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	//commandBufferInfo.commandBufferCount = 1;

	//VkCommandBuffer commandBuffer;
	//result = vkAllocateCommandBuffers(device, &commandBufferInfo, &commandBuffer);
	//assert(result == VK_SUCCESS);

	//uint32_t formatCount = 1;
	//VkSurfaceFormatKHR* surfaceFormats = (VkSurfaceFormatKHR *)malloc(formatCount * sizeof(VkSurfaceFormatKHR));
	//result = vkGetPhysicalDeviceSurfaceFormatsKHR(gpus[0], surface, &formatCount, NULL);
	//assert(result == VK_SUCCESS);
	//result = vkGetPhysicalDeviceSurfaceFormatsKHR(gpus[0], surface, &formatCount, surfaceFormats);
	//assert(result == VK_SUCCESS);

	//VkSurfaceCapabilitiesKHR surfaceCapabilities;

	//result = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(gpus[0], surface, &surfaceCapabilities);
	//assert(result == VK_SUCCESS);

	//uint32_t presentModeCount;
	//result = vkGetPhysicalDeviceSurfacePresentModesKHR(gpus[0], surface, &presentModeCount, NULL);
	//assert(result == VK_SUCCESS);
	//VkPresentModeKHR* presentModes = (VkPresentModeKHR *)malloc(presentModeCount * sizeof(VkPresentModeKHR));
	//assert(result == VK_SUCCESS);

	//VkExtent2D swapchainExtent;
	//if (surfaceCapabilities.currentExtent.width == 0xFFFFFFFF)
	//{
	//	swapchainExtent.width = surfaceCapabilities.maxImageExtent.width;
	//	swapchainExtent.height = surfaceCapabilities.maxImageExtent.height;
	//}
	//else
	//{
	//	swapchainExtent = surfaceCapabilities.currentExtent;
	//}

	//VkPresentModeKHR swapchainPresentMode = VK_PRESENT_MODE_FIFO_KHR;

	//uint32_t desiredNumberOfSwapChainImages = surfaceCapabilities.minImageCount;

	//VkSurfaceTransformFlagBitsKHR preTransform;
	//if (surfaceCapabilities.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
	//{
	//	preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
	//}
	//else
	//{
	//	preTransform = surfaceCapabilities.currentTransform;
	//}

	//VkCompositeAlphaFlagBitsKHR compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	//VkCompositeAlphaFlagBitsKHR compositeAlphaFlags[4] = {
	//	VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
	//	VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
	//	VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
	//	VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
	//};

	//for (uint32_t i = 0; i < sizeof(compositeAlphaFlags); i++) {
	//	if (surfaceCapabilities.supportedCompositeAlpha & compositeAlphaFlags[i]) {
	//		compositeAlpha = compositeAlphaFlags[i];
	//		break;
	//	}
	//}

	//const uint32_t familyIndices = static_cast<uint32_t>(familyIndex);
	//VkSwapchainCreateInfoKHR swapchainInfo = {};
	//swapchainInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	//swapchainInfo.pNext = NULL;
	//swapchainInfo.surface = surface;
	//swapchainInfo.minImageCount = desiredNumberOfSwapChainImages;
	//swapchainInfo.imageFormat = surfaceFormats->format;
	//swapchainInfo.imageExtent.width = swapchainExtent.width;
	//swapchainInfo.imageExtent.height = swapchainExtent.height;
	//swapchainInfo.preTransform = preTransform;
	//swapchainInfo.compositeAlpha = compositeAlpha;
	//swapchainInfo.imageArrayLayers = 1;
	//swapchainInfo.presentMode = swapchainPresentMode;
	//swapchainInfo.oldSwapchain = VK_NULL_HANDLE;
	//swapchainInfo.clipped = true;
	//swapchainInfo.imageColorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR;
	//swapchainInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
	//swapchainInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
	//swapchainInfo.queueFamilyIndexCount = 0;
	//swapchainInfo.pQueueFamilyIndices = NULL;

	//VkSwapchainKHR swapchain;
	//result = vkCreateSwapchainKHR(device, &swapchainInfo, NULL, &swapchain);
	//assert(result == VK_SUCCESS);

	//uint32_t swapchainImageCount;
	//result = vkGetSwapchainImagesKHR(device, swapchain, &swapchainImageCount, NULL);
	//assert(result == VK_SUCCESS);

	//VkImage* swapchainImages = (VkImage *)malloc(swapchainImageCount * sizeof(VkImage));
	//assert(swapchainImages);
	//result = vkGetSwapchainImagesKHR(device, swapchain, &swapchainImageCount, swapchainImages);
	//assert(result == VK_SUCCESS);

	//std::vector<SwapchainBuffer> buffers(swapchainImageCount);
	//for (uint32_t index = 0; index < swapchainImageCount; index++)
	//{
	//	buffers[index].image = swapchainImages[index];
	//}
	//free(swapchainImages);

	//VkImageCreateInfo imageInfo = {};
	//const VkFormat depthFormat = VK_FORMAT_D16_UNORM;
	//VkFormatProperties properties;
	//vkGetPhysicalDeviceFormatProperties(gpus[0], depthFormat, &properties);
	//if (properties.linearTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
	//{
	//	imageInfo.tiling = VK_IMAGE_TILING_LINEAR;
	//}
	//else if (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
	//{
	//	imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	//}
	//else
	//{
	//	std::cout << "VK_FORMAT_D!&_UNORM Unsupported.\n";
	//	exit(-1);
	//}

	//imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	//imageInfo.pNext = NULL;
	//imageInfo.imageType = VK_IMAGE_TYPE_2D;
	//imageInfo.format = depthFormat;
	//imageInfo.extent.width = swapchainExtent.width;
	//imageInfo.extent.height = swapchainExtent.height;
	//imageInfo.extent.depth = 1;
	//imageInfo.mipLevels = 1;
	//imageInfo.arrayLayers = 1;
	//imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	//imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	//imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
	//imageInfo.queueFamilyIndexCount = 0;
	//imageInfo.pQueueFamilyIndices = NULL;
	//imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	//imageInfo.flags = 0;

	//VkMemoryAllocateInfo memoryInfo = {};
	//memoryInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	//memoryInfo.pNext = NULL;
	//memoryInfo.allocationSize = 0;
	//memoryInfo.memoryTypeIndex = 0;

	//VkImageViewCreateInfo viewInfo = {};
	//viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	//viewInfo.pNext = NULL;
	//viewInfo.image = VK_NULL_HANDLE;
	//viewInfo.format = depthFormat;
	//viewInfo.components.r = VK_COMPONENT_SWIZZLE_R;
	//viewInfo.components.g = VK_COMPONENT_SWIZZLE_G;
	//viewInfo.components.b = VK_COMPONENT_SWIZZLE_B;
	//viewInfo.components.a = VK_COMPONENT_SWIZZLE_A;
	//viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
	//viewInfo.subresourceRange.baseMipLevel = 0;
	//viewInfo.subresourceRange.levelCount = 1;
	//viewInfo.subresourceRange.baseArrayLayer = 0;
	//viewInfo.subresourceRange.layerCount = 1;
	//viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	//viewInfo.flags = 0;

	//VkMemoryRequirements memoryRequirements;
	//
	//VkImage image;
	//result = vkCreateImage(device, &imageInfo, NULL, &image);
	//assert(result == VK_SUCCESS);

	//vkGetImageMemoryRequirements(device, image, &memoryRequirements);

	//memoryInfo.allocationSize = memoryRequirements.size;
	//VkPhysicalDeviceMemoryProperties physicalMemoryProperties;
	//vkGetPhysicalDeviceMemoryProperties(gpus[0], &physicalMemoryProperties);
	//uint32_t memoryTypeIndex = 0;
	//bool pass = false;

	////We need to determine the required type of memory
	//for (uint32_t index = 0; index < physicalMemoryProperties.memoryTypeCount; index++)
	//{
	//	if ((memoryRequirements.memoryTypeBits & 1) == 1)
	//	{
	//		if ((physicalMemoryProperties.memoryTypes[index].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
	//		{
	//			memoryTypeIndex = index;
	//			pass = true;
	//			break;
	//		}
	//	}
	//	memoryRequirements.memoryTypeBits >>= 1;
	//}
	//assert(pass);

	////Allocate Memory
	//VkDeviceMemory deviceMemory;
	//result = vkAllocateMemory(device, &memoryInfo, NULL, &deviceMemory);
	//assert(result == VK_SUCCESS);

	////Bind the Memory
	//result = vkBindImageMemory(device, image, deviceMemory, 0);
	//assert(result == VK_SUCCESS);

	////Create Image View
	//VkImageView imageView;
	//result = vkCreateImageView(device, &viewInfo, NULL, &imageView);
	//assert(result == VK_SUCCESS);

 //   // Poll for user input.
 //   

 //   // Clean up.
 //   vkDestroySurfaceKHR(instance, surface, NULL);
 //   
 //   vkDestroyInstance(instance, NULL);

 //   return 0;


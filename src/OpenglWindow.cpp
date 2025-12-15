#include "OpenglWindow.h"

#include "glad/glad.h"

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	OpenglWindow* wnd = reinterpret_cast<OpenglWindow*>(glfwGetWindowUserPointer(window));
	if (wnd && wnd->m_keyClBck)
		wnd->m_keyClBck(static_cast<KeyCode>(key), static_cast<Action>(action), static_cast<Modifier>(mods));
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
	OpenglWindow* wnd = reinterpret_cast<OpenglWindow*>(glfwGetWindowUserPointer(window));
	if (wnd && wnd->m_mouseClBck)
	{
		double xpos, ypos;
		glfwGetCursorPos(window, &xpos, &ypos);
		wnd->m_mouseClBck(static_cast<ButtonCode>(button),
						static_cast<Action>(action),
						static_cast<Modifier>(mods), xpos, ypos);
	}
}

void cursorCallback(GLFWwindow* window, double xpos, double ypos)
{
	OpenglWindow* wnd = reinterpret_cast<OpenglWindow*>(glfwGetWindowUserPointer(window));
	if (wnd && wnd->m_cursorPosClBck)
		wnd->m_cursorPosClBck(xpos, ypos);
}

void scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
	OpenglWindow* wnd = reinterpret_cast<OpenglWindow*>(glfwGetWindowUserPointer(window));
	if (wnd && wnd->m_scrollClBck)
		wnd->m_scrollClBck(xoffset, yoffset);
}

OpenglWindow::OpenglWindow(const std::string& title, uint32_t width, uint32_t height)
	: m_width(width)
	, m_height(height)
{
	if (glfwInit() == GL_FALSE)
	{
		const char* errorDesc = new char[256];
		int code = glfwGetError(&errorDesc);
		if (code == GLFW_NOT_INITIALIZED)
		{
			int a = 0;
			++a;
		}
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
	m_ptrHandle = glfwCreateWindow(width, height, title.data(), nullptr, nullptr);
	glfwMakeContextCurrent(m_ptrHandle);

	static bool initGLAD = false;
	if (!initGLAD)
	{
		initGLAD = true;
		gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
	}

	glfwSetWindowUserPointer(m_ptrHandle, this);

	glfwSetKeyCallback(m_ptrHandle, &keyCallback);
	glfwSetMouseButtonCallback(m_ptrHandle, &mouseButtonCallback);
	glfwSetCursorPosCallback(m_ptrHandle, &cursorCallback);
	glfwSetScrollCallback(m_ptrHandle, &scrollCallback);

}

OpenglWindow::~OpenglWindow()
{
	glfwDestroyWindow(m_ptrHandle);
}

uint32_t OpenglWindow::getWidth() const
{
    return m_width;
}

uint32_t OpenglWindow::getHeight() const
{
    return m_height;
}

void OpenglWindow::setKeyCallback(const KeyCallback& callback)
{
	m_keyClBck = callback;
}

void OpenglWindow::setCursorPosCallback(const CursorPosCallback& callback)
{
	m_cursorPosClBck = callback;
}

void OpenglWindow::setMouseCallback(const MouseCallback& callback)
{
	m_mouseClBck = callback;
}

void OpenglWindow::setScrollCallback(const ScrollCallback& callback)
{
	m_scrollClBck = callback;
}

bool OpenglWindow::shouldClose() const
{
	return glfwWindowShouldClose(m_ptrHandle);
}

void OpenglWindow::pollEvents() const
{
	glfwPollEvents();
}

void OpenglWindow::swapBuffers() const
{
	glfwSwapBuffers(m_ptrHandle);
}

GLFWwindow* OpenglWindow::getGLFWHandle() const
{
    return m_ptrHandle;
}

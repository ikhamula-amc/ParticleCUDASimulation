#pragma once

#define GLFW_INCLUDE_NONE
#include <glfw/glfw3.h>

#include <functional>
#include <string>

enum class Modifier
{
	NoModifier = 0,
	Shift = 1,
	Control = 2,
	Alt = 4,
	Super = 8,
};

enum class Action
{
	Release = 0,
	Press = 1,
	Repeat = 2,
};

enum class ButtonCode
{
	Button_0 = 0,
	//... repeats all buttons codes from the glfw header
};

enum class KeyCode
{
	UNKNOWN = -1,
	Space = 32,
	UP,
	DOWN,
	RIGHT,
	LEFT,
//.... repeats all key codes from the glfw header
};

class OpenglWindow
{
	friend void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
	friend void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
	friend void cursorCallback(GLFWwindow* window, double xpos, double ypos);
	friend void scrollCallback(GLFWwindow* window, double xpos, double ypos);

public:
	using KeyCallback = std::function<void(KeyCode, Action, Modifier)>;
	using CursorPosCallback = std::function<void(double, double)>;
	using MouseCallback = std::function<void(ButtonCode, Action, Modifier, double, double)>;
	using ScrollCallback = std::function<void(double, double)>;

	OpenglWindow(const std::string& title, uint32_t width, uint32_t height);
	~OpenglWindow();

	uint32_t getWidth() const;
	uint32_t getHeight() const;
	void setKeyCallback(const KeyCallback& callback);
	void setCursorPosCallback(const CursorPosCallback& callback);
	void setMouseCallback(const MouseCallback& callback);
	void setScrollCallback(const ScrollCallback& callback);
	bool shouldClose() const;
	void pollEvents() const;
	void swapBuffers() const;

	GLFWwindow* getGLFWHandle() const;

private:
	GLFWwindow* m_ptrHandle;
	uint32_t m_width;
	uint32_t m_height;

	KeyCallback m_keyClBck;
	CursorPosCallback m_cursorPosClBck;
	MouseCallback m_mouseClBck;
	ScrollCallback m_scrollClBck;

};

import pyautogui, time

time.sleep(5)
pyautogui.click()
distance = 300
change = 20
while distance > 0:
    pyautogui.drag(distance, 0, duration=0.2)
    distance = distance - change
    pyautogui.drag(0, distance, duration=0.2)
    pyautogui.drag(-distance, 0, duration=0.2)
    distance = distance - change
    pyautogui.drag(0, -distance, duration=0.2)
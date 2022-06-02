message = input("Tell me something, and I will repeat it back to you: ")
print(message)

promot = "\nTell me something, and I will repeat it back to you: "
promot += "\nEnter 'quit' to end the program. "
message = ""
while message != 'quot':
    message = input(promot)
    if message != 'quit':
        print(message)

active = True
while active:
    message = input(promot)
    if message == 'quit':
        active = False
    else:
        print(message)
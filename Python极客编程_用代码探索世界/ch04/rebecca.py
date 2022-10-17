import sys
import os
import random
from collections import defaultdict, Counter


def main():
    message = input("Enter plaintext or ciphertext: ")
    process = input("Enter 'encrypt' or 'decrypt': ")
    while process not in ('encrypt', 'decrypt'):
        process = input("Invalid process. Enter 'encrypt' or 'decrypt': ")
    shift = int(input("Shift value (1-366) = "))
    while not 1 <= shift <= 366:
        shift = int(input("Invalid value. Enter digit from 1 to 366: "))
    infile = input("Enter filename with extension: ")

    if not os.path.exists(infile):
        print("File {} not found. Terminating.".format(infile), file=sys.stderr)
        sys.exit(1)
    text = load_file(infile)
    char_dict = make_dict(text, shift)

    if process == 'encrypt':
        ciphertext = encrypt(message, char_dict)
        if check_for_fail(ciphertext):
            print("\nProblem finding unique keys.", file=sys.stderr)
            print("Try again, change message, or change code book.\n")
            sys.exit()
        print("\nCharacter and number of occurrences in char_dict: \n")
        for key in sorted(char_dict.keys()):
            print("{: >10}{: >10}{: >10}".format(repr(key)[1:-1], str(ord(key)), len(char_dict[key])))
        print("\nNumber of distinct characters: {}".format(len(char_dict)))
        print("Total number of characters: {:,}\n".format(len(text)))
        print("encrypted ciphertext = \n {}\n".format(ciphertext))
        print("decrypted plaintext = ")
        for i in ciphertext:
            print(text[i - shift], end='', flush=True)
    elif process == 'decrypt':
        plaintext = decrypt(message, text, shift)
        print("\ndecrypted plaintext = \n{}".format(plaintext))


def load_file(infile):
    with open(infile) as f:
        loaded_string = f.read().lower()
    return loaded_string


def make_dict(text, shift):
    char_dict = defaultdict(list)
    for index, char in enumerate(text):
        char_dict[char].append(index + shift)
    return char_dict


def encrypt(message, char_dict):
    encrypted = []
    for char in message.lower():
        if len(char_dict[char]) > 1:
            index = random.choice(char_dict[char])
        elif len(char_dict[char]) == 1:
            index = char_dict[char][0]
        elif len(char_dict[char]) == 0:
            print("\nCHaracter {} not in dictionary.".format(char))
            continue
        encrypted.append(index)
    return encrypted


def decrypt(message, text, shift):
    plaintext = ''
    indexes = [s.replace(',', '').replace('[', '').replace(']', '') for s in message.split()]
    for i in indexes:
        plaintext += text[int(i) - shift]
    return plaintext


def check_for_fail(ciphertext):
    check = [k for k, v in Counter(ciphertext).items() if v>1]
    if len(check) > 0:
        return True


if __name__ == '__main__':
    main()
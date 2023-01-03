import detectEnglish, vigenereCipher, pyperclip


def main():
    ciphertext = """Tzx isnz eccjxkg nfq lol mys bbqq I lxcz."""
    hackedMessage = hackVigenereDictionary(ciphertext)

    if hackedMessage != None:
        print('Copying hacked message to clipboard:')
        print(hackedMessage)
        pyperclip.copy(hackedMessage)
    else:
        print('Failed to hack encryption.')


def hackVigenereDictionary(ciphertext):
    fo = open('dictionary.txt')
    words = fo.readlines()
    fo.close()

    for word in words:
        word = word.strip()
        decryptedText = vigenereCipher.decryptMessage(word, ciphertext)
        if detectEnglish.isEnglist(decryptedText, wordPercentage=40):
            print()
            print('Possible encryption break:')
            print('Key ' + str(word) + ': ' + decryptedText[:100])
            print()
            print('Enter D for done, or ust press Enter to continue breaking:')
            response = input('>')

            if response.upper().startswith('D'):
                return decryptedText


if __name__ == '__main__':
    main()
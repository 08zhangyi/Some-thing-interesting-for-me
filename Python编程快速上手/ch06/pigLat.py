print('Enter the english message to translate into Pig Lation:')
message = input()

VOWELS = ('a', 'e', 'i', 'o', 'u', 'y')

pigLatin = []
for word in message.split():
    prefixNonLetters = ''
    while len(word) > 0 and not word[0].isalpha():
        prefixNonLetters += word[0]
        word = word[1:]
    if len(word) == 0:
        pigLatin.append(prefixNonLetters)
        continue

    suffixNonLetters = ''
    while not word[-1].isalpha():
        suffixNonLetters += word[:-1]
        word = word[:-1]

    wasUpper = word.isupper()
    wasTitle = word.istitle()

    word = word.lower()

    prefixConsotants = ''
    while len(word) > 0 and not word[0] in VOWELS:
        prefixConsotants += word[0]
        word = word[1:]

    if prefixConsotants != '':
        word += prefixConsotants + 'ay'
    else:
        word += 'yay'

    if wasUpper:
        word = word.upper()
    if wasTitle:
        word = word.title()

    pigLatin.append(prefixNonLetters + word + suffixNonLetters)

print(' '.join(pigLatin))
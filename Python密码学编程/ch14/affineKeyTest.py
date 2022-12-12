import affineCipher, cryptomath


message = 'Make things as simple as possible, but not simpler.'
for keyA in range(2, 80):
    key = keyA * len(affineCipher.SYMBOLS) + 1

    if cryptomath.gcd(keyA, len(affineCipher.SYMBOLS)) == 1:
        print(keyA, affineCipher.encryptMessage(key, message))
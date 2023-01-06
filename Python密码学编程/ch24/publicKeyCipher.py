import sys, math

SYMBOLS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890 !?.'


def main():
    filename = 'encrypted_file.txt'
    mode = 'encrypt'

    if mode == 'encrypt':
        message = 'Journalists belong in the gutter because that is where the ruling classes throw their guilty secrets. Gerald Priestland. The Founding Fathers gave the free press the protection it must have to bare the secrets of government and inform the people. Hugo Black.'
        pubKeyFilename = 'al_sweigart_pubkey.txt'
        print('Encrypting and writing to %s...' % (filename))
        encryptedText = encryptAndWriteToFile(filename, pubKeyFilename, message)
        print('Encrypted text:')
        print(encryptedText)

    elif mode == 'decrypt':
        privKeyFIlename = 'al_sweigart_privkey.txt'
        print('Reading from %s and decrypting...' % (filename))
        decryptedText = readFromFileAndDecrypt(filename, privKeyFIlename)
        print('Decrypted text:')
        print(decryptedText)


def getBlocksFromText(message, blockSize):
    for character in message:
        if character not in SYMBOLS:
            print('ERROR: The symbol set does not have the character %s' % (character))
            sys.exit()
    blockInts = []
    for blockStart in range(0, len(message), blockSize):
        blockInt = 0
        for i in range(blockStart, min(blockStart + blockSize, len(message))):
            blockInt += (SYMBOLS.index(message[i])) * (len(SYMBOLS) ** (i % blockSize))
        blockInts.append(blockInt)
    return blockInts


def getTextFromBlocks(blockInts, messageLength, blockSize):
    message = []
    for blockInt in blockInts:
        blockMessage = []
        for i in range(blockSize - 1, -1, -1):
            if len(message) + i < messageLength:
                charIndex = blockInt // (len(SYMBOLS) ** i)
                blockInt = blockInt % (len(SYMBOLS) ** i)
                blockMessage.insert(0, SYMBOLS[charIndex])
        message.extend(blockMessage)
    return ''.join(message)


def encryptMessage(message, key, blockSize):
    encryptedBlocks = []
    n, e = key
    for block in getBlocksFromText(message, blockSize):
        encryptedBlocks.append(pow(block, e, n))
    return encryptedBlocks


def decryptMessage(encryptedBlocks, messageLength, key, blockSize):
    decryptedBlocks = []
    n, d = key
    for block in encryptedBlocks:
        decryptedBlocks.append(pow(block, d, n))
    return getTextFromBlocks(decryptedBlocks, messageLength, blockSize)


def readKeyFile(keyFilename):
    fo = open(keyFilename)
    content = fo.read()
    fo.close()
    keySize, n, EorD = content.split(',')
    return (int(keySize), int(n), int(EorD))


def encryptAndWriteToFile(messageFilename, keyFilename, message, blockSize=None):
    keySize, n, e = readKeyFile(keyFilename)
    if blockSize == None:
        blockSize = int(math.log(2 ** keySize, len(SYMBOLS)))
    if not (math.log(2 ** keySize, len(SYMBOLS)) >= blockSize):
        sys.exit('ERROR: Block size is too large for the key and symbol set size. Did you specify the correct key file and encrypted file?')
    encryptedBlocks = encryptMessage(message, (n, e), blockSize)

    for i in range(len(encryptedBlocks)):
        encryptedBlocks[i] = str(encryptedBlocks[i])
    encryptedContent = ','.join(encryptedBlocks)

    encryptedContent = '%s_%s_%s' % (len(message), blockSize, encryptedContent)
    fo = open(messageFilename, 'w')
    fo.write(encryptedContent)
    fo.close()
    return encryptedContent


def readFromFileAndDecrypt(messageFilename, keyFilename):
    keySize, n, d = readKeyFile(keyFilename)

    fo = open(messageFilename)
    content = fo.read()
    messageLength, blockSize, encryptedMessage = content.split('_')
    messageLength = int(messageLength)
    blockSize = int(blockSize)

    if not (math.log(2 ** keySize, len(SYMBOLS)) >= blockSize):
        sys.exit('ERROR: Block size is too large for the key and symbol set size. Did you specify the correct key file and encrypted file?')

    encryptedBlocks = []
    for block in encryptedMessage.split(','):
        encryptedBlocks.append(int(block))

    return decryptMessage(encryptedBlocks, messageLength, (n, d), blockSize)


if __name__ == '__main__':
    main()
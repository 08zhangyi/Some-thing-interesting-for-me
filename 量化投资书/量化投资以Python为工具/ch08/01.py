class Asset(object):
    pass


asset1 = Asset()
print(asset1)
asset1.id = '001'
print(asset1.id)
asset2 = Asset()
asset2.price = 12
print(asset2.price)


class Asset(object):
    def __init__(self, id, price):
        self.id = id
        self.price = price


asset3 = Asset('003', 11.5)
print(asset3)
print(asset3.id)
print(asset3.price)
# asset4 = Asset('004')
from AssetClass import Asset
print(Asset.__doc__)


def print_id(asset):
    print(' The id of the asset is: %s' % (asset.id))


print_id(asset3)
asset5 = Asset('005', 20)
asset5.print_id()


class Asset(object):
    """
    Asset class with specified attributes "id" and "price"
    """
    def __init__(self, id, price):
        self.__id = id
        self.__price = price

    def print_id(self):
        print('The Id of the asset is: %s' % (self.__id))


asset6 = Asset('006', 30)
# asset6.__id
asset6.print_id()


class Asset(object):
    """
    Asset class with specified attributes "id" and "price"
    """
    def __init__(self, id, price):
        self.__id = id
        self.__price = price

    def getID(self):
        return (self.__id)

    def setID(self, idValue):
        if type(idValue) != str:
            return ("Attention!! The type of id must be string !")
        self.__id = idValue


from datetime import datetime


class Asset(object):
    share = 0
    buyTime = datetime.strptime('1900-01-01', '%Y-%m-%d')
    sellTime = datetime.strptime('1900-01-01', '%Y-%m-%d')

    def __init__(self, id, price):
        self.id = id
        self.price = price

    def buy(self, share, time):
        self.share += share
        self.buyTime = datetime.strptime(time, '%Y-%m-%d')

    def sell(self, share, time):
        self.share -= share
        self.sellTime = datetime.strptime(time, '%Y-%m-%d')


class NonShortableAsset(Asset):
    def sell(self, share, time):
        if self.share >= share:
            self.share -= share
            self.sellTime = datetime.strptime(time, '%Y-%m-%d')
        else:
            print('Not permitted! There are not enough share to sell!!')
            print('CUrrent share is ', self.share)

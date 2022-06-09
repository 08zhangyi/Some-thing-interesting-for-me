import pizza


pizza.make_pizza(16, 'pepperoni')
pizza.make_pizza(12, 'mushroom', 'green peppers', 'extra cheese')


from pizza import make_pizza as mp


mp(16, 'pepperoni')
mp(12, 'mushroom', 'green peppers', 'extra cheese')


import pizza as p


p.make_pizza(16, 'pepperoni')
p.make_pizza(12, 'mushroom', 'green peppers', 'extra cheese')


from pizza import *


make_pizza(16, 'pepperoni')
make_pizza(12, 'mushroom', 'green peppers', 'extra cheese')
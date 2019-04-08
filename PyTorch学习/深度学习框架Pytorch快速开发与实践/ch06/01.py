# 初始运行python -m visdom.server
from visdom import Visdom

vis = Visdom()
print(help(vis.text))
vis.text('Hello, world!')
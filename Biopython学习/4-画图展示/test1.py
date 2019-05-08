# 基因进化树的画图
from Bio import Phylo
import pylab
tree = Phylo.read("data\\test2.dnd", "newick")
print(tree)
Phylo.draw(tree, branch_labels=lambda c: c.branch_length)  # matplotlib输出

# 另一种画图方式，需要安装模块，暂时失败
# Phylo.draw_graphviz(tree, prog='dot')
# # pylab.show()
# # pylab.savefig('data\\phylo-dot.png')
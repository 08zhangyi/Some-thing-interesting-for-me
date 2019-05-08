# 基因进化树的画图
from Bio import Phylo
tree = Phylo.read("data\\test2.dnd", "newick")
print(tree)
Phylo.draw(tree, branch_labels=lambda c: c.branch_length)  # matplotlib输出
print(Phylo.draw_ascii(tree))
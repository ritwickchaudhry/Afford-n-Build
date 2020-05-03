import math
import itertools
import numpy as np
from src.geom_transforms import is_contained, is_smaller

def get_wall_free_boxes():
	pass

class AvgMeter():
	def __init__(self):
		self.val = 0.0
		self.cnt = 0

	def update(self, v, c):
		self.val += v
		self.cnt += c
	
	def get_avg(self):
		if self.cnt == 0:
			return 0
		else:
			return self.val/self.cnt


class Node:
	def __init__(self, corners, label, height):
		self.corners = corners
		self.label = label
		self.height = height
		self.children = []
	
	def add_child(self, child):
		self.children.append(child)
		child.parent = self

	def traverse(self, tier):
		self.tier = tier
		sub_tiers = []
		for child in self.children:
			sub_tier = child.traverse(tier+1)
			sub_tiers.append(sub_tier)
		if len(self.children) == 0:
			return self.tier
		else:
			return max(sub_tiers)


class Tree:
	def __init__(self, all_corners, all_labels, all_heights):
		self.root = Node(None, None, None)
		nodes = [Node(corners, labels, heights) for corners, labels, heights in zip(all_corners, all_labels, 
					all_heights)]

		for i in range(len(nodes)):
			smallest = None
			for j in range(len(nodes)):
				if i == j:
					continue
				# Check if i is contained within j
				if nodes[i].height > nodes[j].height and is_contained(nodes[i].corners, nodes[j].corners):
					if smallest is None or is_smaller(nodes[j].corners, nodes[smallest].corners):
						smallest = j

			if smallest is not None:
				nodes[smallest].add_child(nodes[i])
			else:
				self.root.add_child(nodes[i])
		self.nodes = nodes
		self.traverse()
	
	def traverse(self):
		self.max_tier = self.root.traverse(0)
	
	def __iter__(self):
		for node in self.nodes:
			yield node
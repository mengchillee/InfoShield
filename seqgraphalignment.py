import numpy as np
from math import ceil
from collections import Counter
from scipy.special import comb
from itertools import groupby

from utils import *

class SeqGraphAlignment(object):
	__matchscore = 1
	__mismatchscore = -1
	__gap = -2

	def __init__(self, sequence, graph, globalAlign=True,
				 matchscore=__matchscore, mismatchscore=__mismatchscore,
				 gapscore=__gap, *args, **kwargs):
		self._mismatchscore = mismatchscore
		self._matchscore = matchscore
		self._gap = gapscore
		self.sequence    = sequence
		self.graph       = graph
		self.stringidxs  = None
		self.nodeidxs    = None
		self.globalAlign = globalAlign
		self.stringidxs, self.nodeidxs = self.alignStringToGraphFast(*args, **kwargs)

	def alignment_condition(self):
		startslot = True if self.graph.startslot else False
		condition, clist, sw_count = [], [], []
		if startslot:
			sw_count.append(0)
		for i, j in zip(self.stringidxs, self.nodeidxs):
			if startslot:
				if j == None or (i != None and self.sequence[i] != self.graph.nodedict[j].base):
					condition.append(-1)
					clist.append([-1, self.sequence[i]])
					sw_count[-1] += 1
					continue
				else:
					if (len(condition) != 0 and condition[-1] != -1) or len(condition) == 0:
						clist.append([-1, ' '])
					startslot = False

			if j != None and self.graph.nodedict[j].slot:
				startslot = True
				sw_count.append(0)
			if i != None and j != None:
				sb, nb = self.sequence[i], self.graph.nodedict[j].base
				if sb == nb:
					### Matched
					condition.append(0)
					clist.append([0, self.sequence[i]])
				else:
					### Substitution
					condition.append(1)
					clist.append([1, self.sequence[i]])
			elif i == None:
				### Deletion
				condition.append(2)
				clist.append([2, ' '])
			elif j == None:
				### Insertion
				condition.append(3)
				clist.append([3, self.sequence[i]])
		return condition, clist, sw_count

	def alignment_encoding_cost(self):
		condition, clist, sw_count = self.alignment_condition()
		ct = Counter(condition)

		vh = len(self.stringidxs)
		u = self.graph.vocabulary
		u_s = [base for base, c in zip(self.sequence, condition) if c == -1]
		u_a = [base for base, c in zip(self.sequence, condition) if c == 1 or c == 3]
		e = ct[1] + ct[2] + ct[3]

		### Alignment Information
		bits = log_star(vh) + vh

		### Slot Content
		bits += np.sum([1 + log_star(sw) for sw in sw_count]) + word_cost() * len(u_s)

		### Unmatched Words
		bits += log_star(e) + e * ceil(np.log2(vh)) + 2 * e + len(u_a) * word_cost()

		return bits, clist

	def alignmentStrings(self):
		return (" ".join([self.sequence[i] if i is not None else "-" for i in self.stringidxs]),
				" ".join([self.graph.nodedict[j].base if j is not None else "-" for j in self.nodeidxs]))

	def matchscore(self, c1, c2):
		if c1 == c2:
			return self._matchscore
		else:
			return self._mismatchscore

	def matchscoreVec(self, c, v):
		res = np.where(v == c, self._matchscore, self._mismatchscore)
		return res

	def alignStringToGraphFast(self):
		"""Align string to graph - using np to vectorize across the string
		at each iteration."""
		# if not type(self.sequence) == str:
		# 	raise TypeError("Invalid Type")

		l2 = len(self.sequence)
		seqvec = np.array(list(self.sequence))

		nodeIDtoIndex, nodeIndexToID, scores, backStrIdx, backGrphIdx = self.initializeDynamicProgrammingData()
		inserted = np.zeros((l2), dtype=np.bool)

		# having the inner loop as a function improves performance
		# can use Cython, etc on this for significant further improvements
		# can't vectorize this since there's a loop-carried dependency
		#  along the string
		def insertions(i, l2, scores, inserted):
			inserted[:] = False
			for j in range(l2):
				insscore = scores[i+1, j] + self._gap
				if insscore >= scores[i+1, j+1]:
					scores[i+1, j+1] = insscore
					inserted[j] = True

		# Dynamic Programming
		ni = self.graph.nodeiterator()
		for i, node in enumerate(ni()):
			gbase = node.base
			predecessors = self.prevIndices(node, nodeIDtoIndex)

			# calculate all best deletions, matches in one go over all
			# predecessors.

			# First calculate for the first predecessor, over all string posns:
			deletescore = scores[predecessors[0]+1, 1:] + self._gap
			bestdelete = np.zeros((l2), dtype=np.int)+predecessors[0]+1

			matchpoints = self.matchscoreVec(gbase, seqvec)
			matchscore = scores[predecessors[0]+1, 0:-1] + matchpoints
			bestmatch = np.zeros((l2), dtype=np.int)+predecessors[0]+1

			# then, the remaining
			for predecessor in predecessors[1:]:
				newdeletescore = scores[predecessor+1, 1:] + self._gap
				bestdelete     = np.where(newdeletescore > deletescore, predecessor+1, bestdelete)
				deletescore    = np.maximum(newdeletescore, deletescore)

				gbase = self.graph.nodeIdxToBase(predecessor)
				matchpoints = self.matchscoreVec(gbase, seqvec)
				newmatchscore = scores[predecessor+1, 0:-1] + matchpoints
				bestmatch     = np.where(newmatchscore > matchscore, predecessor+1, bestmatch)
				matchscore    = np.maximum(newmatchscore, matchscore)

			# choose best options available of match, delete
			deleted       = deletescore >= matchscore
			backGrphIdx[i+1, 1:] = np.where(deleted, bestdelete, bestmatch)
			backStrIdx [i+1, 1:] = np.where(deleted, np.arange(1, l2+1), np.arange(0, l2))
			scores[i+1, 1:] = np.where(deleted, deletescore, matchscore)

			# insertions: updated in place, don't depend on predecessors
			insertions(i, l2, scores, inserted)
			backGrphIdx[i+1, 1:] = np.where(inserted, i+1, backGrphIdx[i+1, 1:])
			backStrIdx[i+1, 1:] = np.where(inserted, np.arange(l2), backStrIdx[i+1, 1:])

			# if we're doing local alignment, don't let bad global alignment
			# drag us negative
			if not self.globalAlign:
				backGrphIdx[i+1, :] = np.where(scores[i+1, :] > 0, backGrphIdx[i+1, :], -1)
				backStrIdx [i+1, :] = np.where(scores[i+1, :] > 0, backStrIdx[i+1, :], -1)
				scores[i+1, :]      = np.maximum(scores[i+1, :], 0)

		return self.backtrack(scores, backStrIdx, backGrphIdx, nodeIndexToID)

	def prevIndices(self, node, nodeIDtoIndex):
		"""Return a list of the previous dynamic programming table indices
		   corresponding to predecessors of the current node."""
		prev = []
		for predID in list(node.inEdges.keys()):
			prev.append(nodeIDtoIndex[predID])
		# if no predecessors, point to just before the graph
		if len(prev) == 0:
			prev = [-1]
		return prev

	def initializeDynamicProgrammingData(self):
		"""Initalize the dynamic programming tables:
			- set up scores array
			- set up backtracking array
			- create index to Node ID table and vice versa"""
		l1 = self.graph.nNodes
		l2 = len(self.sequence)

		nodeIDtoIndex = {}
		nodeIndexToID = {-1: None}
		# generate a dict of (nodeID) -> (index into nodelist (and thus matrix))
		ni = self.graph.nodeiterator()
		for (index, node) in enumerate(ni()):
			nodeIDtoIndex[node.ID] = index
			nodeIndexToID[index] = node.ID

		# Dynamic Programming data structures; scores matrix and backtracking
		# matrix
		scores = np.zeros((l1+1, l2+1), dtype=np.int)

		# initialize insertion score
		# if global align, penalty for starting at head != 0
		if self.globalAlign:
			scores[0, :] = np.arange(l2+1)*self._gap

			ni = self.graph.nodeiterator()
			for (index, node) in enumerate(ni()):
				prevIdxs = self.prevIndices(node, nodeIDtoIndex)
				best = scores[prevIdxs[0]+1, 0]
				for prevIdx in prevIdxs:
					best = max(best, scores[prevIdx+1, 0])
				scores[index+1, 0] = best + self._gap

		# backtracking matrices
		backStrIdx = np.zeros((l1+1, l2+1), dtype=np.int)
		backGrphIdx = np.zeros((l1+1, l2+1), dtype=np.int)

		return nodeIDtoIndex, nodeIndexToID, scores, backStrIdx, backGrphIdx

	def backtrack(self, scores, backStrIdx, backGrphIdx, nodeIndexToID):
		"""Backtrack through the scores and backtrack arrays.
		   Return a list of sequence indices and node IDs (not indices, which
		   depend on ordering)."""
		besti, bestj = scores.shape
		besti -= 1
		bestj -= 1
		if not self.globalAlign:
			besti, bestj = np.argwhere(scores == np.amax(scores))[-1]
		else:
			# still have to find best final index to start from
			terminalIndices = []
			ni = self.graph.nodeiterator()
			for (index, node) in enumerate(ni()):
				if node.outDegree == 0:
					terminalIndices.append(index)
			besti = terminalIndices[0] + 1
			bestscore = scores[besti, bestj]
			for i in terminalIndices[1:]:
				score = scores[i+1, bestj]
				if score > bestscore:
					bestscore, besti = score, i+1

		matches = []
		strindexes = []
		while (self.globalAlign or scores[besti, bestj] > 0) and not(besti == 0 and bestj == 0):
			nexti, nextj = backGrphIdx[besti, bestj], backStrIdx[besti, bestj]
			curstridx, curnodeidx = bestj-1, nodeIndexToID[besti-1]

			strindexes.insert(0, curstridx if nextj != bestj else None)
			matches.insert   (0, curnodeidx if nexti != besti else None)

			besti, bestj = nexti, nextj

		return strindexes, matches

import numpy as np
import textwrap
import collections
from math import ceil

from utils import *

class Node(object):
	def __init__(self, nodeID=-1, base='N'):
		self.ID = nodeID
		self.base = base
		self.inEdges = {}
		self.outEdges = {}
		self.alignedTo = []
		self.slot = False

	def __str__(self):
		return "(%d:%s)" % (self.ID, self.base)

	def _add_edge(self, edgeset, neighbourID, label, from_neighbour):
		if neighbourID is None:
			return
		if neighbourID in edgeset:
			edgeset[neighbourID].addLabel(label)
		else:
			if from_neighbour:
				edge = Edge(outNodeID=neighbourID, inNodeID=self.ID, label=label)
			else:
				edge = Edge(outNodeID=self.ID, inNodeID=neighbourID, label=label)
			edgeset[neighbourID] = edge

	def addInEdge(self, neighbourID, label):
		self._add_edge(self.inEdges, neighbourID, label, from_neighbour=True)

	def addOutEdge(self, neighbourID, label):
		self._add_edge(self.outEdges, neighbourID, label, from_neighbour=False)

	def nextNode(self, label):
		"""Returns the first (presumably only) outward neighbour
		   having the given edge label"""
		nextID = None
		for e in self.outEdges:
			if label in self.outEdges[e].labels:
				nextID = e
		return nextID

	@property
	def inDegree(self):
		return len(self.inEdges)

	@property
	def outDegree(self):
		return len(self.outEdges)

	@property
	def labels(self):
		"""Returns all the labels associated with an in-edge or an out edge."""
		labelset = set([])
		for e in list(self.inEdges.values()):
			labelset = labelset.union(e.labels)
		for e in list(self.outEdges.values()):
			labelset = labelset.union(e.labels)
		return list(labelset)


class Edge(object):
	def __init__(self, inNodeID=-1, outNodeID=-1, label=None):
		self.inNodeID  = inNodeID
		self.outNodeID = outNodeID
		if label is None:
			self.labels = []
		elif type(label) == list:
			self.labels = label
		else:
			self.labels = [label]

	def addLabel(self, newlabel):
		if newlabel not in self.labels:
			self.labels.append(newlabel)
		return

	def __str__(self):
		nodestr = "(%d) -> (%d) " % (self.inNodeID, self.outNodeID)
		if self.labels is None:
			return nodestr
		else:
			return nodestr + self.labels.__str__()

class POAGraph(object):
	def addUnmatchedSeq(self, seq, label=None, updateSequences=True):
		"""Add a completely independant (sub)string to the graph,
		   and return node index to initial and final node"""
		if seq is None:
			return

		firstID, lastID = None, None
		neededSort = self.needsSort

		for base in seq:
			nodeID = self.addNode(base)
			if firstID is None:
				firstID = nodeID
			if lastID is not None:
				self.addEdge(lastID, nodeID, label)
			lastID = nodeID

		self.__needsort = neededSort  # no new order problems introduced
		if updateSequences:
			self.__seqs.append(seq)
			self.__labels.append(label)
			self.__starts.append(firstID)
		return firstID, lastID

	def __init__(self, seq=None, label=None):
		self._nextnodeID = 0
		self._nnodes = 0
		self._nedges = 0
		self.nodedict = {}
		self.nodeidlist = []   # allows a (partial) order to be imposed on the nodes

		self.vocabulary = set()
		self.startslot = False

		self.__needsort = False
		self.__labels = []
		self.__seqs = []
		self.__starts = []

		if seq is not None:
			self.addUnmatchedSeq(seq, label)
			self.check_vocabulary()

	def check_vocabulary(self):
		ni = self.nodeiterator()
		for node in ni():
			self.vocabulary.add(node.base)

	def encoding_cost(self):
		v = self._nnodes
		u = self.vocabulary
		s = np.sum([1 if node.slot else 0 for node in self.nodedict.values()])

		### Node information
		bits = log_star(v) + v * word_cost()

		### Slot Position
		bits += log_star(s) + s * ceil(np.log2(v))

		return bits

	def nodeIdxToBase(self, idx):
		return self.nodedict[self.nodeidlist[idx]].base

	def addNode(self, base):
		nid = self._nextnodeID
		newnode = Node(nid, base)
		self.nodedict[nid] = newnode
		self.nodeidlist.append(nid)
		self._nnodes += 1
		self._nextnodeID += 1
		self._needsSort = True
		return nid

	def addEdge(self, start, end, label):
		if start is None or end is None:
			return

		if start not in self.nodedict:
			raise KeyError('addEdge: Start node not in graph: '+str(start))
		if end not in self.nodedict:
			raise KeyError('addEdge: End node not in graph: '+str(end))

		oldNodeEdges = self.nodedict[start].outDegree + self.nodedict[end].inDegree

		self.nodedict[start].addOutEdge(end, label)
		self.nodedict[end].addInEdge(start, label)

		newNodeEdges = self.nodedict[start].outDegree + self.nodedict[end].inDegree

		if newNodeEdges != oldNodeEdges:
			self._nedges += 1

		self._needsSort = True
		return

	@property
	def needsSort(self):
		return self.__needsort

	@property
	def nNodes(self):
		return self._nnodes

	@property
	def nEdges(self):
		return self._nedges

	def _simplified_graph_rep(self):
		## TODO: The need for this suggests that the way the graph is currently represented
		## isn't really right and needs some rethinking.

		node_to_pn = {}
		pn_to_nodes = {}

		# Find the mappings from nodes to pseudonodes
		cur_pnid = 0
		for _, node in self.nodedict.items():
			if node.ID not in node_to_pn:
				node_ids = [node.ID] + node.alignedTo
				pn_to_nodes[cur_pnid] = node_ids
				for nid in node_ids:
					node_to_pn[nid] = cur_pnid
				cur_pnid += 1

		# create the pseudonodes
		Pseudonode = collections.namedtuple("Pseudonode", ["pnode_id", "predecessors", "successors", "node_ids"])
		pseudonodes = []

		for pnid in range(cur_pnid):
			nids, preds, succs = pn_to_nodes[pnid], [], []
			for nid in nids:
				node = self.nodedict[nid]
				preds += [node_to_pn[inEdge.outNodeID] for _, inEdge in node.inEdges.items()]
				succs += [node_to_pn[outEdge.inNodeID] for _, outEdge in node.outEdges.items()]

			pn = Pseudonode(pnode_id=pnid, predecessors=preds, successors=succs, node_ids=nids)
			pseudonodes.append(pn)

		return pseudonodes

	def toposort(self):
		"""Sorts node list so that all incoming edges come from nodes earlier in the list."""
		sortedlist = []
		completed = set([])

		##
		## The topological sort of this graph is complicated by the alignedTo edges;
		## we want to nodes connected by such edges to remain near each other in the
		## topological sort.
		##
		## Here we'll create a simple version of the graph that merges nodes that
		## are alignedTo each other, performs the sort, and then decomposes the
		## 'pseudonodes'.
		##
		## The need for this suggests that the way the graph is currently represented
		## isn't quite right and needs some rethinking.
		##

		pseudonodes = self._simplified_graph_rep()

		def dfs(start, complete, sortedlist):
			stack, started = [start], set()
			while stack:
				pnodeID = stack.pop()

				if pnodeID in complete:
					continue

				if pnodeID in started:
					complete.add(pnodeID)
					for nid in pseudonodes[pnodeID].node_ids:
						sortedlist.insert(0, nid)
					started.remove(pnodeID)
					continue

				successors = pseudonodes[pnodeID].successors
				started.add(pnodeID)
				stack.append(pnodeID)
				stack.extend(successors)

		while len(sortedlist) < self.nNodes:
			found = None
			for pnid in range(len(pseudonodes)):
				if pnid not in completed and len(pseudonodes[pnid].predecessors) == 0:
					found = pnid
					break
			assert found is not None
			dfs(found, completed, sortedlist)

		assert len(sortedlist) == self.nNodes
		self.nodeidlist = sortedlist
		self._needsSort = False
		return

	def nodeiterator(self):
		if self.needsSort:
			self.toposort()

		def nodegenerator():
			for nodeidx in self.nodeidlist:
				yield self.nodedict[nodeidx]

		return nodegenerator

	def seq(self):
		selfstr = ""
		ni = self.nodeiterator()
		for node in ni():
			selfstr += node.base + " "
		return selfstr

	def __str__(self):
		selfstr = ""
		ni = self.nodeiterator()
		for node in ni():
			selfstr += node.__str__() + "\n"
			for outIdx in node.outEdges:
				selfstr += "        " + node.outEdges[outIdx].__str__() + "\n"
		return selfstr

	def incorporateSeqAlignment(self, alignment, seq, label=None):
		"""Incorporate a SeqGraphAlignment into the graph."""
		newseq     = alignment.sequence
		stringidxs = alignment.stringidxs
		nodeidxs   = alignment.nodeidxs

		firstID = None
		headID = None
		tailID = None

		# head, tail of sequence may be unaligned; just add those into the
		# graph directly
		validstringidxs = [si for si in stringidxs if si is not None]
		startSeqIdx, endSeqIdx = validstringidxs[0], validstringidxs[-1]
		if startSeqIdx > 0:
			firstID, headID = self.addUnmatchedSeq(newseq[0:startSeqIdx], label, updateSequences=False)
		if endSeqIdx < len(newseq):
			tailID, __ = self.addUnmatchedSeq(newseq[endSeqIdx+1:], label, updateSequences=False)

		# now we march along the aligned part. For each base, we find or create
		# a node in the graph:
		#   - if unmatched, the corresponding node is a new node
		#   - if matched:
		#       - if matched to a node with the same base, the node is that node
		#       - if matched to a node with a different base whch is in turn
		#         aligned to a node with the same base, that aligned node is
		#         the node
		#       - otherwise, we create a new node.
		# In all cases, we create edges (or add labels) threading through the
		# nodes.
		for sindex, matchID in zip(stringidxs, nodeidxs):
			if sindex is None:
				continue
			base = newseq[sindex]
			if matchID is None:
				nodeID = self.addNode(base)
			elif self.nodedict[matchID].base == base:
				nodeID = matchID
			else:
				otherAligns = self.nodedict[matchID].alignedTo
				foundNode = None
				for otherNodeID in otherAligns:
					if self.nodedict[otherNodeID].base == base:
						foundNode = otherNodeID
				if foundNode is None:
					nodeID = self.addNode(base)
					self.nodedict[nodeID].alignedTo = [matchID] + otherAligns
					for otherNodeID in [matchID] + otherAligns:
						self.nodedict[otherNodeID].alignedTo.append(nodeID)
				else:
					nodeID = foundNode

			self.addEdge(headID, nodeID, label)
			headID = nodeID
			if firstID is None:
				firstID = headID

		# finished the unaligned portion: now add an edge from the current headID to the tailID.
		self.addEdge(headID, tailID, label)

		# resort
		self.toposort()
		self.check_vocabulary()

		self.__seqs.append(seq)
		self.__labels.append(label)
		self.__starts.append(firstID)
		return

	def selectEdge(self, threshold):
		"""New"""

		del_nodes = []
		ni = self.nodeiterator()
		for idx, node in enumerate(ni()):
			nodeID = str(node.ID)
			no_out = True
			for neighbourID, edge in node.outEdges.items():
				if len(edge.labels) > threshold:
					no_out = False
					break
			no_in = True
			for neighbourID, edge in node.inEdges.items():
				if len(edge.labels) > threshold:
					no_in = False
					break
			if (no_out and idx != self.nNodes - 1) or (no_in and idx != 0):
				del_nodes.append(nodeID)

		template = POAGraph()
		firstID, lastID = None, None

		ni = self.nodeiterator()
		for node in ni():
			nodeID = str(node.ID)
			if nodeID not in del_nodes:
				tnodeID = template.addNode(node.base)
				if firstID is None:
					firstID = tnodeID
				if lastID is not None:
					template.addEdge(lastID, tnodeID, None)
				lastID = tnodeID
		template.vocabulary = self.vocabulary
		return template

	def jsOutput(self):
		"""returns a list of strings containing a a description of the graph for viz.js, http://visjs.org"""
		lines = ['var nodes = [']

		ni = self.nodeiterator()
		count = 0
		for node in ni():
			line = '    {id:' + str(node.ID) + ', label: "' + node.base + '"'
			if node.slot:
				line += ', color: "red"'
			else:
				line += '},'
			lines.append(line)

		lines[-1] = lines[-1][:-1]
		lines.append('];')
		lines.append(' ')

		lines.append('var edges = [')
		ni = self.nodeiterator()
		for node in ni():
			nodeID = str(node.ID)
			for edge in node.outEdges:
				target = str(edge)
				weight = str(len(node.outEdges[edge].labels)+1)
				lines.append('    {from: ' + nodeID + ', to: ' + target + ', value: ' + weight + '},')
			for alignededge in node.alignedTo:
				if node.ID > alignededge:
					continue
				target = str(alignededge)
				lines.append('    {from: ' + nodeID + ', to: ' + target + ', value: 1, style: "dash-line"},')
		lines[-1] = lines[-1][:-1]
		lines.append('];')
		return lines


	def htmlOutput(self, outfile):
		header = """
				  <!doctype html>
				  <html>
				  <head>
					<title>POA Graph Alignment</title>

					<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/vis/3.11.0/vis.min.js"></script>
				  </head>

				  <body>

				  <div id="mynetwork"></div>

				  <script type="text/javascript">
					// create a network
				  """
		outfile.write(textwrap.dedent(header[1:]))
		lines = self.jsOutput()
		for line in lines:
			outfile.write(line+'\n')
		footer = """
				  var container = document.getElementById('mynetwork');
				  var data= {
					nodes: nodes,
					edges: edges,
				  };
				  var options = {
					width: '100%',
					height: '800px'
				  };
				  var network = new vis.Network(container, data, options);
				</script>

				</body>
				</html>
				"""
		outfile.write(textwrap.dedent(footer))

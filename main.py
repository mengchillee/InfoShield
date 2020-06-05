import argparse
import sys
import pickle
import progressbar
import numpy as np
import copy
import os
from collections import defaultdict
import matplotlib.pyplot as plt

import poagraph
import seqgraphalignment
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file_name', type=str, default='data/sythetic.pkl', help='Input file name')
args = parser.parse_args()

def all_sequences_cost(pads, gid_arr, gid, template):
	align_cost, cond_int = 0, []
	for id in gid:
		sequence = pads[gid_arr[id]]
		alignment = seqgraphalignment.SeqGraphAlignment(sequence, template)
		ac, ci = alignment.alignment_encoding_cost()
		align_cost += ac
		cond_int.append(np.array(ci)[:, 0].astype(int))
	return align_cost + template.encoding_cost(), cond_int

def sequential_search(pads, gid_arr, gid, graph):
	prev_template = graph.selectEdge(0)
	prev_cost, _ = all_sequences_cost(pads, gid_arr, gid, prev_template)

	### Sweep begin
	for ths in progressbar.progressbar(range(1, len(gid) - 1)):
		template = graph.selectEdge(ths)
		if template.nNodes == 0:
			break

		cost, _ = all_sequences_cost(pads, gid_arr, gid, template)
		### Return the template with local minimum
		if cost > prev_cost:
			return prev_template, prev_cost
		prev_template, prev_cost = template, cost

	return prev_template, prev_cost

def slot_identify(pads, gid_arr, gid, template):
	_, cond_int = all_sequences_cost(pads, gid_arr, gid, template)
	result, e_arr, vh_arr = defaultdict(dict), [], []
	for idx, cond in enumerate(cond_int):
		startslot, count, tmp = True, 0, 0
		e_arr.append(len(cond[cond > 0]))
		vh_arr.append(len(cond))
		for c in cond:
			if startslot:
				if c in [1, 3]:
					tmp += 1
					continue
				result[-1][idx] = tmp
				startslot, tmp = False, 0
				continue
			if c in [1, 3]:
				tmp += 1
			else:
				if tmp != 0:
					result[count][idx] = tmp
				count, tmp = count + 1, 0
		if tmp != 0:
			result[count][idx] = tmp

	slot_count, v = 0, template.nNodes
	for k, n in result.items():
		sp1 = log_star(slot_count) + slot_count * ceil(np.log2(v))
		sp2 = log_star(slot_count + 1) + (slot_count + 1) * ceil(np.log2(v))

		sc = len(cond_int) + np.sum([log_star(nn) + nn * word_cost() for nn in n.values()])
		uw1, uw2 = 0, 0
		for kk, vv in n.items():
			e, vh = e_arr[kk], vh_arr[kk]
			uw1 += log_star(e) + e * ceil(np.log2(vh)) + 2 * e + e * word_cost()
			e -= vv
			uw2 += log_star(e) + e * ceil(np.log2(vh)) + 2 * e + e * word_cost()

		if uw1 + sp1 > uw2 + sp2 + sc:
			slot_count += 1
			for kk, vv in n.items():
				e_arr[kk] -= vv
			if k == -1:
				template.startslot = True
			else:
				template.nodedict[k].slot = True

	return template

if __name__ == '__main__':
	with open(args.file_name, 'rb') as handle:
		pads = pickle.load(handle)

	### Set gloabel vocabulary cost
	gvc = ceil(np.log2(len(np.unique(np.concatenate(list(pads.values()))))))
	set_global_voc_cost(gvc)

	prev_total_cost = np.sum([sequence_cost(s) for _, s in pads.items()])
	gid_arr = [l for l, _ in pads.items()]

	print('\nInput %d Sequences' % (len(pads)))
	print('Initial Total Cost: %d\n' % (prev_total_cost))

	temp_arr, cond_arr, iter = [], [], 0
	while len(gid_arr) > 0:
		iter += 1
		print('---\nIteration %d\n' % iter)
		graph, gid = poagraph.POAGraph(pads[gid_arr[0]], gid_arr[0]), [0]
		seq_total_cost = sequence_cost(pads[gid_arr[0]])
		graph_0 = copy.deepcopy(graph)

		print('Candidate Generation Start...')
		for idx, label in enumerate(progressbar.progressbar(gid_arr[1:])):
			sequence = pads[label]
			alignment = seqgraphalignment.SeqGraphAlignment(sequence, graph_0)
			align_mdl, _ = alignment.alignment_encoding_cost()
			seq_cost = sequence_cost(sequence)

			if align_mdl < seq_cost:
				gid.append(idx + 1)
				alignment = seqgraphalignment.SeqGraphAlignment(sequence, graph)
				graph.incorporateSeqAlignment(alignment, sequence, label)
				seq_total_cost += seq_cost
		print('Finish!\n')

		if len(gid) > 1:
			print('Threshold Search Start...')
			template, min_cost = sequential_search(pads, gid_arr, gid, graph)
			print('Finish!\n')

			print('Slot Identification Start...')
			template = slot_identify(pads, gid_arr, gid, template)
			print('Finish!\n')

			align_cost, c_arr = 0, []
			for id in gid:
				sequence = pads[gid_arr[id]]
				alignment = seqgraphalignment.SeqGraphAlignment(sequence, template)
				cost, cond = alignment.alignment_encoding_cost()
				align_cost += cost
				c_arr.append(cond)
			total_cost = prev_total_cost - seq_total_cost + (template.encoding_cost() + align_cost)

			### Check whether total cost decreases by this template
			if total_cost < prev_total_cost:
				prev_total_cost = total_cost
				temp_arr.append(template)
				cond_arr.append(c_arr)
				print('Find Template for %d Sequences' % (len(gid)))
			else:
				print('Find %d Noises' % (len(gid)))
		else:
			### Only one senqeucne is definitely a noise
			print('Find a Noise')

		### Delete the assigned sequences
		print('Total Cost: ', prev_total_cost, '\n')
		gid_arr = np.delete(gid_arr, gid)

	print('Finish %d Iterations' % (iter))
	print('Find %d Template(s)' % (len(temp_arr)))

	output_results(temp_arr, cond_arr, args.file_name)

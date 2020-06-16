import sys
import pickle
import numpy as np
import copy
import os
from collections import defaultdict
from termcolor import colored
import time
import pandas as pd
import progressbar
import matplotlib.pyplot as plt

import poagraph
import seqgraphalignment
from utils import *

def all_sequences_cost(pads, gid_arr, gid, template):
	align_cost, cond_int = 0, []
	for id in gid:
		sequence = pads[gid_arr[id]]
		alignment = seqgraphalignment.SeqGraphAlignment(sequence, template)
		ac, ci = alignment.alignment_encoding_cost()
		align_cost += ac
		cond_int.append(np.array(ci)[:, 0].astype(int))
	return align_cost + template.encoding_cost(), cond_int

def dichotomous_search(pads, gid_arr, gid, graph):
	m_l, m_r = 0, len(gid) - 1
	cost_dict = {}
	while m_l < m_r:
		m_m = int((m_l + m_r) / 2)
		t1, t2 = graph.selectEdge(m_m - 1), graph.selectEdge(m_m + 1)
		if m_m - 1 not in cost_dict.keys():
			t1 = graph.selectEdge(m_m - 1)
			cost_dict[m_m - 1] = all_sequences_cost(pads, gid_arr, gid, t1)[0]
		asc1 = cost_dict[m_m - 1]
		if m_m + 1 not in cost_dict.keys():
			t2 = graph.selectEdge(m_m + 1)
			cost_dict[m_m + 1] = all_sequences_cost(pads, gid_arr, gid, t2)[0]
		asc2 = cost_dict[m_m + 1]

		if asc1 <= asc2:
			m_r = max(m_l, m_m - 1)
		else:
			m_l = min(m_r, m_m + 1)
	m_m = m_r if asc1 <= asc2 else m_l

	template = graph.selectEdge(m_m)
	cost = all_sequences_cost(pads, gid_arr, gid, template)
	return template, cost

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
			uw1 += e * ceil(np.log2(vh)) + 2 * e + e * word_cost()
			e -= vv
			uw2 += e * ceil(np.log2(vh)) + 2 * e + e * word_cost()

		if uw1 + sp1 > uw2 + sp2 + sc:
			slot_count += 1
			for kk, vv in n.items():
				e_arr[kk] -= vv
			if k == -1:
				template.startslot = True
			else:
				template.nodedict[k].slot = True

	return template

def AutoDup_MDL(pads, output_path):
	### Set gloabel vocabulary cost
	gvc = ceil(np.log2(len(np.unique(np.concatenate(list(pads.values()))))))
	set_global_voc_cost(gvc)

	init_cost = prev_total_cost = np.sum([sequence_cost(s) for _, s in pads.items()]) + len(pads)
	gid_arr = np.array([l for l, _ in pads.items()])

	start0 = time.time()
	temp_arr, cond_arr, temp_dict, iter = [], [], {}, 0
	while len(gid_arr) > 0:
		iter += 1

		graph, gid = poagraph.POAGraph(pads[gid_arr[0]], gid_arr[0]), [0]
		seq_total_cost = sequence_cost(pads[gid_arr[0]])
		graph_0 = copy.deepcopy(graph)

		# print(colored('Candidate Generation Start...', 'green'))
		start1 = time.time()
		for idx, label in enumerate(gid_arr[1:]):
			sequence = pads[label]
			alignment = seqgraphalignment.SeqGraphAlignment(sequence, graph_0)
			align_mdl, _ = alignment.alignment_encoding_cost()
			seq_cost = sequence_cost(sequence)

			if align_mdl < seq_cost:
				gid.append(idx + 1)
				alignment = seqgraphalignment.SeqGraphAlignment(sequence, graph)
				graph.incorporateSeqAlignment(alignment, sequence, label)
				seq_total_cost += seq_cost
		end1 = time.time()

		if len(gid) > 1:
			template, min_cost = dichotomous_search(pads, gid_arr, gid, graph)
			template = slot_identify(pads, gid_arr, gid, template)

			align_cost, c_arr = 0, []
			for id in gid:
				sequence = pads[gid_arr[id]]
				alignment = seqgraphalignment.SeqGraphAlignment(sequence, template)
				cost, cond = alignment.alignment_encoding_cost()
				align_cost += cost
				c_arr.append(cond)

			total_cost = prev_total_cost - seq_total_cost
			if len(temp_arr) != 0:
				total_cost -= log_star(len(temp_arr)) + len(gid_arr) * ceil(np.log2(len(temp_arr)))
			total_cost += (len(gid_arr) + len(gid)) * ceil(np.log2(len(temp_arr) + 1))
			total_cost += log_star(len(temp_arr) + 1) + template.encoding_cost() + align_cost

			### Check whether total cost decreases by this template
			if total_cost < prev_total_cost:
				prev_total_cost = total_cost
				temp_arr.append(template)
				cond_arr.append(c_arr)
				temp_dict[len(temp_arr)] = gid_arr[gid]

		### Delete the assigned sequences
		gid_arr = np.delete(gid_arr, gid)

	end0 = time.time()
	output_results(temp_arr, cond_arr, output_path)
	return (init_cost - prev_total_cost) / init_cost, temp_dict, end0 - start0


def run_autodupfine(filename, id_str='id', text_str='text'):
	data = read_data(filename)

	col1, col2, total_time = [], [], 0
	lab_id, tmp_id, seq_id = [], [], []
	for k, v in progressbar.progressbar(data.items()):
		output_path = os.path.join('results', str(k))
		if not os.path.exists(output_path):
			os.makedirs(output_path)
		comp_rate, temp_dict, exec_time = AutoDup_MDL(v, output_path)
		col1.append(k)
		col2.append(comp_rate)
		for k2, v2 in temp_dict.items():
			for v3 in v2:
				lab_id.append(k)
				tmp_id.append(k2)
				seq_id.append(v3)
		total_time += exec_time
	print(total_time, 'Seconds')

	d = {'Cluster Label': col1, 'Compression Rate': col2}
	df = pd.DataFrame(data=d)
	df.to_csv('compression_rate.csv', index=False)

	d = {'LSH Label': lab_id, 'Template #': tmp_id, 'ID': seq_id}
	df = pd.DataFrame(data=d)
	df.to_csv('template_table.csv', index=False)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Please provide a filename!')

    if len(sys.argv) == 4:
        id_str = sys.argv[2]
        text_str = sys.argv[3]
    else:
        id_str = 'id'
        text_str = 'text'

    run_autodupfine(sys.argv[1], id_str, text_str)

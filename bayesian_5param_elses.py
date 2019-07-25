# -*- coding: utf-8 -*-
import pymc
import numpy as np
import matplotlib.pyplot as plt
import copy, math, os, sys, collections, random, argparse

def main(l, y_range):
    f = open('elapse_time_table.txt', 'r')
    dict_value = collections.OrderedDict()
    cnt = 0
    table_item = ''

    for line in (f):
        if line[0] == '#' and not cnt == 0:
            table_item = line[1:].split()

            for nd, dic in enumerate(table_item):
                dict_value.update({dic:[]})
            cnt += 1

        if not line[0] == '#':
            data = line.split()
            for n, item in enumerate(table_item):
                tmp_list = dict_value[item]
                tmp_list.append(float(data[n]))
                dict_value.update({item:tmp_list})

        cnt += 1

    routine_list = []
    for item in table_item:
        if not item in ['node', 'SEP', 'Reducer']:
            routine_list.append(item)

    parameter = ['c1','c2','c3','c4','c5','eps','tau']

    def model(x, y):
        c1 = pymc.Uniform('c1', lower=0, upper=1000000)#c1の初期分布（lowerからupperまでの一様分布）
        c2 = pymc.Uniform('c2', lower=0, upper=1000000)#c2の初期分布（lowerからupperまでの一様分布）
        c3 = pymc.Uniform('c3', lower=0, upper=1000000)#c3の初期分布（lowerからupperまでの一様分布）
        c4 = pymc.Uniform('c4', lower=0, upper=100000000)#c4の初期分布（lowerからupperまでの一様分布）
        c5 = pymc.Uniform('c5', lower=0, upper=1000000)#c5の初期分布（lowerからupperまでの一様分布）
        eps = pymc.Uniform('eps', lower=0, upper=0.5)#誤差パラメータepsの初期分布（lowerからupperまでの一様分布）
        #c1 = pymc.Beta('c1', alpha=1, beta=9)
        #c2 = pymc.Beta('c2', alpha=1, beta=9)
        #c3 = pymc.Beta('c3', alpha=1, beta=9)
        #c4 = pymc.Beta('c4', alpha=1, beta=1)
        #c5 = pymc.Beta('c5', alpha=1, beta=1)
        #eps = pymc.Beta('eps', alpha=1, beta=1)

        @pymc.deterministic
        def function(x=x, c1=c1, c2=c2, c3=c3, c4=c4, c5=c5):
            if l:
                #return np.log((c4 / np.exp(2.0*x)) + (c1 / np.exp(x)) + c2 + (c3 * x) + (c5 * x)/(np.exp(0.5*x)))
                x_list = []
                for i in range(len(x)):
                    if x[i]>100:
                        term5 = 0.0
                    else:
                        term5 = (c5 * x[i])/(np.exp(0.5*x[i]))
                    x_list.append(np.log((c4 / np.exp(2.0*x[i])) + (c1 / np.exp(x[i])) + c2 + (c3 * x[i]) + term5))
                return x_list                
                
            else:
                return np.log((c4 / np.exp(2.0*x)) + (c1 / np.exp(x)) + c2 + (c3 * np.exp(x)) + c5*(np.log(x)/np.sqrt(x)))

        @pymc.deterministic
        def tau(eps=eps):
            return np.power(eps, -2)

        y = pymc.Normal('y', mu=function, tau=tau, value=y, observed=True)
        return locals()

    split_start = 0
    n_split = 15 #教師データの数（n_split個目までを教師データとして学習）
    n_split_true = 5

    x_true = copy.deepcopy(dict_value['node'][split_start:n_split_true])
    x_hpc = [128., 256., 512., 1024., 2048., 4096., 8192.]
    node = np.log(dict_value['node'][split_start:n_split])

    print(dict_value['node'][split_start:n_split])
    data_list = []
    print("routine_list:",routine_list)
    for i in routine_list:
        print(i, dict_value[i][split_start:n_split])
        y_true = copy.deepcopy(dict_value[i][split_start:n_split_true])
        time = copy.deepcopy(dict_value[i][split_start:n_split])
        time = np.log(time)

        pymc.numpy.random.seed(0)
        mcmc = pymc.MCMC(model(node, time))
        mcmc.sample(iter=100000, burn=50000, thin=10)

        os.system('mkdir %s' % i)

        trace_list = []
        for j in parameter:
            pymc.Matplot.plot(mcmc.trace(j))
            mcmctrace = np.array(mcmc.trace("%s" % j, chain=None)[:])
            print("mcmctrace:",mcmctrace)
            trace_list.append(mcmctrace)
            pymc.Matplot.savefig('./%s/graph_%s.png' % (i, j))
            plt.clf()
            plt.close()

        print("len(trace_list)=%s:" % len(trace_list))
        print("len(trace_list[0])=%s:" % len(trace_list[0]))
        print("trace_list[0]:",trace_list[0])
        j_list = [A+1 for A in range(len(trace_list[0]))]
        T_list = []
        for P in x_hpc:#x_true:
            T_list_single = []
            for C in range(len(trace_list[0])):
                c1 = trace_list[0][C]
                c2 = trace_list[1][C]
                c3 = trace_list[2][C]
                c4 = trace_list[3][C]
                c5 = trace_list[4][C]
                if l:
                    if P>100:
                        term5 = 0.0
                    else:
                        term5 = (c5*np.log(P))/(np.sqrt(P))
                    T_list_single.append((c4 / P**2.0) + (c1 / P) + c2 + (c3 * np.log(P)) + term5)
                else:
                    T_list_single.append((c4 / P**2.0) + (c1 / P) + c2 + (c3 * P) + c5*(np.log(P)/np.sqrt(P)))
            T_list.append(T_list_single)

        print("len(trace_list[0])",len(trace_list[0]))
        print("trace_list[0]")
        print(trace_list[0])
        f1 = open('./%s/trace.txt' % i,"w")
        f1.write("#j c1 c2 c3 c4 c5")
        for P in x_hpc:#x_true:
            f1.write(" %s"% int(P))
        f1.write("\n")
        for C in range(len(trace_list[0])):
            f1.write(str(j_list[C]))
            f1.write(" ")
            f1.write(str(trace_list[0][C]))
            f1.write(" ")
            f1.write(str(trace_list[1][C]))
            f1.write(" ")
            f1.write(str(trace_list[2][C]))
            f1.write(" ")
            f1.write(str(trace_list[3][C]))
            f1.write(" ")
            f1.write(str(trace_list[4][C]))
            for CC in range(len(T_list)):
                f1.write(" ")
                f1.write(str(T_list[CC][C]))
            f1.write("\n")
        f1.close()

        sys.stdout = open('./%s/detail.txt' % i, 'w')
        mcmc.summary()
        sys.stdout.close()
        sys.stdout = sys.__stdout__

        c1_median = np.median(mcmc.trace('c1', chain=None)[:])
        c2_median = np.median(mcmc.trace('c2', chain=None)[:])
        c3_median = np.median(mcmc.trace('c3', chain=None)[:])
        c4_median = np.median(mcmc.trace('c4', chain=None)[:])
        c5_median = np.median(mcmc.trace('c5', chain=None)[:])

        c1_min = mcmc.stats()['c1']['95% HPD interval'][0]
        c1_max = mcmc.stats()['c1']['95% HPD interval'][1]
        c2_min = mcmc.stats()['c2']['95% HPD interval'][0]
        c2_max = mcmc.stats()['c2']['95% HPD interval'][1]
        c3_min = mcmc.stats()['c3']['95% HPD interval'][0]
        c3_max = mcmc.stats()['c3']['95% HPD interval'][1]
        c4_min = mcmc.stats()['c4']['95% HPD interval'][0]
        c4_max = mcmc.stats()['c4']['95% HPD interval'][1]
        c5_min = mcmc.stats()['c5']['95% HPD interval'][0]
        c5_max = mcmc.stats()['c5']['95% HPD interval'][1]
        y_pre = []
        y_pre_min = []
        y_pre_max = []

        for P in x_hpc:#x_true:
            if l:
                if P>100:
                    term5_median = 0.0
                    term5_min = 0.0
                    term5_max = 0.0
                else:
                    term5_median = (c5_median*np.log(P))/(np.sqrt(P))
                    term5_min = (c5_min*np.log(P))/(np.sqrt(P))
                    term5_max = (c5_max*np.log(P))/(np.sqrt(P))
                y_pre.append((c4_median / P**2.0) + (c1_median / P) + c2_median + (c3_median * np.log(P)) + term5_median)
                y_pre_min.append((c4_min / P**2.0) + (c1_min / P) + c2_min + (c3_min * np.log(P)) + term5_min)
                y_pre_max.append((c4_max / P**2.0) + (c1_max / P) + c2_max + (c3_max * np.log(P)) + term5_max)
            else:
                y_pre.append((c4_median / P**2.0) + (c1_median / P) + c2_median + (c3_median * P) + c5_median*(np.log(P)/np.sqrt(P)))
                y_pre_min.append((c4_min / P**2.0) + (c1_min / P) + c2_min + (c3_min * P) + c5_min*(np.log(P)/np.sqrt(P)))
                y_pre_max.append((c4_max / P**2.0) + (c1_max / P) + c2_max + (c3_max * P) + c5_max*(np.log(P)/np.sqrt(P)))

        data_list.append([y_true, y_pre, i])
        plt.plot(x_true, y_true, ls='-', lw=1, label='True', marker='o')
        plt.plot(x_hpc, y_pre, label='Predict (median of c1, c2, c3, c4, c5)', marker='o')
        #plt.fill_between(x_true, y_pre_min, y_pre_max, color='r', alpha=0.1, label='95% HPD interval of c1, c2, c3, c4, c5')
        # 95% HPD interval
        #---------------------------------------------------------------------------------------------------------------------#
        param = int(0.) # [c1, c2, c3, c4, c5, 128, 256, 512, 1024, 2048, 4096, 8192]
        total_num = int(7.) # [128, 256, 512, 1024, 2048, 4096, 8192]
        length = 5000 # iter=100000, burn=50000, thin=10
        hpd = (100. - 95.) / 2. # 95% HPD interval
        bottom_line = length * (hpd/100.)
        top_line = length * (100.-hpd)/100.
        #
        node_data = []
        for iter_i in xrange(param+1, param+5+total_num+1):
            routine_list = []
            with open("Total/trace.txt") as fp:
                comment = fp.readline()
                for line in fp:
                    ss = line.split()
                    routine_list.append(float(ss[iter_i]))
                node_data.append(routine_list)

        hpd_interval = []
        for iter_i in xrange(len(node_data)):
            node_data[iter_i].sort()
            hpd_interval.append(node_data[iter_i][int(bottom_line):int(top_line)])
        hpd_min = []
        hpd_max = []
        for iter_i in xrange(len(hpd_interval)):
            hpd_min += [(min(hpd_interval[iter_i]))]
            hpd_max += [(max(hpd_interval[iter_i]))]
        print "hpd_min= ", hpd_min
        print "hpd_max= ", hpd_max
        plt.fill_between(x_hpc, hpd_min[param+5:], hpd_max[param+5:], color='r', alpha=0.1, label='95% HPD interval')
        #---------------------------------------------------------------------------------------------------------------------#
        plotting(plt, './%s/graph.png' % i, i, '(%s)' % i, x_true, [min(y_pre_min), max(y_pre_max)], x_hpc)
        f2 = open('./%s/text.txt' % i,'w')
        f2.write("#x_true y_true y_pre y_95%HPD_min y_95%HPD_max")
        f2.write("\n")
        for k in range(len(x_hpc)):
            f2.write(str(x_hpc[k]))
            f2.write(" ")
            if k < (int(len(x_hpc)) - (int(len(x_hpc)) - int(len(x_true)))):
                f2.write(str(y_true[k]))
            else:
                f2.write("0")
            f2.write(" ")
            f2.write(str(y_pre[k]))
            f2.write(" ")
            ##f2.write(str(y_pre_min[k]))
            f2.write(str(hpd_min[k+5]))
            f2.write(" ")
            ##f2.write(str(y_pre_max[k]))
            f2.write(str(hpd_max[k+5]))
            f2.write(" ")
            #f2.write(str(c1_median))
            #f2.write(" ")
            #f2.write(str(hpd_min[k]))
            #f2.write(" ")
            #f2.write(str(hpd_max[k]))
            #f2.write(" ")
            f2.write("\n")
        f2.write("#c1_median c1_95HPD_min c1_95HPD_max \n")
        f2.write("%s %s %s\n" % (c1_median, hpd_min[0], hpd_max[0]))
        f2.write("#c2_median c2_95HPD_min c2_95HPD_max \n")
        f2.write("%s %s %s\n" % (c2_median, hpd_min[1], hpd_max[1]))
        f2.write("#c3_median c3_95HPD_min c3_95HPD_max \n")
        f2.write("%s %s %s\n" % (c3_median, hpd_min[2], hpd_max[2]))
        f2.write("#c4_median c4_95HPD_min c4_95HPD_max \n")
        f2.write("%s %s %s\n" % (c4_median, hpd_min[3], hpd_max[3]))
        f2.write("#c5_median c5_95HPD_min c5_95HPD_max \n")
        f2.write("%s %s %s\n" % (c5_median, hpd_min[4], hpd_max[4]))
        f2.close()

    cmap = plt.get_cmap("tab10")
    for jjj in range(len(data_list)):
        if data_list[jjj][2] == "Total":
            plt.plot(x_true, data_list[jjj][0], label='True (total)', marker='o', color=cmap(0))
            plt.plot(x_hpc, data_list[jjj][1], label='Predict (total)', marker='o', color=cmap(0), linestyle='--')
        if data_list[jjj][2] == "pdsytrd":
            plt.plot(x_true, data_list[jjj][0], label='True (pdsytrd)', marker='o', color=cmap(1))
            plt.plot(x_hpc, data_list[jjj][1], label='Predict (pdsytrd)', marker='o', color=cmap(1), linestyle='--')
        if data_list[jjj][2] == "pdsygst":
            plt.plot(x_true, data_list[jjj][0], label='True (pdsygst)', marker='o', color=cmap(2))
            plt.plot(x_hpc, data_list[jjj][1], label='Predict (pdsygst)', marker='o', color=cmap(2), linestyle='--')

    combined = [0.0 for i in range(len(x_true))]
    for kkk in range(len(data_list)):
        if not data_list[kkk][2] == "Total":
                for ll in range(len(combined)):
                    combined[ll] += data_list[kkk][1][ll]
    plt.plot(x_true, combined, label='Predict(added ALL)', marker='o', color=cmap(4), linestyle='--')
    plotting(plt, './graph_predict.png', 'Predicting Total, pdsytrd and pdsygst', '', x_true, y_range, x_hpc)
    f3 = open('./text_predict.txt',"w")
    f3.write("#x_true combined(added ALL)")
    f3.write("\n")
    for kk in range(len(x_true)):
        f3.write(str(x_true[kk]))
        f3.write(" ")
        f3.write(str(combined[kk]))
        f3.write("\n")
    f3.close()


def plotting(plt, path, title, y_label, node, y_range, x_hpc):
    plt.xlim(x_hpc[0], x_hpc[-1])
    plt.xticks(node)

    if not y_range[1] == 'default':
        plt.ylim(y_range[0], y_range[1])

    plt.legend()
    plt.xscale('log', basex=2)
    plt.yscale('log', basey=10)

    plt.xlabel("Number of CPU (P)")
    plt.ylabel('Elapse Time %s [sec]' % y_label)

    plt.title('%s' % title)
    plt.grid(which='major', linestyle='-')
    plt.grid(which='minor', linestyle=':')

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.subplots_adjust(right=0.5)
    plt.savefig('%s' % path, dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('-yr', '--y_range',
                        nargs = 2,
                        type = float,
                        default = [10, 'default'],
                        help = 'y range (min max)')

    parser.add_argument('-l', '--log',
                        action = 'store_true',
                        help = 'log flag (default = False)')

    args = parser.parse_args()
    y_range = args.y_range
    l = args.log
    main(l, y_range)

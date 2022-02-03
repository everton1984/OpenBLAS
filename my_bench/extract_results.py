#!python
import json
import os
import matplotlib.pyplot as plt
import argparse
import html
import hashlib
from datetime import datetime
from operator import itemgetter

def gen_graphs(benchmarks):
    for bench in benchmarks.keys():
        if 'median' in bench:
            R = [ v['cpu_time'] for v in benchmarks[bench] ]
            IDs = [ v['id'] for v in benchmarks[bench] ]
            fname = "out/" + bench.replace('/','_').replace('_median','') + ".png"
            plt.plot(IDs, R)
            plt.title(bench.replace('/','_').replace('_median',''))
            plt.xlabel("Commit (ordered in ascending date)")
            plt.xticks(rotation=45, ha='right')
            plt.ylabel("Time")
            plt.tight_layout()
            plt.savefig(fname)
            plt.clf()

def clean_data(benchmarks):
    res = {}
    for bench in benchmarks.keys():
        if 'median' in bench:
            clean_key = bench.replace('_median','')
            res[clean_key] = []
            res[clean_key] = [ v['cpu_time'] for v in benchmarks[bench]]
    return res

def gen_diff(benchmarks):
    res = clean_data(benchmarks)
    for resKey in res.keys():
        R = res[resKey]
        R0 = [0] + R[:-1]
        Rdiff = [y/x for x, y in zip(R,R0)][1:]
        total = R[0]/R[-1]
        res[resKey] = {'timeseries':Rdiff, 'total': total}
    return res

# Load commit list
commit_list = open("commit_list").readlines()
commit_list = [x.strip() for x in commit_list]

# Get commit dates
stream = os.popen("git log --no-decorate --pretty=format:'%h %as'")
streamLines = stream.readlines()
L = {x.strip().split()[0]:datetime.strptime(x.strip().split()[1],'%Y-%m-%d') for x in streamLines}

commit_with_date = [[x,L[x]] for x in commit_list]
commit_with_date = sorted(commit_with_date, key=itemgetter(1))
commit_list = [x[0] for x in commit_with_date]

# Extract data
benchmarks = {}
for commit in commit_list:
    fname = commit + '/out.json'

    with open(fname, 'r') as file:
        rjson = json.loads(file.read())
        for bb in rjson['benchmarks']:
            if not bb['name'] in benchmarks.keys():
                benchmarks[bb['name']] = []
            benchmarks[bb['name']].append({'id': commit, 'cpu_time': bb['cpu_time']})

# bDiffs = gen_diff(benchmarks)
# for key in bDiffs:
#     m = min(bDiffs[key]['timeseries'])
#     M = bDiffs[key]['timeseries'][-1]
#     if m != M and M < m:
#         print(key, v, m, M)

res = clean_data(benchmarks)
for key in res.keys():
    T = res[key]
    m = min(T)
    M = T[-1]
    v = m / M
    if m < M and v < 0.98:
        uncleanKey = key+'_mean'
        cid = ''
        for elem in benchmarks[uncleanKey]:
            time = elem['cpu_time']
            if elem['cpu_time'] == m:
                cid = elem['id']
        print(key.replace('BM_',''), ',', "%.0f" % ((1-v)*100) + "%",cid, time)


# benchmarkInfo = {}

# def readBenchmark(bRoot,name):
#     global benchmarkInfo
#     benchmarkResultsJSON = []

#     for root, dirs, files in os.walk(bRoot):
#         for fname in files:
#             if fname.endswith('.json'):
#                 p = os.path.join(root,fname)
#                 fnameSplit = fname.split('_')
#                 dt = int(fnameSplit[3])
#                 commit = fnameSplit[2]
#                 benchmarkResultsJSON.append([dt,p,commit])
#     benchmarkResultsJSON = sorted(benchmarkResultsJSON)

#     infoFname = benchmarkResultsJSON[0][1].replace('gemm','info').replace('.json','')
#     with open(infoFname, 'r') as file:
#         info = file.read().split('Commit')[1].split('\n')[1:]
#         benchmarkInfo[name] = info

#     benchmarks = {}
#     for benchmark in benchmarkResultsJSON:
#         with open(benchmark[1], 'r') as file:
#             fjson = file.read().replace('\n','')

#             j = json.loads(fjson)
#             for bench in j["benchmarks"]:
#                 key = bench["name"]
#                 value = float(bench["cpu_time"])
#                 if bench["name"] in benchmarks:
#                     benchmarks[key]["times"].append(value)
#                     benchmarks[key]["dates"].append(str(benchmark[0]))
#                     benchmarks[key]["commits"].append(benchmark[2])
#                 else:
#                     benchmarks[key] = {"times":[value], "dates":[str(benchmark[0])],"commits":[benchmark[2]]}
#     return benchmarks

# def getSpeedups(data,data2=None):
#     s = []
#     t0 = data[0]
#     if not data2:
#         for d in data[1:]:
#             s.append(float(t0/d))
#             t0 = d
#     else:
#         for i in range(len(data)):
#             s.append(float(data[i]/data2[i]))
#     return s

# def getDate(fromStr):
#     return {"year": fromStr[0:4], "month":fromStr[4:6], "day": fromStr[6:8]}

# def reportSingle(bench,name,img_root=None):
#     benchmarks = readBenchmark(bench,name)
#     report = {}
#     for key in benchmarks.keys():
#         bDates = benchmarks[key]["dates"]
#         bTimes = benchmarks[key]["times"]
#         safeKey = str(hashlib.md5(key.encode()).hexdigest())
#         fname = ""

#         if img_root:
#             ## Plot graphs
#             fname = "{}/{}.png".format(img_root, safeKey)
#             if not os.path.isfile(fname):
#                 plt.plot(bDates, bTimes)
#                 plt.title(key)
#                 plt.xlabel("Date")
#                 plt.ylabel("Time (ns)")
#                 plt.savefig(fname)
#                 plt.clf()

#         speedups = getSpeedups(bTimes)
#         dtList = []
#         for i in range(1,len(bTimes)):
#             dt = getDate(bDates[i])
#             dtList.append(dt["month"]+"/"+dt["day"]+"/"+dt["year"])

#         sMin = min(speedups)
#         sMax = max(speedups)
#         report[key] = {"title": key,
#                        "id":safeKey,
#                        "speedups": speedups,
#                        "date": dtList,
#                        "min": sMin,
#                        "max": sMax,
#                        "accumulatedSpeedup": bTimes[0] / bTimes[len(bTimes)-1],
#                        "commits": benchmarks[key]["commits"][1:]}
#         if img_root:
#             report["figure"] = fname

#     return report

# def reportCompare(bench1, bench2, label1, label2, name1, name2, img_root=None):
#     benchmarks1 = readBenchmark(bench1, name1)
#     benchmarks2 = readBenchmark(bench2, name2)

#     report = {}
#     for key in benchmarks1.keys():
#         if key in benchmarks2.keys():
#             bDates = benchmarks1[key]["dates"]
#             bTimes1 = benchmarks1[key]["times"]
#             bTimes2 = benchmarks2[key]["times"]

#             ## Use only available data
#             size = min(len(bTimes1),len(bTimes2))
#             bTimes1 = bTimes1[:size]
#             bTimes2 = bTimes2[:size]
#             bDates = bDates[:size]
#             safeKey = str(hashlib.md5(key.encode()).hexdigest())
#             fname=""

#             if img_root:
#                 ## Plot graphs
#                 fname = "{}/{}.png".format(img_root, safeKey)
#                 if not os.path.isfile(fname):
#                     plt.plot(bDates, bTimes1, label=label1)
#                     plt.plot(bDates, bTimes2, label=label2)
#                     plt.title(key)
#                     plt.xlabel("Date")
#                     plt.ylabel("Time (ns)")
#                     plt.legend()
#                     plt.savefig(fname)
#                     plt.clf()

#             speedups = getSpeedups(bTimes1,bTimes2)
#             dtList = []
#             for i in range(len(bTimes1)):
#                 dt = getDate(bDates[i])
#                 dtList.append(dt["month"]+"/"+dt["day"]+"/"+dt["year"])

#             sMin = min(speedups)
#             sMax = max(speedups)
#             report[key] = {"title": key,
#                         "id":safeKey,
#                         "speedups": speedups,
#                         "date": dtList,
#                         "min": sMin,
#                         "max": sMax,
#                         "accumulatedSpeedup": speedups[-1],
#                         "commits": benchmarks1[key]["commits"]}
#             if img_root:
#                 report["figure"] = fname

#     return report

# def changeColor(txt, color):
#     return '<p style="color:{1};font-weight:bold;">{0}</p>'.format(txt,color)

# def htmlIndex(report):
#     s = "<table class='tg'>"
#     s += "<tr><th class='tg-0lax'><b>Test</b></th><th class='tg-0lax'><b>Accumulated speedup</b></th></tr>"
#     for key in report.keys():
#         singleReport = report[key]
#         s += '<tr><td><a href="#{}">{}</a></td><td>'.format(singleReport["id"], html.escape(singleReport["title"]))
#         acc = singleReport["accumulatedSpeedup"]
#         accStr = '{:.2f}'.format(acc)
#         if acc > 1.05:
#             s += changeColor(accStr,'green')
#         elif acc < 0.96:
#             s += changeColor(accStr,'red')
#         else:
#             s += changeColor(accStr,'blue')
#         s += "</td></tr>"
#     s += "</table>"
#     s += "<br/>"
#     return s

# def singleToHtml(singleReport):
#     s = "<h2 style='text-align:center;'><section id={}><b>{}</b></section></h2>".format(singleReport["id"], html.escape(singleReport["title"]))
#     s += "<table class='tg'>"
#     speedups = singleReport["speedups"]
#     dates = singleReport["date"]
#     commits = singleReport["commits"]
#     s += "<tr><th class='tg-0lax'><b>Speedup</b></th><th class='tg-0lax'>Commit</th><th class='tg-0lax'><b>Date</b></th></tr>"
#     for i in range(len(speedups)):
#         s += '<tr><td>{0:.2f}</td><td><a href="https://gitlab.com/libeigen/eigen/-/commit/{1}">{1}</a></td><td>{2}</td></tr>'.format(speedups[i],commits[i],dates[i])
#     acc = singleReport["accumulatedSpeedup"]
#     accStr = '{:.2f}'.format(acc)
#     s += "<tr><td>Accumulated speedup</td><td colspan='2'>"
#     if acc > 1.05:
#         s += changeColor(accStr,'green')
#     elif acc < 0.96:
#         s += changeColor(accStr,'red')
#     else:
#         s += changeColor(accStr,'blue')
#     s += "</td></tr>"

#     if "figure" in singleReport:
#         s += "<tr><td colspan='3'><img src='{}'/></td></tr>".format(singleReport["figure"])

#     s += "</table>"
#     s += "<br/>"

#     return s

# def htmlHeader(title, bRoots):
#     s = "<!DOCTYPE html><html><head><title>" + title + "</title></head>"
#     s += '<style type="text/css"> \
#             .tg  {border-collapse:collapse;border-color:#ccc;border-spacing:0;margin-left:auto;margin-right:auto;} \
#             .tg td{background-color:#fff;border-color:#ccc;border-style:solid;border-width:1px;color:#333; \
#             font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;word-break:normal;} \
#             .tg th{background-color:#f0f0f0;border-color:#ccc;border-style:solid;border-width:1px;color:#333; \
#             font-family:Arial, sans-serif;font-size:14px;font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;} \
#             .tg .tg-0lax{text-align:left;vertical-align:top} \
#           </style>'
#     s += "<body><h1 style='text-align:center;'>" + title + "</h1>"
#     for root in bRoots:
#         s += "<h4 style='text-align:center;'><a href='#info{0}'>Info {0}</a></h4>".format(root)
#     return s

# def htmlFooter(bRoots):
#     s = ""
#     for root in bRoots:
#         s += "<section id='info{}'><h2 style='text-align:center;'>Info {}</h2>".format(root, root.split('/')[-1])
#         s += "<table class='tg'>"
#         for el in benchmarkInfo[root]:
#             elsplit = el.split(':')
#             if len(elsplit) > 1:
#                 s += "<tr><th class='tg-0lax'>{}</th><td>{}</td></tr>".format(elsplit[0].strip(), elsplit[1].strip())
#         s += "</table>"
#         s += "</section>"
#     s += "</body></html>"
#     return s

# def getHtmlReportSingle(name, root,img_root=None):
#     htmlStr = ""

#     report = reportSingle(root, name, img_root=img_root)
#     htmlStr += htmlHeader(name, [root])

#     htmlStr += htmlIndex(report)
#     for key in report.keys():
#         htmlStr += singleToHtml(report[key])

#     htmlStr += htmlFooter([name])

#     return htmlStr

# def getHtmlReportCompare(name1, name2, root1, root2,img_root=None):
#     htmlStr = ""

#     name = '{} vs {}'.format(name1, name2)
#     report = reportCompare(root1, root2, name1, name2, name1, name2, img_root=img_root)
#     htmlStr += htmlHeader(name, [name1, name2])

#     htmlStr += htmlIndex(report)
#     for key in report.keys():
#         htmlStr += singleToHtml(report[key])

#     htmlStr += htmlFooter([name1, name2])

#     return htmlStr

# description = 'Compare benchmark sets.'

# parser = argparse.ArgumentParser(description=description)
# parser.add_argument('--name', help='Name on benchmark report.', required=True)
# parser.add_argument('--root', help='Root directory for benchmark set.', required=True)
# parser.add_argument('--name_vs', help='Name on contender benchmark report.')
# parser.add_argument('--root_vs', help='Root directory for contender benchmark set.')
# parser.add_argument('--img_root', help='Root directory for images, if not set no graphs will be generated.')
# args = parser.parse_args()


# if args.name_vs and args.root_vs:
#     print(getHtmlReportCompare(args.name, args.name_vs, args.root, args.root_vs, img_root=args.img_root))
# else:
#     print(getHtmlReportSingle(args.name, args.root, img_root=args.img_root))

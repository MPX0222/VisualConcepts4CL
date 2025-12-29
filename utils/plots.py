import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def inference(home):
    plt.rc("font", family="Times New Roman")
    # 示例数据
    categories = ["CIFAR100", "ImageNet-R"]  # 横坐标为非数值标签
    series1 = [80.06, 82.0]  # 第二组数据
    series2 = [81.6, 82.51]  # 第一组数据
    series3 = [81.55, 82.84]
    series4 = [82.52, 83.58]
    std1 = [0.33, 0.39]
    std2 = [0.33, 0.25]
    std3 = [0.25, 0.47]
    std4 = [0.19, 0.31]
    # 横坐标索引
    x = np.arange(len(categories))
    colors = ['#8FBAC8',  # 柔和蓝色
              '#A9CDA4',  # 浅绿色
              '#F4CDA5',  # 米黄色/柔和橙
              '#E8AFAF']
    # 设置每组柱子的宽度
    bar_width = 0.2
    error_attri = {"elinewidth": 2, "ecolor": "black", "capsize": 8}
    # 绘制双系列柱状图
    fig, ax = plt.subplots()
    bars1 = ax.bar(x - bar_width * 1.5, series1, width=bar_width, yerr=std1, error_kw=error_attri, label='Energy', color=colors[0])
    bars2 = ax.bar(x - bar_width * 0.5, series2, width=bar_width, yerr=std2, error_kw=error_attri, label='Max', color=colors[1])
    bars3 = ax.bar(x + bar_width * 0.5, series3, width=bar_width, yerr=std3, error_kw=error_attri, label='Entropy', color=colors[2])
    bars4 = ax.bar(x + bar_width * 1.5, series4, width=bar_width, yerr=std4, error_kw=error_attri, label='Entropy-m', color=colors[3])
    for i in range(len(categories)):
        plt.text(x[i] - bar_width * 1.5, series1[i]+std1[i], str(round(series1[i], 2)), ha="center", va="bottom", fontsize=15)
        plt.text(x[i] - bar_width * 0.5, series2[i]+std2[i], "+" + str(round(series2[i] - series1[i], 2)), ha="center",
                 va="bottom", fontsize=15)
        plt.text(x[i] + bar_width * 0.5, series3[i]+std3[i], "+" + str(round(series3[i] - series1[i], 2)), ha="center",
                 va="bottom", fontsize=15)
        plt.text(x[i] + bar_width * 1.5, series4[i]+std4[i], "+" + str(round(series4[i] - series1[i], 2)), ha="center",
                 va="bottom", fontsize=15)
    # 设置横坐标为非数值的分类标签
    plt.tick_params(labelsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=17)
    # 添加标题和标签
    ax.set_ylim(78, 88)
    ax.set_ylabel('Last Accuracy/MCR(%)', fontsize=17)
    # ax.set_title('Dual Bar Chart with Categorical X-axis')
    # 添加图例
    ax.legend(fontsize=17, loc="upper left")
    # 显示图表
    # plt.show()
    plt.savefig(home+"/inference.png", dpi=400, bbox_inches="tight")


def proj1(home):
    import numpy as np
    import matplotlib.pyplot as plt
    plt.rc("font", family="Times New Roman")
    # 示例数据
    # categories = ["CIFAR100", "IN-R", "mini-IN100", "Cars196"]  # 横坐标为非数值标签
    # series1 = [81.16, 81.92, 93.10, 80.25]  # 第一组数据
    # series2 = [82.52, 83.25, 93.83, 81.52]  # 第二组数据
    # std1 = [0.43, 0.39, 0.25, 0.57]
    # std2 = [0.33, 0.45, 0.19, 0.35]

    categories = ["CIFAR100", "ImageNet-R"]  # 横坐标为非数值标签
    series1 = [81.16, 81.92]  # 第一组数据
    series2 = [82.52, 83.25]  # 第二组数据
    std1 = [0.43, 0.39]
    std2 = [0.33, 0.45]

    # 横坐标索引
    x = np.arange(len(categories))
    colors = ['#8FBAC8',  # 柔和蓝色
              # '#A9CDA4',  # 浅绿色
              '#F4CDA5',  # 米黄色/柔和橙
              # '#E8AFAF'
              ]
    # 设置每组柱子的宽度
    bar_width = 0.4
    error_attri = {"elinewidth": 2, "ecolor": "black", "capsize": 14}
    # 绘制双系列柱状图
    fig, ax = plt.subplots()
    bars1 = ax.bar(x - bar_width * 0.5, series1, width=bar_width, yerr=std1, error_kw=error_attri, label='single projector', color=colors[0])
    bars2 = ax.bar(x + bar_width * 0.5, series2, width=bar_width, yerr=std2, error_kw=error_attri, label='MoP', color=colors[1])
    # bars3 = ax.bar(x + bar_width*0.5, series3, width=bar_width, label='Entropy', color=colors[2])
    # bars4 = ax.bar(x + bar_width*1.5, series4, width=bar_width, label='Entropy-m', color=colors[3])
    for i in range(len(categories)):
        plt.text(x[i] - bar_width * 0.5, series1[i]+std1[i], str(round(series1[i], 2)), ha="center", va="bottom", fontsize=20)
        plt.text(x[i] + bar_width * 0.5, series2[i]+std2[i], "+" + str(round(series2[i] - series1[i], 2)), ha="center",
                 va="bottom", fontsize=20)
    # 设置横坐标为非数值的分类标签
    plt.tick_params(labelsize=17)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=17)
    # 添加标题和标签
    ax.set_ylim(75, 88)
    ax.set_ylabel('Last Accuracy/MCR(%)', fontsize=17)
    # ax.set_title('Dual Bar Chart with Categorical X-axis')
    # 添加图例
    ax.legend(fontsize=17, loc="upper left")
    # 显示图表
    # plt.show()
    plt.savefig(home+"/1proj.png", dpi=400, bbox_inches="tight")


def mm(home):
    plt.rc("font", family="Times New Roman")
    red = "#D4242A"
    blue = "#2481C4"
    x1 = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    y1 = [83.25, 83.58, 83.33, 83.19, 83.105, 82.89, 82.72, 82.87, 82.56, 82.32]
    y2 = [81.22, 81.52, 81.39, 81.22, 81.33, 81.14, 80.78, 80.89, 80.43, 80.2]
    fig = plt.figure()
    ax1 = fig.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x1, y1, color=red, marker="o", label="Ours on ImageNet-R")
    ax1.axhline(81.77, color=red, linestyle="-.", label="Lang-CL on ImageNet-R")
    # plt.axhline(76.21, color="red", linestyle="--", label="SLCA on ImageNet-R")
    # plt.axhline(66.82, color="blue", linestyle="--", label="SLCA on Cars196")
    ax2.plot(x1, y2, color=blue, marker="^", label="Ours on Cars196")
    ax2.axhline(78.07, color=blue, linestyle="--", label="Lang-CL on Cars196")
    ax1.set_ylim((65, 88))
    ax2.set_ylim((70, 90))
    ax1.tick_params(axis="x", labelsize=15)
    # 设置轴标签颜色
    ax1.tick_params('y', colors=red, labelsize=15)
    ax2.tick_params('y', colors=blue, labelsize=15)
    # 设置轴颜色
    ax1.spines['left'].set_color(red)
    ax2.spines['left'].set_color(red)
    ax1.spines['right'].set_color(blue)
    ax2.spines['right'].set_color(blue)

    ax1.set_xlabel("m", fontsize=17)
    ax1.set_ylabel("Last MCR on ImageNet-R (%)", fontsize=20, color=red)
    ax2.set_ylabel("Last MCR on Cars196 (%)", fontsize=20, color=blue)
    ax1.legend(fontsize=13, loc="lower left")
    ax2.legend(fontsize=13, loc="lower right")
    ax1.set_xticks([0, 0.02, 0.04, 0.06, 0.08, 0.1])
    ax2.set_xticks([0, 0.02, 0.04, 0.06, 0.08, 0.1])

    # plt.show()
    plt.savefig(home+"/mm.png", dpi=400, bbox_inches="tight")


def M(home):
    plt.rc("font", family="Times New Roman")
    red = "#D4242A"
    blue = "#2481C4"
    x1 = [2, 3, 4, 5, 6, 7, 8]
    y1 = [83.25, 82.91, 83.06, 82.751, 82.562, 82.19, 82.05]
    y2 = [81.35, 81.82, 81.67, 81.82, 81.05, 80.39, 80.84]
    fig = plt.figure()
    ax1 = fig.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x1, y1, color=red, marker="o", label="Ours on ImageNet-R")
    ax1.axhline(81.77, color=red, linestyle="-.", label="Lang-CL on ImageNet-R")
    # plt.axhline(76.21, color="red", linestyle="--", label="SLCA on ImageNet-R")
    # plt.axhline(66.82, color="blue", linestyle="--", label="SLCA on Cars196")
    ax2.plot(x1, y2, color=blue, marker="^", label="Ours on Cars196")
    ax2.axhline(78.07, color=blue, linestyle="--", label="Lang-CL on Cars196")
    ax1.set_ylim((70, 88))
    ax2.set_ylim((70, 90))
    ax1.tick_params(axis="x", labelsize=15)
    # 设置轴标签颜色
    ax1.tick_params('y', colors=red, labelsize=15)
    ax2.tick_params('y', colors=blue, labelsize=15)
    # 设置轴颜色
    ax1.spines['left'].set_color(red)
    ax2.spines['left'].set_color(red)
    ax1.spines['right'].set_color(blue)
    ax2.spines['right'].set_color(blue)

    ax1.set_xlabel("M", fontsize=17)
    ax1.set_ylabel("Last MCR on ImageNet-R (%)", fontsize=20, color=red)
    ax2.set_ylabel("Last MCR on Cars196 (%)", fontsize=20, color=blue)
    ax1.legend(fontsize=13, loc="lower left")
    ax2.legend(fontsize=13, loc="lower right")

    # plt.show()
    plt.savefig(home+"/M.png", dpi=400, bbox_inches="tight")


def results(home):
    markers = {"PROOF": "1",
               "PR-CLIP": "h",
               "Continual-CLIP": "8",
               "L2P": "x",
               "DualPrompt": "d",
               "CODA-Prompt": "P",
               "AttriCLIP": "*",
               "EASE": "s",
               "MoE-Adapters": "p",
               "SLCA": "v",
               "Lang-CL": "|",
               "Ours": ".",
               }

    colors = {"PROOF": "#1c78b4",
               "PR-CLIP": "#f68304",
               "Continual-CLIP": "#29a128",
               "L2P": "#cf3736",
               "DualPrompt": "#9264b3",
               "CODA-Prompt": "#89564f",
               "AttriCLIP": "#00FF97",
               "EASE": "#e276c9",
               "MoE-Adapters": "#808080",
               "SLCA": "#b9bb4c",
               "Lang-CL": "#1ebcc9",
               "Ours": "#E90D62",
               }

    datasets = ["ImageNet-R", "CIFAR100", "Cars196", "Skin40", "ImageNet100"]
    plt.rc("font", family="Times New Roman")
    for dataset_name in datasets:
        data = pd.read_excel(home+"/1.xlsx", sheet_name=dataset_name)
        tasks = np.unique(data["task"])
        for t in tasks:
            selected_data = data[data["task"] == t].dropna(axis=1)
            methods = np.array(selected_data["methods"])
            plt.figure()
            # plt.axes(facecolor="#EAEAF2")
            x = np.arange(1, t + 1)
            for i in range(len(selected_data)):
                y = np.array(selected_data.iloc[i, 4:])
                plt.plot(x, y, label=methods[i], marker=markers[methods[i]], color=colors[methods[i]])
            plt.xticks(x)
            plt.title(dataset_name+" "+str(t)+"-task", fontsize=25)
            plt.tick_params(labelsize=15)
            plt.xlabel("Number of Tasks", fontsize=25)
            plt.ylabel("MCR(%)" if dataset_name in ["ImageNet-R", "Cars196"] else "Accuracy(%)", fontsize=25)
            if dataset_name == "Skin40":
                plt.ylim((10, 100))
            elif dataset_name == "ImageNet100":
                plt.ylim((85, 100))
            elif dataset_name == "Cars196":
                plt.ylim((55, 100))
            elif dataset_name == "ImageNet-R":
                plt.ylim((60, 100))
            else:
                plt.ylim((50, 100))
            plt.grid(linestyle="--", linewidth=1)
            plt.legend(fontsize=10, ncol=2, loc="lower left")
            # plt.show()
            plt.savefig(home+"/curve/"+dataset_name+f"_{t}task.png", dpi=400, bbox_inches="tight")


def param_ir(home):
    plt.rc("font", family="Times New Roman")
    methods = ["CODA-Prompt", "DualPrompt", "EASE", "L2P", "MoE-Adapters", "PROOF", "SLCA", "Ours", "Lang-CL"]
    mcr = [72.47, 76.94, 76.72, 75.55, 78.68, 78.19, 79.54, 84.02, 82.46]
    param = [0.04428, 0.00467, 0.06436, 0.00734, 0.0258, 0.02395, 1, 0.038742, 0.037926]
    markers = {"PROOF": "v",
               "Continual-CLIP": "8",
               "L2P": "X",
               "DualPrompt": "d",
               "CODA-Prompt": "P",
               "EASE": "s",
               "MoE-Adapters": "p",
               "SLCA": "D",
               "Lang-CL": "h",
               "Ours": "*",
               }
    colors = {"PROOF": "#1c78b4",
               "PR-CLIP": "#f68304",
               "Continual-CLIP": "#29a128",
               "L2P": "#cf3736",
               "DualPrompt": "#9264b3",
               "CODA-Prompt": "#89564f",
               "AttriCLIP": "#00FF97",
               "EASE": "#e276c9",
               "MoE-Adapters": "#808080",
               "SLCA": "#b9bb4c",
               "Lang-CL": "#1ebcc9",
               "Ours": "#E90D62",
               }
    plt.figure()
    # plt.grid(linestyle="--", linewidth=1)
    plt.axvline(0.038742, color="grey", linestyle="--", linewidth=1, zorder=1)
    for i in range(len(methods)):
        if i != 1 and i != 5:
            plt.scatter(param[i], mcr[i], marker=markers[methods[i]], color=colors[methods[i]], label=methods[i], s=300 if methods[i]!="Ours" else 400)
    for i in range(len(methods)):
        if i != 1 and i != 5:
            if methods[i] == "SLCA":
                plt.annotate(methods[i], xy=(param[i], mcr[i]), xytext=(param[i]-0.4, mcr[i] - 1.5), fontsize=15)
            elif methods[i] == "Lang-CL":
                plt.annotate(methods[i], xy=(param[i], mcr[i]), xytext=(param[i]+0.005, mcr[i]-0.5), fontsize=15)
            elif methods[i] == "MoE-Adapters":
                plt.annotate(methods[i], xy=(param[i], mcr[i]), xytext=(param[i] - 0.015, mcr[i] - 1.3), fontsize=15)
            elif methods[i] == "Ours":
                plt.annotate(methods[i], xy=(param[i], mcr[i]), xytext=(param[i], mcr[i]+1), fontsize=15)
            else:
                plt.annotate(methods[i], xy=(param[i], mcr[i]), xytext=(param[i], mcr[i] - 1.5), fontsize=15)

    plt.xscale("log")
    plt.tick_params(labelsize=15)
    plt.ylim((70, 86))
    plt.yticks(np.arange(70, 86, 5))
    plt.xlabel("Trainable Parameters (%)", fontsize=17)
    plt.ylabel("Last MCR (%)", fontsize=17)
    # plt.legend(fontsize=10)
    # plt.show()
    plt.savefig(home+"/param_mcr.png", dpi=400, bbox_inches="tight")

def skin_car(home):
    plt.rc("font", family="Times New Roman")
    methods = ["Continual-CLIP", "PR-CLIP", "CODA-Prompt", "DualPrompt", "EASE", "L2P", "MoE-Adapter", "PROOF", "SLCA", "Ours", "Lang-CL"]
    car = [55.4, 71.17, 67.75, 66.35, 65.88, 66.40, 67.30, 79.85, 66.82, 81.52, 78.07]
    skin = [14.75, 50.08, 30.25, 37.50, 50.75, 37.25, 23.75, 50.08, 46.75, 54.50, 51.42]
    markers = {"PROOF": "v",
               "PR-CLIP": ">",
               "Continual-CLIP": "8",
               "L2P": "X",
               "DualPrompt": "d",
               "CODA-Prompt": "P",
               "EASE": "d",
               "MoE-Adapter": "p",
               "SLCA": "s",
               "Lang-CL": "h",
               "Ours": "*",
               }
    colors = {"PROOF": "#1c78b4",
              "PR-CLIP": "#f68304",
              "Continual-CLIP": "#29a128",
              "L2P": "#cf3736",
              "DualPrompt": "#9264b3",
              "CODA-Prompt": "#89564f",
              "AttriCLIP": "#00FF97",
              "EASE": "#e276c9",
              "MoE-Adapter": "#808080",
              "SLCA": "#b9bb4c",
              "Lang-CL": "#1ebcc9",
              "Ours": "#E90D62",
              }
    plt.figure()
    plt.grid(linestyle="--", linewidth=1)
    # plt.axvline(0.038742, color="grey", linestyle="--", linewidth=1, zorder=1)
    for i in range(len(methods)):
        if methods[i] != "DualPrompt" and methods[i] != "EASE" and methods[i] != "Continual-CLIP":
            plt.scatter(skin[i], car[i], marker=markers[methods[i]], color=colors[methods[i]], label=methods[i], s=300 if methods[i]!="Ours" else 400)
    for i in range(len(methods)):
        if methods[i] != "DualPrompt" and methods[i] != "EASE" and methods[i] != "Continual-CLIP":
            if methods[i] == "CODA-Prompt":
                plt.annotate(methods[i], xy=(skin[i], car[i]), xytext=(skin[i] - 3, car[i] + 1.5), fontsize=13)
            elif methods[i] == "Lang-CL":
                plt.annotate(methods[i], xy=(skin[i], car[i]), xytext=(skin[i]-1, car[i]-2), fontsize=13)
            elif methods[i] == "SLCA" or  methods[i] == "MoE-Adapter" or methods[i] == "L2P":
                plt.annotate(methods[i], xy=(skin[i], car[i]), xytext=(skin[i] - 2, car[i] - 2), fontsize=13)
            elif methods[i] == "PROOF":
                plt.annotate(methods[i], xy=(skin[i], car[i]), xytext=(skin[i]-2.5, car[i]+1.5), fontsize=13)
            elif methods[i] == "Ours":
                plt.annotate(methods[i], xy=(skin[i], car[i]), xytext=(skin[i], car[i]+1), fontsize=13)
            else:
                plt.annotate(methods[i], xy=(skin[i], car[i]), xytext=(skin[i], car[i] - 1.5), fontsize=13)

    # plt.xscale("log")
    plt.tick_params(labelsize=15)
    plt.ylim((62, 85))
    plt.xlim((20, 58))
    # plt.yticks(np.arange(70, 86, 5))
    plt.xlabel("Last Accuracy on Skin40 (%)", fontsize=17)
    plt.ylabel("Last MCR on Cars196 (%)", fontsize=17)
    # plt.legend(fontsize=10)
    # plt.show()
    plt.savefig(home+"/skin_car.png", dpi=400, bbox_inches="tight")

def car_ir(home):
    plt.rc("font", family="Times New Roman")
    methods = ["Continual-CLIP", "PR-CLIP", "CODA-Prompt", "DualPrompt", "EASE", "L2P", "MoE-Adapters", "PROOF", "SLCA", "Ours", "Lang-CL"]
    car = [55.4, 71.17, 67.75, 66.35, 65.88, 66.40, 67.30, 79.85, 66.82, 81.52, 78.07]
    ir = [72.80, 76.40, 65.54, 73.76, 75.38, 72.86, 78.54, 77.39, 76.21, 83.58, 81.77]
    markers = {"PROOF": "v",
               "PR-CLIP": ">",
               "Continual-CLIP": "8",
               "L2P": "X",
               "DualPrompt": "D",
               "CODA-Prompt": "P",
               "EASE": "d",
               "MoE-Adapters": "p",
               "SLCA": "s",
               "Lang-CL": "h",
               "Ours": "*",
               }
    colors = {"PROOF": "#1c78b4",
              "PR-CLIP": "#f68304",
              "Continual-CLIP": "#29a128",
              "L2P": "#cf3736",
              "DualPrompt": "#9264b3",
              "CODA-Prompt": "#89564f",
              "AttriCLIP": "#00FF97",
              "EASE": "#e276c9",
              "MoE-Adapters": "#808080",
              "SLCA": "#b9bb4c",
              "Lang-CL": "#1ebcc9",
              "Ours": "#E90D62",
              }
    plt.figure()
    plt.grid(linestyle="--", linewidth=1)
    # plt.axvline(0.038742, color="grey", linestyle="--", linewidth=1, zorder=1)
    for i in range(len(methods)):
        if methods[i] != "CODA-Prompt" and methods[i] != "DualPrompt":
            plt.scatter(ir[i], car[i], marker=markers[methods[i]], color=colors[methods[i]], label=methods[i], s=300 if methods[i]!="Ours" else 400)
    for i in range(len(methods)):
        if methods[i] != "CODA-Prompt" and methods[i] != "DualPrompt":
            if methods[i] == "SLCA" or methods[i] == "PR-CLIP":
                plt.annotate(methods[i], xy=(ir[i], car[i]), xytext=(ir[i]-0.5, car[i]+1.5), fontsize=13)
            elif methods[i] == "MoE-Adapters":
                plt.annotate(methods[i], xy=(ir[i], car[i]), xytext=(ir[i]+0.5, car[i] - 1.5), fontsize=13)
            elif methods[i] == "Ours":
                plt.annotate(methods[i], xy=(ir[i], car[i]), xytext=(ir[i], car[i]+1), fontsize=13)
            else:
                plt.annotate(methods[i], xy=(ir[i], car[i]), xytext=(ir[i]-1, car[i] - 2), fontsize=13)

    # plt.xscale("log")
    plt.tick_params(labelsize=15)
    plt.ylim((62, 85))
    plt.xlim((70, 85))
    # plt.yticks(np.arange(70, 86, 5))
    plt.xlabel("Last MCR on ImageNet-R (%)", fontsize=17)
    plt.ylabel("Last MCR on Cars196 (%)", fontsize=17)
    # plt.legend(fontsize=10)
    # plt.show()
    plt.savefig(home+"/ir_car.png", dpi=400, bbox_inches="tight")

def sim(home):
    plt.rc("font", family="Times New Roman")
    similarity1 = [16.037036895751953, 9.623058319091797, 14.504819869995117, 29.472679138183594, 14.918977737426758, 21.8457088470459, 16.505287170410156, 17.497337341308594, 20.754545211791992, 15.841581344604492, 19.451068878173828, 17.119613647460938, 16.88067626953125, 19.001014709472656, 21.653411865234375, 11.353984832763672, 16.31109046936035, 15.383180618286133, 25.460054397583008, 16.34518051147461, 31.43338394165039, 14.452617645263672, 15.515519142150879, 17.4516544342041, 20.39766502380371, 18.86067771911621, 20.146879196166992, 22.856910705566406, 14.403997421264648, 14.459041595458984, 18.08422088623047, 13.86178207397461, 18.891454696655273, 17.914325714111328, 19.50491714477539, 19.017492294311523, 17.81846809387207, 14.819499015808105, 18.61827278137207, 24.88412094116211, 19.9068660736084, 16.93052864074707, 19.810874938964844, 20.3841552734375, 20.642026901245117, 20.738187789916992, 19.930328369140625, 19.956634521484375, 19.25067901611328, 15.974321365356445, 19.341716766357422, 21.974567413330078, 17.81782341003418, 19.923900604248047, 20.575693130493164, 19.38368034362793, 20.43114471435547, 24.806663513183594, 19.57233428955078, 22.944543838500977, 19.867332458496094, 18.512535095214844, 21.99803924560547, 18.87407684326172, 17.656612396240234, 16.78699493408203, 13.7942476272583, 18.725215911865234, 17.473316192626953, 25.19342041015625, 22.422637939453125, 20.940845489501953, 15.446849822998047, 21.081584930419922, 22.797311782836914, 19.690671920776367, 17.84527587890625, 17.48512840270996, 26.301931381225586, 20.35443878173828, 16.547138214111328, 15.897455215454102, 28.551271438598633, 20.201147079467773, 18.038169860839844, 18.02191162109375, 19.913724899291992, 15.334867477416992, 20.072351455688477, 20.421977996826172, 20.540935516357422, 20.771413803100586, 16.51426887512207, 23.113542556762695, 21.593128204345703, 20.717086791992188, 19.184226989746094, 20.404712677001953, 20.775279998779297, 19.268238067626953]
    similarity2 = [13.338555335998535, 1.7364559173583984, 4.4976911544799805, 9.706836700439453, 7.808988571166992, 9.04389476776123, 3.1091651916503906, 8.349664688110352, 8.125359535217285, 3.467820644378662, 7.576009273529053, 10.043116569519043, 4.453552722930908, 11.84376049041748, 10.337244033813477, 5.775503635406494, 7.792586326599121, 4.433212757110596, 10.19816780090332, 4.240273952484131, 8.856311798095703, 5.328285217285156, 7.098365783691406, 8.462751388549805, 12.887858390808105, 13.974752426147461, 9.296735763549805, 5.8448381423950195, 0.2309582233428955, 5.915815830230713, 6.401850700378418, 5.858822822570801, 10.718238830566406, 6.329084873199463, 4.939222812652588, 7.935503959655762, 8.646268844604492, 1.324317216873169, 6.206776142120361, 13.507554054260254, 11.387617111206055, 7.054771423339844, 7.077633857727051, 8.284403800964355, 10.879664421081543, 6.492557525634766, 11.84804916381836, 7.245430946350098, 6.060579776763916, 6.467247486114502, 5.981530666351318, 3.822072744369507, 6.428624153137207, 8.969470977783203, 11.21121597290039, 4.411055088043213, 6.338845729827881, 8.53861141204834, 3.0350723266601562, 8.089550018310547, 8.915624618530273, 3.020503282546997, 10.945719718933105, 7.614832878112793, 10.193429946899414, 7.40134334564209, 1.792555570602417, 6.8902997970581055, 0.8077688217163086, 12.546374320983887, 9.719733238220215, 12.124568939208984, 3.398894786834717, 4.019363880157471, 8.564149856567383, 9.679405212402344, 4.500581741333008, 3.9081578254699707, 8.12680721282959, 10.307151794433594, 5.839807987213135, 0.323488712310791, 27.014827728271484, 8.20619010925293, 2.516741991043091, 7.385371685028076, 12.024492263793945, 6.359596252441406, 6.405593395233154, 8.073577880859375, 6.613000869750977, 6.344925880432129, 6.977643966674805, 8.730338096618652, 5.940912246704102, 3.9712576866149902, 5.487063884735107, 16.64576530456543, 6.706890106201172, 5.017791748046875]
    similarity1 = [i/100 for i in similarity1]
    similarity2 = [i/100 for i in similarity2]
    # 横坐标索引
    # if flag == 19:
    #     print(similarity1.tolist())
    #     print(similarity2.tolist())
    x = np.arange(0, len(similarity1))
    colors = ['deepskyblue',  # 柔和蓝色
              'orangered']  # 浅绿色
    # 设置每组柱子的宽度
    # bar_width = 0.4
    # # 绘制双系列柱状图
    # fig, ax = plt.subplots()
    # bars1 = ax.bar(x - bar_width * 0.5, similarity1, width=bar_width, label='1',
    #                color=colors[0])
    # bars2 = ax.bar(x + bar_width * 0.5, similarity2, width=bar_width, label='2',
    #                color=colors[1])
    # # plt.bar(np.arange(0, len(similarity1)), similarity1)
    # plt.savefig("../3.jpg")
    width = 0.8
    # 绘制双系列柱状图
    fig, ax = plt.subplots()
    bars1 = ax.bar(x, similarity1, label='w/o CTRC', width=width,
                   color=colors[0])
    bars2 = ax.bar(x, similarity2, label='ours', width=width,
                   color=colors[1])
    plt.xlabel("Class", fontsize=17)
    plt.ylabel("Image/Text Similarity", fontsize=17)
    plt.tick_params(labelsize=15)
    plt.legend(fontsize=10)
    # plt.bar(np.arange(0, len(similarity1)), similarity1)
    # plt.show()
    plt.savefig(home+"/confusion.png", dpi=400, bbox_inches="tight")


def uncertainty_plot(home):
    plt.rc("font", family="Times New Roman")
    dist = np.array([4.6812,  4.9328,  5.4073,  6.2365,  7.1909,  8.2304,  9.4035, 10.6184, 11.8341, 13.1101])
    y = np.array([0.0069, 0.00942, 0.0126, 0.00965, 0.0218, 0.0197, 0.0289, 0.0425, 0.0839, 0.134])
    y1 = np.array([0.0011, 0.0023, 0.0007, 0.0015, 0.0055, 0.0236, 0.0252, 0.0714, 0.1219, 0.2462])

    fig = plt.figure(figsize=(6, 5.2))
    plt.axes(facecolor="#EAEAF2")
    plt.plot(dist, y, marker="o", color="r", label="pseudo-features before MoP")
    plt.plot(dist, y1, marker="o", color="blue", label="pseudo-features after MoP")
    plt.tick_params(labelsize=15)
    plt.grid(color="white", linewidth=2)
    plt.xlabel("Distance to class center", fontsize=20)
    plt.ylabel("Entropy", fontsize=20)
    plt.legend(fontsize=15)

    # plt.show()
    plt.savefig(home+"/dist_entropy.png", dpi=400, bbox_inches="tight")


def delta_entropy(home):
    plt.rc("font", family="Times New Roman")
    # ir 10
    # data = np.array([[581, 390, 481, 400, 352, 401, 431, 438, 395, 416],
    #                  [327, 520, 332, 376, 361, 368, 364, 356, 331, 333],
    #                  [417, 359, 518, 327, 361, 361, 381, 368, 390, 378],
    #                  [304, 314, 285, 417, 301, 247, 276, 293, 273, 283],
    #                  [322, 404, 336, 364, 500, 375, 346, 346, 332, 334],
    #                  [310, 360, 351, 285, 322, 523, 365, 315, 305, 314],
    #                  [388, 381, 397, 387, 390, 416, 497, 417, 402, 379],
    #                  [401, 440, 477, 443, 460, 367, 455, 605, 446, 485],
    #                  [266, 256, 282, 291, 300, 268, 264, 283, 391, 288],
    #                  [301, 327, 363, 322, 334, 275, 330, 336, 366, 499]])
    # total = [0, 697, 1347, 1928, 2458, 3001, 3634, 4193, 4911, 5407, 6000]
    # ir 20
    data = np.array([[313, 201, 249, 202, 260, 201, 206, 187, 242, 194, 237, 223, 215, 209, 228, 225, 235, 237, 237, 235],
                    [265, 314, 261, 262, 268, 266, 240, 239, 272, 239, 290, 259, 267, 242, 280, 260, 253, 274, 275, 274],
                    [248, 200, 256, 230, 236, 202, 231, 184, 225, 201, 221, 198, 226, 215, 219, 221, 221, 225, 219, 231],
                    [234, 241, 217, 311, 232, 242, 240, 233, 243, 235, 248, 268, 267, 246, 267, 222, 246, 250, 258, 257],
                    [216, 180, 194, 202, 244, 187, 191, 164, 199, 194, 196, 198, 207, 194, 197, 200, 191, 211, 207, 218],
                    [229, 191, 195, 202, 214, 267, 206, 212, 218, 184, 213, 199, 229, 214, 218, 198, 211, 224, 231, 225],
                    [191, 159, 171, 192, 171, 188, 239, 188, 175, 177, 170, 188, 201, 194, 208, 173, 174, 183, 183, 193],
                    [156, 142, 151, 154, 162, 152, 149, 191, 153, 124, 137, 151, 159, 148, 151, 148, 152, 164, 161, 174],
                    [263, 238, 265, 278, 242, 257, 252, 212, 315, 276, 280, 271, 278, 240, 280, 263, 262, 272, 264, 264],
                    [103, 101,  91, 109,  96,  87, 116,  89,  92, 144, 100, 108, 117,  98, 114, 100,  95,  99, 100, 112],
                    [202, 176, 195, 211, 179, 193, 201, 156, 204, 192, 243, 203, 208, 197, 217, 190, 201, 204, 184, 194],
                    [249, 194, 220, 263, 246, 241, 256, 213, 248, 243, 266, 326, 260, 249, 257, 244, 230, 258, 258, 267],
                    [188, 163, 164, 189, 169, 177, 176, 174, 166, 156, 175, 180, 226, 180, 180, 175, 172, 189, 193, 201],
                    [247, 257, 212, 254, 224, 251, 250, 254, 247, 257, 245, 262, 228, 295, 249, 248, 257, 216, 235, 249],
                    [232, 195, 198, 216, 195, 224, 232, 203, 227, 216, 203, 179, 200, 211, 288, 204, 201, 215, 213, 234],
                    [289, 245, 269, 291, 272, 268, 265, 258, 299, 237, 270, 291, 272, 254, 278, 328, 273, 287, 294, 305],
                    [190, 162, 180, 171, 177, 179, 179, 166, 187, 138, 194, 159, 179, 162, 180, 171, 227, 182, 178, 173],
                    [158, 124, 160, 160, 164, 152, 165, 134, 144, 153, 150, 155, 159, 160, 164, 144, 160, 201, 172, 168],
                    [218, 163, 209, 207, 220, 183, 188, 167, 194, 171, 173, 207, 218, 165, 199, 202, 182, 217, 279, 233],
                    [189, 157, 184, 211, 193, 187, 203, 214, 201, 173, 174, 201, 210, 182, 196, 186, 186, 198, 218, 265]])
    total = [0, 338, 697, 1019, 1347, 1635, 1928, 2198, 2458, 2843, 3001, 3264, 3634, 3876, 4193, 4555, 4911, 5165, 5407, 5721, 6000]
    total = [total[i+1]-total[i] for i in range(len(total)-1)]
    # cars 10
    # data = np.array([[766, 559, 612, 598, 640, 632, 674, 666, 662, 674],
    #                 [633, 804, 593, 621, 629, 656, 511, 609, 603, 631],
    #                 [615, 592, 805, 642, 722, 745, 675, 720, 741, 712],
    #                 [570, 536, 531, 778, 702, 602, 703, 729, 655, 693],
    #                 [587, 574, 652, 682, 799, 643, 660, 735, 728, 757],
    #                 [537, 530, 587, 530, 585, 787, 587, 604, 653, 606],
    #                 [535, 482, 589, 594, 659, 606, 781, 647, 639, 670],
    #                 [591, 580, 592, 699, 730, 666, 692, 779, 739, 742],
    #                 [630, 587, 686, 691, 775, 723, 725, 797, 818, 766],
    #                 [565, 467, 563, 641, 634, 574, 637, 657, 627, 696]])
    # total = [0, 819, 1646, 2468, 3266, 4099, 4918, 5714, 6500, 7328, 8041]

    # cifar 10
    # data = np.array([[974, 858, 851, 821, 839, 846, 854, 866, 879, 863],
    #                  [837, 987, 851, 810, 852, 848, 864, 879, 855, 836],
    #                  [874, 833, 985, 920, 906, 932, 897, 920, 928, 942],
    #                  [770, 816, 876, 992, 902, 919, 898, 938, 949, 939],
    #                  [842, 830, 875, 916, 985, 913, 914, 907, 928, 914],
    #                  [797, 862, 844, 915, 916, 970, 922, 919, 920, 924],
    #                  [783, 774, 847, 887, 902, 935, 986, 906, 911, 925],
    #                  [707, 741, 809, 847, 879, 899, 882, 989, 885, 906],
    #                  [801, 814, 870, 913, 880, 904, 921, 905, 988, 915],
    #                  [698, 793, 894, 926, 921, 941, 937, 953, 948, 979],])
    # total = [1000]*10



    # total = [total[i+1]-total[i] for i in range(len(total)-1)]
    data = np.divide(data, np.array(total).reshape(-1, 1))
    fig = plt.figure()
    plt.imshow(data, cmap=plt.get_cmap('Blues'))
    plt.colorbar()
    plt.tick_params(labelsize=12)
    plt.xlabel("Task identity t$_{1}$", fontsize=15)
    plt.ylabel("Task identity t$_{2}$", fontsize=15)
    # plt.show()
    plt.savefig(home+"/delta_entropy_ir20.png", dpi=400, bbox_inches="tight")

def r_N(home):
    plt.rc("font", family="Times New Roman")
    red = "#D4242A"
    blue = "#168676"
    x1 = [8, 16, 32, 64, 128]
    y1 = [83.18, 82.77, 83.14, 83.13, 82.68]
    y2 = [88.47, 88.40, 88.91, 89.11, 88.81]

    # x1 = [32, 64, 128, 256, 512]
    # y1 = [82.29, 82.89, 82.93, 83.13, 82.78,]
    # y2 = [88.25, 88.17, 88.83, 89.11, 88.92,]
    fig = plt.figure(figsize=(7, 5)) #figsize=(6.5, 5)
    ax1 = fig.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x1, y1, color=red, marker="o", label="Last MCR of Ours")
    ax1.axhline(81.77, color=red, linestyle="-.", label="Last MCR of Lang-CL")
    # plt.axhline(76.21, color="red", linestyle="--", label="SLCA on ImageNet-R")
    # plt.axhline(66.82, color="blue", linestyle="--", label="SLCA on Cars196")
    ax2.plot(x1, y2, color=blue, marker="^", label="Average MCR of Ours")
    ax2.axhline(87.65, color=blue, linestyle="--", label="Average MCR of Lang-CL")
    ax1.set_ylim((77, 87))
    ax2.set_ylim((80, 90))
    ax1.tick_params(axis="x", labelsize=15)
    # 设置轴标签颜色
    ax1.tick_params('y', colors=red, labelsize=15)
    ax2.tick_params('y', colors=blue, labelsize=15)
    # 设置轴颜色
    ax1.spines['left'].set_color(red)
    ax2.spines['left'].set_color(red)
    ax1.spines['right'].set_color(blue)
    ax2.spines['right'].set_color(blue)
    ax1.set_xscale("log", base=2)
    ax2.set_xscale("log", base=2)

    ax1.set_xlabel("r", fontsize=17)
    ax1.set_ylabel("Last MCR on ImageNet-R (%)", fontsize=20, color=red)
    ax2.set_ylabel("Average MCR on ImageNet-R (%)", fontsize=20, color=blue)
    ax1.legend(fontsize=13, loc="lower left")
    ax2.legend(fontsize=13, loc="lower right")

    # plt.show()
    plt.savefig(home+"/r.png", dpi=400, bbox_inches="tight")

home = "C:/Users/tjt/Desktop/cvpr/fig"
# inference(home)
# proj1(home)
# M(home)
# mm(home)
# param_ir(home)
# results(home)
# skin_car(home)
# car_ir(home)
# sim(home)
r_N(home)
# uncertainty_plot(home)

# delta_entropy(home)
# 导入操作系统模块，用于文件操作和环境变量访问
import os
# 导入随机数模块，用于生成随机数
import random

# 导入xgboost模块，用于机器学习中的梯度提升算法
import xgboost as xgb
# 从io模块导入BytesIO类，用于在内存中处理二进制数据
from io import BytesIO
# 导入PyTorch模块，用于深度学习框架
import torch
# 导入正则表达式模块，用于模式匹配和字符串操作
import re
# 导入pdfplumber模块，用于解析PDF文档
import pdfplumber
# 设置日志模块的级别为WARNING，减少日志输出
import logging
logging.getLogger("pdfminer").setLevel(logging.WARNING)
# 导入PIL模块，用于图像处理
from PIL import Image, ImageDraw
# 导入NumPy模块，用于科学计算
import numpy as np
# 导入timeit模块，用于测量代码执行时间
from timeit import default_timer as timer
# 导入PyPDF2模块，用于处理PDF文档
from PyPDF2 import PdfReader as pdf2_read

# 从file_utils模块导入get_project_base_directory函数，用于获取项目基础目录
from api.utils.file_utils import get_project_base_directory
# 从vision模块导入OCR、Recognizer、LayoutRecognizer和TableStructureRecognizer类，用于视觉识别
from deepdoc.vision import OCR, Recognizer, LayoutRecognizer, TableStructureRecognizer
# 从nlp模块导入rag_tokenizer，用于自然语言处理中的区域关注（RAG）分词
from rag.nlp import rag_tokenizer
# 导入deepcopy函数，用于创建对象的深拷贝
from copy import deepcopy
# 从huggingface_hub模块导入snapshot_download函数，用于下载Hugging Face Hub上的模型快照
from huggingface_hub import snapshot_download



class RAGFlowPdfParser:
    """
    RAGFlowPdfParser类初始化文档结构识别的各个组件。
    它负责从PDF文档中提取文本、布局信息，并识别表格结构。
    还包括上下文计数模型的初始化和加载，用于进一步优化文档结构的理解。
    """

    def __init__(self):
        """
        初始化RAGFlowPdfParser实例。
        设置OCR引擎、布局识别器、表格结构识别器，并初始化上下文计数模型。
        如果可用，将模型部署到CUDA设备上，并尝试加载预训练模型。
        """
        # 初始化OCR引擎，用于从PDF中提取文本。
        self.ocr = OCR()
        # 根据是否有指定的模型物种，初始化布局识别器。
        if hasattr(self, "model_speciess"):
            self.layouter = LayoutRecognizer("layout." + self.model_speciess)
        else:
            self.layouter = LayoutRecognizer("layout")
        # 初始化表格结构识别器，用于识别PDF中的表格结构。
        self.tbl_det = TableStructureRecognizer()

        # 初始化上下文计数模型，用于识别文本块的上下文关系。
        self.updown_cnt_mdl = xgb.Booster()
        # 如果CUDA设备可用，将模型部署到CUDA设备上。
        if torch.cuda.is_available():
            self.updown_cnt_mdl.set_param({"device": "cuda"})
        try:
            # 尝试从项目基础目录加载预训练的上下文计数模型。
            model_dir = os.path.join(
                get_project_base_directory(),
                "rag/res/deepdoc")
            self.updown_cnt_mdl.load_model(os.path.join(
                model_dir, "updown_concat_xgb.model"))
        except Exception as e:
            # 如果加载失败，从云端下载模型。
            model_dir = snapshot_download(
                repo_id="InfiniFlow/text_concat_xgb_v1.0",
                local_dir=os.path.join(get_project_base_directory(), "rag/res/deepdoc"),
                local_dir_use_symlinks=False)
            self.updown_cnt_mdl.load_model(os.path.join(
                model_dir, "updown_concat_xgb.model"))

        # 初始化页面起始索引，用于跟踪处理的页面。
        self.page_from = 0

        """
        If you have trouble downloading HuggingFace models, -_^ this might help!!

        For Linux:
        export HF_ENDPOINT=https://hf-mirror.com

        For Windows:
        Good luck
        ^_-

        """

    def __char_width(self, c):
        """
        计算字符的宽度。

        这个方法通过比较字符的坐标差值和字符文本的长度，来确定字符的逻辑宽度。
        它主要用于处理和计算文本显示时每个字符所占据的空间。

        参数:
            c (dict): 包含字符信息的字典，字典中应包含字符的坐标信息（"x0" 和 "x1"）以及字符的文本内容（"text"）。

        返回:
            int: 字符的逻辑宽度，计算方式为字符的右边界坐标减去左边界坐标，然后除以字符文本长度的最大值（防止除以0）。
        """
        # 计算字符的宽度，考虑到字符可能为空的情况，使用max确保除数不为0
        return (c["x1"] - c["x0"]) // max(len(c["text"]), 1)

    def __height(self, c):
        """
        计算给定字典表示的窗口的高度。

        参数:
            c (dict): 包含窗口顶部和底部坐标的字典。

        返回:
            int: 窗口的高度，通过底部减去顶部得到。
        """
        return c["bottom"] - c["top"]

    def _x_dis(self, a, b):
        """
        计算两个物体在x轴方向上的最小距离。

        该函数通过比较物体a和物体b在x轴上的不同位置组合下的距离，找出最小的x轴距离。
        物体的位置由字典表示，包含"x0"和"x1"两个键，分别代表物体在x轴上的两个点。

        参数:
        a (dict): 表示物体a的字典，包含"x0"和"x1"两个键值对。
        b (dict): 表示物体b的字典，包含"x0"和"x1"两个键值对。

        返回:
        float: 返回物体a和物体b在x轴方向上的最小距离。
        """
        # 计算三种可能的x轴距离差，并返回最小值
        # 第一种情况：a的x0到b的x0的距离
        # 第二种情况：a的x1到b的x1的距离
        # 第三种情况：a和b的x坐标和的差的一半，用于处理物体横跨x轴的情况
        return min(abs(a["x1"] - b["x0"]), abs(a["x0"] - b["x1"]),
                   abs(a["x0"] + a["x1"] - b["x0"] - b["x1"]) / 2)

    def _y_dis(self, a, b):
        """
        计算两个矩形在y轴方向上的距离。

        该函数用于处理两个矩形在垂直方向上的相对位置，通过计算它们顶部和底部之间的净间隔的一半，
        来得到一个矩形的中心到另一个矩形中心的y轴距离。

        参数:
        a (dict): 第一个矩形的边界信息，包含"top"和"bottom"两个键值对。
        b (dict): 第二个矩形的边界信息，包含"top"和"bottom"两个键值对。

        返回:
        float: 两个矩形中心在y轴方向上的距离。
        """
        # 计算两个矩形顶部和底部之间的总距离，然后除以2得到中心点的垂直距离
        return (b["top"] + b["bottom"] - a["top"] - a["bottom"]) / 2

    def _match_proj(self, b):
        """
        检查文本是否符合特定的项目标记模式。

        该函数通过匹配一系列正则表达式来判断给定文本是否包含项目标记。这些标记通常是用于区分不同项目的符号或数字序列。

        参数:
        b: 字典, 包含待检查文本的字典项。

        返回:
        布尔值, 如果文本匹配任何项目标记模式，则返回True，否则返回False。
        """
        # 定义项目标记的正则表达式列表
        proj_patt = [
            r"第[零一二三四五六七八九十百]+章",
            r"第[零一二三四五六七八九十百]+[条节]",
            r"[零一二三四五六七八九十百]+[、是 　]",
            r"[\(（][零一二三四五六七八九十百]+[）\)]",
            r"[\(（][0-9]+[）\)]",
            r"[0-9]+(、|\.[　 ]|）|\.[^0-9./a-zA-Z_%><-]{4,})",
            r"[0-9]+\.[0-9.]+(、|\.[ 　])",
            r"[⚫•➢①② ]",
        ]

        # 遍历所有正则表达式，如果任何正则表达式与文本匹配，则返回True
        return any([re.match(p, b["text"]) for p in proj_patt])

    def _updown_concat_features(self, up, down):
        """
        结合上下文特征，用于特征工程中的向量拼接。

        参数:
        up: 上文的特征字典。
        down: 下文的特征字典。

        返回:
        fea: 结合上下文后的特征列表。
        """
        # 计算上下文的最大字符宽度
        w = max(self.__char_width(up), self.__char_width(down))
        # 计算上下文的最大高度
        h = max(self.__height(up), self.__height(down))
        # 计算上下文的y方向间距
        y_dis = self._y_dis(up, down)
        # 定义特征长度
        LEN = 6
        # 分割并获取下文前LEN个字符的特征
        tks_down = rag_tokenizer.tokenize(down["text"][:LEN]).split(" ")
        # 分割并获取上文后LEN个字符的特征
        tks_up = rag_tokenizer.tokenize(up["text"][-LEN:]).split(" ")
        # 拼接上下文LEN个字符的特征，考虑字符间是否需要添加空格
        tks_all = up["text"][-LEN:].strip() + \
                  (" " if re.match(r"[a-zA-Z0-9]+", up["text"][-1] + down["text"][0]) else "") + \
                  down["text"][:LEN].strip()
        # 对拼接后的特征进行分词
        tks_all = rag_tokenizer.tokenize(tks_all).split(" ")

        # 初始化特征列表
        fea = [
            # 判断上下文是否在同一行
            up.get("R", -1) == down.get("R", -1),
            # y方向间距与高度的比值
            y_dis / h,
            # 上下文页码差
            down["page_number"] - up["page_number"],
            # 判断上下文布局类型是否相同
            up["layout_type"] == down["layout_type"],
            # 判断上文是否为文本类型
            up["layout_type"] == "text",
            # 判断下文是否为文本类型
            down["layout_type"] == "text",
            # 判断上文是否为表格类型
            up["layout_type"] == "table",
            # 判断下文是否为表格类型
            down["layout_type"] == "table",
            # 判断上文是否以标点符号结尾
            True if re.search(r"([。？！；!?;+)）]|[a-z]\.)$", up["text"]) else False,
            # 判断上文是否以逗号、数字或特定符号结尾
            True if re.search(r"[，：‘“、0-9（+-]$", up["text"]) else False,
            # 判断下文是否以特定标点符号开头
            True if re.search(r"(^.?[/,?;:\]，。；：’”？！》】）-])", down["text"]) else False,
            # 判断上文是否为括号内的内容
            True if re.match(r"[\(（][^\(\)（）]+[）\)]$", up["text"]) else False,
            # 判断上文是否以逗号结尾且后面有内容
            True if re.search(r"[，,][^。.]+$", up["text"]) else False,
            # 重复的特征，可能是代码复制粘贴的错误
            True if re.search(r"[，,][^。.]+$", up["text"]) else False,
            # 判断上下文之间是否存在匹配的括号
            True if re.search(r"[\(（][^\)）]+$", up["text"]) and re.search(r"[\)）]", down["text"]) else False,
            # 判断下文是否与上文在相同列中
            self._match_proj(down),
            # 判断下文是否以大写字母开头
            True if re.match(r"[A-Z]", down["text"]) else False,
            # 判断上文是否以大写字母结尾
            True if re.match(r"[A-Z]", up["text"][-1]) else False,
            # 判断上文是否以小写字母或数字结尾
            True if re.match(r"[a-z0-9]", up["text"][-1]) else False,
            # 判断下文是否以数字、百分比、逗号或破折号开头
            True if re.match(r"[0-9.%,-]+$", down["text"]) else False,
            # 判断上文最后两个字符是否与下文相同
            up["text"].strip()[-2:] == down["text"].strip()[-2:] if len(up["text"].strip()) > 1 and len(down["text"].strip()) > 1 else False,
            # 判断上文的左边界是否在下文的右边界左侧
            up["x0"] > down["x1"],
            # 计算高度差与最小高度的比值
            abs(self.__height(up) - self.__height(down)) / min(self.__height(up), self.__height(down)),
            # 计算x方向间距与最大宽度的比值
            self._x_dis(up, down) / max(w, 0.000001),
            # 计算上文和下文文本长度的差异，反映信息量的差异
            (len(up["text"]) - len(down["text"])) /
            max(len(up["text"]), len(down["text"])),
            # 计算总词数减去上文和下文词数的差，反映信息的完整性
            len(tks_all) - len(tks_up) - len(tks_down),
            # 计算下文词数减去上文词数的差，反映信息的偏向性
            len(tks_down) - len(tks_up),
            # 判断上文和下文的最后一个词是否相同，反映信息的连贯性
            tks_down[-1] == tks_up[-1],
            # 取上文和下文中in_row值较大的一个，反映信息的重要程度
            max(down["in_row"], up["in_row"]),
            # 计算上文和下文中in_row值的差，反映信息的分布变化
            abs(down["in_row"] - up["in_row"]),
            # 判断下文是否只有一个词且被标记为名词，反映信息的特定类型
            len(tks_down) == 1 and rag_tokenizer.tag(tks_down[0]).find("n") >= 0,
            # 判断上文是否只有一个词且被标记为名词，反映信息的特定类型
            len(tks_up) == 1 and rag_tokenizer.tag(tks_up[0]).find("n") >= 0
        ]
        return fea

    @staticmethod
    def sort_X_by_page(arr, threashold):
        """
        按页面编号、X坐标及顶部位置对数组进行排序。

        首先，根据页面编号、元素的左边缘X坐标及顶部位置对数组进行初步排序，确保大致顺序。
        随后，对顺序进行细致调整：如果在同一页面上，当前元素与前一个元素的X坐标差距小于设定阈值，并且当前元素的顶部位置更低，则交换这两个元素的位置。

        参数:
        arr: 一个字典列表，每个字典包含页面编号、X坐标、顶部位置等信息。
        threashold: 判断X坐标接近程度的阈值。

        返回:
        排序后的数组。
        """
        # 根据页面编号、X坐标（左边缘）及顶部位置进行初步排序
        # 使用y坐标作为第二关键字进行排序
        arr = sorted(arr, key=lambda r: (r["page_number"], r["x0"], r["top"]))

        # 对排序结果进行详细调整
        for i in range(len(arr) - 1):
            for j in range(i, -1, -1):
                # 检查条件：X坐标差值小于阈值，后一元素顶部位置更低，且在同一页面上
                # 根据上述条件调整元素顺序
                if abs(arr[j + 1]["x0"] - arr[j]["x0"]) < threashold \
                        and arr[j + 1]["top"] < arr[j]["top"] \
                        and arr[j + 1]["page_number"] == arr[j]["page_number"]:
                    # 交换两个相邻元素的位置
                    tmp = arr[j]
                    arr[j] = arr[j + 1]
                    arr[j + 1] = tmp
        # 返回最终排序完成的数组
        return arr

    def _has_color(self, o):
        """
        判断对象是否包含颜色信息。

        该方法主要用于分析给定对象是否具有颜色属性。如果对象的颜色属性符合特定条件，
        则认为该对象不包含颜色信息；否则，认为包含颜色信息。

        参数:
        o: 字典类型的对象，包含颜色相关的属性。

        返回值:
        布尔类型，如果对象不包含颜色信息，则返回True；否则返回False。
        """
        # 检查颜色空间是否为“DeviceGray”，即单色空间
        if o.get("ncs", "") == "DeviceGray":
            # 检查填充色和描边色是否均为纯黑色（值为1）
            if o["stroking_color"] and o["stroking_color"][0] == 1 and o["non_stroking_color"] and \
                    o["non_stroking_color"][0] == 1:
                # 如果对象文本符合特定正则表达式模式，则认为不包含颜色信息
                # 这里的正则表达式用于匹配一些特定的文本模式，具体含义可能需要结合上下文进一步理解
                if re.match(r"[a-zT_\[\]\(\)-]+", o.get("text", "")):
                    return False
        # 默认情况下，认为对象包含颜色信息
        return True


    def _table_transformer_job(self, ZM):
        """
        处理表格识别任务。

        对页面中的表格进行定位和识别，将识别出的表格组件存储起来。

        参数:
        ZM: 表格放大倍数，用于调整表格边界。
        """
        # 记录日志信息，表示开始处理表格
        logging.info("Table processing...")
        imgs, pos = [], []
        tbcnt = [0]
        # 定义表格边界的缓冲区大小
        MARGIN = 10
        # 初始化存储表格组件的列表
        self.tb_cpns = []
        # 确保页面布局和图片的数量一致
        assert len(self.page_layout) == len(self.page_images)
        # 遍历每个页面上的表格
        for p, tbls in enumerate(self.page_layout):  # for page
            # 筛选出页面上的表格组件
            tbls = [f for f in tbls if f["type"] == "table"]
            # 记录当前页面上的表格数量
            tbcnt.append(len(tbls))
            # 如果当前页面没有表格，则跳过
            if not tbls:
                continue
            # 遍历当前页面上的每个表格
            for tb in tbls:  # for table
                # 计算表格的边界，并考虑边界的缓冲区
                left, top, right, bott = tb["x0"] - MARGIN, tb["top"] - MARGIN, \
                                         tb["x1"] + MARGIN, tb["bottom"] + MARGIN
                # 将边界值转换为指定的单位
                left *= ZM
                top *= ZM
                right *= ZM
                bott *= ZM
                # 记录表格的位置信息
                pos.append((left, top))
                # 截取表格对应的图片区域
                imgs.append(self.page_images[p].crop((left, top, right, bott)))

        # 确保处理后的表格数量与页面数量一致
        assert len(self.page_images) == len(tbcnt) - 1
        # 如果没有截取到任何表格图片，则直接返回
        if not imgs:
            return
        # 使用表格检测算法识别每个表格中的组件
        recos = self.tbl_det(imgs)
        # 计算每个页面上累计的表格数量，用于后续的表格组件定位
        tbcnt = np.cumsum(tbcnt)
        # 遍历每个页面，对页面上的每个表格进行处理
        for i in range(len(tbcnt) - 1):  # for page
            pg = []
            # 遍历当前页面上的每个表格，以及对应的识别结果
            for j, tb_items in enumerate(
                    recos[tbcnt[i]: tbcnt[i + 1]]):  # for table
                # 获取当前表格的位置信息
                poss = pos[tbcnt[i]: tbcnt[i + 1]]
                # 遍历当前表格中的每个组件，调整其位置信息
                for it in tb_items:  # for table components
                    # 根据表格的位置信息，调整组件的绝对位置
                    it["x0"] = (it["x0"] + poss[j][0])
                    it["x1"] = (it["x1"] + poss[j][0])
                    it["top"] = (it["top"] + poss[j][1])
                    it["bottom"] = (it["bottom"] + poss[j][1])
                    # 将位置信息转换为原始单位
                    for n in ["x0", "x1", "top", "bottom"]:
                        it[n] /= ZM
                    # 根据页面的累计高度，调整组件的垂直位置
                    it["top"] += self.page_cum_height[i]
                    it["bottom"] += self.page_cum_height[i]
                    # 添加页面编号和表格编号信息
                    it["pn"] = i
                    it["layoutno"] = j
                    # 将处理后的表格组件添加到结果列表中
                    pg.append(it)
            # 将当前页面的所有表格组件添加到全局列表中
            self.tb_cpns.extend(pg)


        def gather(kwd, fzy=10, ption=0.6):
            """
            根据关键词聚集相关元素。

            本函数主要用于从一组元素中，根据提供的关键词和模糊匹配程度，
            以及布局清理参数，筛选并聚集出符合条件的元素集合。

            参数:
            kwd: str - 用于匹配的关键词。
            fzy: int - 模糊匹配的阈值，用于控制匹配的宽松程度。
            ption: float - 布局清理的阈值，用于控制在清理布局时的保留程度。

            返回:
            list - 匹配并聚集后的元素列表。
            """
            # 根据关键词模糊匹配元素，并按照Y轴排序
            eles = Recognizer.sort_Y_firstly(
                [r for r in self.tb_cpns if re.match(kwd, r["label"])], fzy)
            # 清理元素布局，并进行第二次Y轴排序
            eles = Recognizer.layouts_cleanup(self.boxes, eles, 5, ption)
            return Recognizer.sort_Y_firstly(eles, 0)


        # add R,H,C,SP tag to boxes within table layout
        # 通过正则表达式收集表格中的头部分组
        headers = gather(r".*header$")
        # 收集表格中的行元素，包括行和头
        rows = gather(r".* (row|header)")
        # 收集表格中的跨列元素
        spans = gather(r".*spanning")
        # 筛选并排序表格中的列元素
        clmns = sorted([r for r in self.tb_cpns if re.match(
            r"table column$", r["label"])], key=lambda x: (x["pn"], x["layoutno"], x["x0"]))
        # 对列元素进行布局清理
        clmns = Recognizer.layouts_cleanup(self.boxes, clmns, 5, 0.5)
        # 遍历所有框，为属于表格布局的框添加标识
        for b in self.boxes:
            # 如果框的布局类型不是表格，则跳过
            if b.get("layout_type", "") != "table":
                continue
            # 查找与框重叠的行元素，并为框添加行标识
            ii = Recognizer.find_overlapped_with_threashold(b, rows, thr=0.3)
            if ii is not None:
                b["R"] = ii
                b["R_top"] = rows[ii]["top"]
                b["R_bott"] = rows[ii]["bottom"]

            # 查找与框重叠的头元素，并为框添加头标识及相关位置信息
            ii = Recognizer.find_overlapped_with_threashold(
                b, headers, thr=0.3)
            if ii is not None:
                b["H_top"] = headers[ii]["top"]
                b["H_bott"] = headers[ii]["bottom"]
                b["H_left"] = headers[ii]["x0"]
                b["H_right"] = headers[ii]["x1"]
                b["H"] = ii

            # 查找与框水平紧密匹配的列元素，并为框添加列标识及相关位置信息
            ii = Recognizer.find_horizontally_tightest_fit(b, clmns)
            if ii is not None:
                b["C"] = ii
                b["C_left"] = clmns[ii]["x0"]
                b["C_right"] = clmns[ii]["x1"]

            # 查找与框重叠的跨列元素，并为框添加跨列标识及相关位置信息
            ii = Recognizer.find_overlapped_with_threashold(b, spans, thr=0.3)
            if ii is not None:
                b["H_top"] = spans[ii]["top"]
                b["H_bott"] = spans[ii]["bottom"]
                b["H_left"] = spans[ii]["x0"]
                b["H_right"] = spans[ii]["x1"]
                b["SP"] = ii


    def __ocr(self, pagenum, img, chars, ZM=3):
        """
        使用OCR技术识别图像中的字符，并将识别结果整理为结构化数据。

        参数:
        pagenum: int, 图像对应的页码。
        img: Image对象, 需要识别的图像。
        chars: List[Dict], 原始字符的列表，每个字符由一个字典表示。
        ZM: int, 图像缩放因子，默认为3。

        返回:
        无返回值，但将识别结果更新到self.boxes中。
        """
        # 使用OCR检测技术获取图像中的文本框边界
        bxs = self.ocr.detect(np.array(img))
        # 如果没有检测到文本框，则添加一个空列表到self.boxes中并返回
        if not bxs:
            self.boxes.append([])
            return
        # 整理OCR检测结果，只保留文本框的左上角和右下角坐标
        bxs = [(line[0], line[1][0]) for line in bxs]
        # 根据整理后的边界框和页面缩放因子，计算并排序文本框的中间点坐标
        bxs = Recognizer.sort_Y_firstly(
            [{"x0": b[0][0] / ZM, "x1": b[1][0] / ZM,
              "top": b[0][1] / ZM, "text": "", "txt": t,
              "bottom": b[-1][1] / ZM,
              "page_number": pagenum} for b, t in bxs if b[0][0] <= b[1][0] and b[0][1] <= b[-1][1]],
            self.mean_height[-1] / 3
        )

        # 对检测到的字符进行处理，尝试将它们合并到相应的文本框中
        # merge chars in the same rect
        for c in Recognizer.sort_X_firstly(
                chars, self.mean_width[pagenum - 1] // 4):
            # 寻找与当前字符重叠的文本框
            ii = Recognizer.find_overlapped(c, bxs)
            # 如果没有找到重叠的文本框，则将字符添加到左侧未匹配字符列表中
            if ii is None:
                self.lefted_chars.append(c)
                continue
            # 如果字符与文本框的高度差异过大，或字符为空格，则也将其添加到左侧未匹配字符列表中
            ch = c["bottom"] - c["top"]
            bh = bxs[ii]["bottom"] - bxs[ii]["top"]
            if abs(ch - bh) / max(ch, bh) >= 0.7 and c["text"] != ' ':
                self.lefted_chars.append(c)
                continue
            # 如果字符是空格且文本框已有文本，则将空格添加到文本框中
            if c["text"] == " " and bxs[ii]["text"]:
                if re.match(r"[0-9a-zA-Z,.?;:!%%]", bxs[ii]["text"][-1]):
                    bxs[ii]["text"] += " "
            # 否则，将字符添加到文本框的文本中
            else:
                bxs[ii]["text"] += c["text"]

        # 对于没有文本的文本框，使用OCR技术进行再次识别
        for b in bxs:
            if not b["text"]:
                left, right, top, bott = b["x0"] * ZM, b["x1"] * \
                                         ZM, b["top"] * ZM, b["bottom"] * ZM
                b["text"] = self.ocr.recognize(np.array(img),
                                               np.array([[left, top], [right, top], [right, bott], [left, bott]],
                                                        dtype=np.float32))
            # 删除临时字段"txt"
            del b["txt"]
        # 移除没有文本的文本框，并更新页面的平均字符高度
        bxs = [b for b in bxs if b["text"]]
        if self.mean_height[-1] == 0:
            self.mean_height[-1] = np.median([b["bottom"] - b["top"]
                                              for b in bxs])
        # 将处理后的文本框列表添加到self.boxes中
        self.boxes.append(bxs)

    def _layouts_rec(self, ZM, drop=True):
        """
       递归处理页面元素的版面布局。

       此方法根据布局结果调整元素位置，确保元素根据其所在页码及页面累计高度正确放置。

       参数:
       缩放因子ZM : int
           用于调整布局的缩放因子。
       drop : bool, 可选
           指示是否丢弃无法布局的元素，默认为True。

       返回:
           无
       """
        # 断言页面图片数量与边界框数量相等
        assert len(self.page_images) == len(self.boxes)
        # 使用布局引擎处理页面图片和边界框，并更新布局结果
        self.boxes, self.page_layout = self.layouter(
            self.page_images, self.boxes, ZM, drop=drop)
        # 根据所在页面的累计高度更新每个边界框的顶部和底部位置
        # 累计Y轴高度
        # cumlative Y
        for i in range(len(self.boxes)):
            # 将页面的累计高度加到边界框的顶部和底部
            self.boxes[i]["top"] += \
                self.page_cum_height[self.boxes[i]["page_number"] - 1]
            self.boxes[i]["bottom"] += \
                self.page_cum_height[self.boxes[i]["page_number"] - 1]

    def _text_merge(self):
        """
        合并符合条件的文字框以增强文本连贯性。

        此函数首先定义两个辅助函数，用于检查一个文字框是否以特定文本结束或开始。
        随后遍历文字框列表，寻找应根据其布局特性和位置关系合并的相邻文字框。
        合并的条件包括相同的布局编号、垂直方向的接近性以及特定的文本关联。
        合并后更新文字框于原位，并从列表中移除后续文字框。
        最终，将更新后的文字框列表重新分配给实例变量。
        """

        # 获取文字框列表以进行后续处理
        # merge adjusted boxes
        bxs = self.boxes

        def end_with(b, txt):
            """
           判断文字框b的文本是否以指定txt结束。

           :param b: 待检查的文字框。
           :param txt: 指定的文本。
           :return: 若b的文本以txt结束则为True，否则为False。
           """
            txt = txt.strip()
            tt = b.get("text", "").strip()
            return tt and tt.find(txt) == len(tt) - len(txt)

        def start_with(b, txts):
            """
           判断文字框b的文本是否以txts中的任一文本项开始。

           :param b: 待检查的文字框。
           :param txts: 指定的文本列表。
           :return: 若b的文本以txts中的某项开始则为True，否则为False。
           """
            tt = b.get("text", "").strip()
            return tt and any([tt.find(t.strip()) == 0 for t in txts])

        bxs = self.boxes
        i = 0
        while i < len(bxs) - 1:
            b = bxs[i]
            b_ = bxs[i + 1]

            # 若布局编号不同或当前布局类型为特殊类型，则跳过
            if b.get("layoutno", "0") != b_.get("layoutno", "1") or b.get("layout_type", "") in ["table", "figure",
                                                                                                 "equation"]:
                i += 1
                continue

            # 垂直距离足够小则合并
            if abs(self._y_dis(b, b_)) < self.mean_height[b["page_number"] - 1] / 3:
                b["x1"] = b_["x1"]
                b["top"] = (b["top"] + b_["top"]) / 2
                b["bottom"] = (b["bottom"] + b_["bottom"]) / 2
                b["text"] += b_["text"]
                bxs.pop(i + 1)
                continue

            # 判断两个文本框的水平距离
            dis = b["x1"] - b_["x0"]
            dis_thr = 1
            if b.get("layout_type", "") != "text" or b_.get("layout_type", "") != "text":
                if end_with(b, "，") or start_with(b_, ["（，"]):
                    dis_thr = -8
                else:
                    i += 1
                    continue

            # 垂直距离与水平距离的综合判断
            if abs(self._y_dis(b, b_)) < self.mean_height[b["page_number"] - 1] / 5 and dis >= dis_thr and b["x1"] < b_[
                "x1"]:
                b["x1"] = b_["x1"]
                b["top"] = (b["top"] + b_["top"]) / 2
                b["bottom"] = (b["bottom"] + b_["bottom"]) / 2
                b["text"] += b_["text"]
                bxs.pop(i + 1)
                continue

            i += 1

        self.boxes = bxs

    def _naive_vertical_merge(self):
        bxs = Recognizer.sort_Y_firstly(
            self.boxes, np.median(
                self.mean_height) / 3)
        i = 0
        while i + 1 < len(bxs):
            b = bxs[i]
            b_ = bxs[i + 1]
            if b["page_number"] < b_["page_number"] and re.match(
                    r"[0-9  •一—-]+$", b["text"]):
                bxs.pop(i)
                continue
            if not b["text"].strip():
                bxs.pop(i)
                continue
            concatting_feats = [
                b["text"].strip()[-1] in ",;:'\"，、‘“；：-",
                len(b["text"].strip()) > 1 and b["text"].strip(
                )[-2] in ",;:'\"，‘“、；：",
                b_["text"].strip() and b_["text"].strip()[0] in "。；？！?”）),，、：",
            ]
            # features for not concating
            feats = [
                b.get("layoutno", 0) != b_.get("layoutno", 0),
                b["text"].strip()[-1] in "。？！?",
                self.is_english and b["text"].strip()[-1] in ".!?",
                b["page_number"] == b_["page_number"] and b_["top"] -
                b["bottom"] > self.mean_height[b["page_number"] - 1] * 1.5,
                b["page_number"] < b_["page_number"] and abs(
                    b["x0"] - b_["x0"]) > self.mean_width[b["page_number"] - 1] * 4,
            ]
            # split features
            detach_feats = [b["x1"] < b_["x0"],
                            b["x0"] > b_["x1"]]
            if (any(feats) and not any(concatting_feats)) or any(detach_feats):
                print(
                    b["text"],
                    b_["text"],
                    any(feats),
                    any(concatting_feats),
                    any(detach_feats))
                i += 1
                continue
            # merge up and down
            b["bottom"] = b_["bottom"]
            b["text"] += b_["text"]
            b["x0"] = min(b["x0"], b_["x0"])
            b["x1"] = max(b["x1"], b_["x1"])
            bxs.pop(i + 1)
        self.boxes = bxs

    def _concat_downward(self, concat_between_pages=True):
        # count boxes in the same row as a feature
        for i in range(len(self.boxes)):
            mh = self.mean_height[self.boxes[i]["page_number"] - 1]
            self.boxes[i]["in_row"] = 0
            j = max(0, i - 12)
            while j < min(i + 12, len(self.boxes)):
                if j == i:
                    j += 1
                    continue
                ydis = self._y_dis(self.boxes[i], self.boxes[j]) / mh
                if abs(ydis) < 1:
                    self.boxes[i]["in_row"] += 1
                elif ydis > 0:
                    break
                j += 1

        # concat between rows
        boxes = deepcopy(self.boxes)
        blocks = []
        while boxes:
            chunks = []

            def dfs(up, dp):
                chunks.append(up)
                i = dp
                while i < min(dp + 12, len(boxes)):
                    ydis = self._y_dis(up, boxes[i])
                    smpg = up["page_number"] == boxes[i]["page_number"]
                    mh = self.mean_height[up["page_number"] - 1]
                    mw = self.mean_width[up["page_number"] - 1]
                    if smpg and ydis > mh * 4:
                        break
                    if not smpg and ydis > mh * 16:
                        break
                    down = boxes[i]
                    if not concat_between_pages and down["page_number"] > up["page_number"]:
                        break

                    if up.get("R", "") != down.get(
                            "R", "") and up["text"][-1] != "，":
                        i += 1
                        continue

                    if re.match(r"[0-9]{2,3}/[0-9]{3}$", up["text"]) \
                            or re.match(r"[0-9]{2,3}/[0-9]{3}$", down["text"]) \
                            or not down["text"].strip():
                        i += 1
                        continue

                    if not down["text"].strip():
                        i += 1
                        continue

                    if up["x1"] < down["x0"] - 10 * \
                            mw or up["x0"] > down["x1"] + 10 * mw:
                        i += 1
                        continue

                    if i - dp < 5 and up.get("layout_type") == "text":
                        if up.get("layoutno", "1") == down.get(
                                "layoutno", "2"):
                            dfs(down, i + 1)
                            boxes.pop(i)
                            return
                        i += 1
                        continue

                    fea = self._updown_concat_features(up, down)
                    if self.updown_cnt_mdl.predict(
                            xgb.DMatrix([fea]))[0] <= 0.5:
                        i += 1
                        continue
                    dfs(down, i + 1)
                    boxes.pop(i)
                    return

            dfs(boxes[0], 1)
            boxes.pop(0)
            if chunks:
                blocks.append(chunks)

        # concat within each block
        boxes = []
        for b in blocks:
            if len(b) == 1:
                boxes.append(b[0])
                continue
            t = b[0]
            for c in b[1:]:
                t["text"] = t["text"].strip()
                c["text"] = c["text"].strip()
                if not c["text"]:
                    continue
                if t["text"] and re.match(
                        r"[0-9\.a-zA-Z]+$", t["text"][-1] + c["text"][-1]):
                    t["text"] += " "
                t["text"] += c["text"]
                t["x0"] = min(t["x0"], c["x0"])
                t["x1"] = max(t["x1"], c["x1"])
                t["page_number"] = min(t["page_number"], c["page_number"])
                t["bottom"] = c["bottom"]
                if not t["layout_type"] \
                        and c["layout_type"]:
                    t["layout_type"] = c["layout_type"]
            boxes.append(t)

        self.boxes = Recognizer.sort_Y_firstly(boxes, 0)

    def _filter_forpages(self):
        if not self.boxes:
            return
        findit = False
        i = 0
        while i < len(self.boxes):
            if not re.match(r"(contents|目录|目次|table of contents|致谢|acknowledge)$",
                            re.sub(r"( | |\u3000)+", "", self.boxes[i]["text"].lower())):
                i += 1
                continue
            findit = True
            eng = re.match(
                r"[0-9a-zA-Z :'.-]{5,}",
                self.boxes[i]["text"].strip())
            self.boxes.pop(i)
            if i >= len(self.boxes):
                break
            prefix = self.boxes[i]["text"].strip()[:3] if not eng else " ".join(
                self.boxes[i]["text"].strip().split(" ")[:2])
            while not prefix:
                self.boxes.pop(i)
                if i >= len(self.boxes):
                    break
                prefix = self.boxes[i]["text"].strip()[:3] if not eng else " ".join(
                    self.boxes[i]["text"].strip().split(" ")[:2])
            self.boxes.pop(i)
            if i >= len(self.boxes) or not prefix:
                break
            for j in range(i, min(i + 128, len(self.boxes))):
                if not re.match(prefix, self.boxes[j]["text"]):
                    continue
                for k in range(i, j):
                    self.boxes.pop(i)
                break
        if findit:
            return

        page_dirty = [0] * len(self.page_images)
        for b in self.boxes:
            if re.search(r"(··|··|··)", b["text"]):
                page_dirty[b["page_number"] - 1] += 1
        page_dirty = set([i + 1 for i, t in enumerate(page_dirty) if t > 3])
        if not page_dirty:
            return
        i = 0
        while i < len(self.boxes):
            if self.boxes[i]["page_number"] in page_dirty:
                self.boxes.pop(i)
                continue
            i += 1

    def _merge_with_same_bullet(self):
        i = 0
        while i + 1 < len(self.boxes):
            b = self.boxes[i]
            b_ = self.boxes[i + 1]
            if not b["text"].strip():
                self.boxes.pop(i)
                continue
            if not b_["text"].strip():
                self.boxes.pop(i + 1)
                continue

            if b["text"].strip()[0] != b_["text"].strip()[0] \
                    or b["text"].strip()[0].lower() in set("qwertyuopasdfghjklzxcvbnm") \
                    or rag_tokenizer.is_chinese(b["text"].strip()[0]) \
                    or b["top"] > b_["bottom"]:
                i += 1
                continue
            b_["text"] = b["text"] + "\n" + b_["text"]
            b_["x0"] = min(b["x0"], b_["x0"])
            b_["x1"] = max(b["x1"], b_["x1"])
            b_["top"] = b["top"]
            self.boxes.pop(i)

    def _extract_table_figure(self, need_image, ZM,
                              return_html, need_position):
        tables = {}
        figures = {}
        # extract figure and table boxes
        i = 0
        lst_lout_no = ""
        nomerge_lout_no = []
        while i < len(self.boxes):
            if "layoutno" not in self.boxes[i]:
                i += 1
                continue
            lout_no = str(self.boxes[i]["page_number"]) + \
                      "-" + str(self.boxes[i]["layoutno"])
            if TableStructureRecognizer.is_caption(self.boxes[i]) or self.boxes[i]["layout_type"] in ["table caption",
                                                                                                      "title",
                                                                                                      "figure caption",
                                                                                                      "reference"]:
                nomerge_lout_no.append(lst_lout_no)
            if self.boxes[i]["layout_type"] == "table":
                if re.match(r"(数据|资料|图表)*来源[:： ]", self.boxes[i]["text"]):
                    self.boxes.pop(i)
                    continue
                if lout_no not in tables:
                    tables[lout_no] = []
                tables[lout_no].append(self.boxes[i])
                self.boxes.pop(i)
                lst_lout_no = lout_no
                continue
            if need_image and self.boxes[i]["layout_type"] == "figure":
                if re.match(r"(数据|资料|图表)*来源[:： ]", self.boxes[i]["text"]):
                    self.boxes.pop(i)
                    continue
                if lout_no not in figures:
                    figures[lout_no] = []
                figures[lout_no].append(self.boxes[i])
                self.boxes.pop(i)
                lst_lout_no = lout_no
                continue
            i += 1

        # merge table on different pages
        nomerge_lout_no = set(nomerge_lout_no)
        tbls = sorted([(k, bxs) for k, bxs in tables.items()],
                      key=lambda x: (x[1][0]["top"], x[1][0]["x0"]))

        i = len(tbls) - 1
        while i - 1 >= 0:
            k0, bxs0 = tbls[i - 1]
            k, bxs = tbls[i]
            i -= 1
            if k0 in nomerge_lout_no:
                continue
            if bxs[0]["page_number"] == bxs0[0]["page_number"]:
                continue
            if bxs[0]["page_number"] - bxs0[0]["page_number"] > 1:
                continue
            mh = self.mean_height[bxs[0]["page_number"] - 1]
            if self._y_dis(bxs0[-1], bxs[0]) > mh * 23:
                continue
            tables[k0].extend(tables[k])
            del tables[k]

        def x_overlapped(a, b):
            return not any([a["x1"] < b["x0"], a["x0"] > b["x1"]])

        # find captions and pop out
        i = 0
        while i < len(self.boxes):
            c = self.boxes[i]
            # mh = self.mean_height[c["page_number"]-1]
            if not TableStructureRecognizer.is_caption(c):
                i += 1
                continue

            # find the nearest layouts
            def nearest(tbls):
                nonlocal c
                mink = ""
                minv = 1000000000
                for k, bxs in tbls.items():
                    for b in bxs:
                        if b.get("layout_type", "").find("caption") >= 0:
                            continue
                        y_dis = self._y_dis(c, b)
                        x_dis = self._x_dis(
                            c, b) if not x_overlapped(
                            c, b) else 0
                        dis = y_dis * y_dis + x_dis * x_dis
                        if dis < minv:
                            mink = k
                            minv = dis
                return mink, minv

            tk, tv = nearest(tables)
            fk, fv = nearest(figures)
            # if min(tv, fv) > 2000:
            #    i += 1
            #    continue
            if tv < fv and tk:
                tables[tk].insert(0, c)
                logging.debug(
                    "TABLE:" +
                    self.boxes[i]["text"] +
                    "; Cap: " +
                    tk)
            elif fk:
                figures[fk].insert(0, c)
                logging.debug(
                    "FIGURE:" +
                    self.boxes[i]["text"] +
                    "; Cap: " +
                    tk)
            self.boxes.pop(i)

        res = []
        positions = []

        def cropout(bxs, ltype, poss):
            nonlocal ZM
            pn = set([b["page_number"] - 1 for b in bxs])
            if len(pn) < 2:
                pn = list(pn)[0]
                ht = self.page_cum_height[pn]
                b = {
                    "x0": np.min([b["x0"] for b in bxs]),
                    "top": np.min([b["top"] for b in bxs]) - ht,
                    "x1": np.max([b["x1"] for b in bxs]),
                    "bottom": np.max([b["bottom"] for b in bxs]) - ht
                }
                louts = [l for l in self.page_layout[pn] if l["type"] == ltype]
                ii = Recognizer.find_overlapped(b, louts, naive=True)
                if ii is not None:
                    b = louts[ii]
                else:
                    logging.warn(
                        f"Missing layout match: {pn + 1},%s" %
                        (bxs[0].get(
                            "layoutno", "")))

                left, top, right, bott = b["x0"], b["top"], b["x1"], b["bottom"]
                if right < left: right = left + 1
                poss.append((pn + self.page_from, left, right, top, bott))
                return self.page_images[pn] \
                    .crop((left * ZM, top * ZM,
                           right * ZM, bott * ZM))
            pn = {}
            for b in bxs:
                p = b["page_number"] - 1
                if p not in pn:
                    pn[p] = []
                pn[p].append(b)
            pn = sorted(pn.items(), key=lambda x: x[0])
            imgs = [cropout(arr, ltype, poss) for p, arr in pn]
            pic = Image.new("RGB",
                            (int(np.max([i.size[0] for i in imgs])),
                             int(np.sum([m.size[1] for m in imgs]))),
                            (245, 245, 245))
            height = 0
            for img in imgs:
                pic.paste(img, (0, int(height)))
                height += img.size[1]
            return pic

        # crop figure out and add caption
        for k, bxs in figures.items():
            txt = "\n".join([b["text"] for b in bxs])
            if not txt:
                continue

            poss = []
            res.append(
                (cropout(
                    bxs,
                    "figure", poss),
                 [txt]))
            positions.append(poss)

        for k, bxs in tables.items():
            if not bxs:
                continue
            bxs = Recognizer.sort_Y_firstly(bxs, np.mean(
                [(b["bottom"] - b["top"]) / 2 for b in bxs]))
            poss = []
            res.append((cropout(bxs, "table", poss),
                        self.tbl_det.construct_table(bxs, html=return_html, is_english=self.is_english)))
            positions.append(poss)

        assert len(positions) == len(res)

        if need_position:
            return list(zip(res, positions))
        return res

    def proj_match(self, line):
        if len(line) <= 2:
            return
        if re.match(r"[0-9 ().,%%+/-]+$", line):
            return False
        for p, j in [
            (r"第[零一二三四五六七八九十百]+章", 1),
            (r"第[零一二三四五六七八九十百]+[条节]", 2),
            (r"[零一二三四五六七八九十百]+[、 　]", 3),
            (r"[\(（][零一二三四五六七八九十百]+[）\)]", 4),
            (r"[0-9]+(、|\.[　 ]|\.[^0-9])", 5),
            (r"[0-9]+\.[0-9]+(、|[. 　]|[^0-9])", 6),
            (r"[0-9]+\.[0-9]+\.[0-9]+(、|[ 　]|[^0-9])", 7),
            (r"[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+(、|[ 　]|[^0-9])", 8),
            (r".{,48}[：:?？]$", 9),
            (r"[0-9]+）", 10),
            (r"[\(（][0-9]+[）\)]", 11),
            (r"[零一二三四五六七八九十百]+是", 12),
            (r"[⚫•➢✓]", 12)
        ]:
            if re.match(p, line):
                return j
        return

    def _line_tag(self, bx, ZM):
        pn = [bx["page_number"]]
        top = bx["top"] - self.page_cum_height[pn[0] - 1]
        bott = bx["bottom"] - self.page_cum_height[pn[0] - 1]
        page_images_cnt = len(self.page_images)
        if pn[-1] - 1 >= page_images_cnt: return ""
        while bott * ZM > self.page_images[pn[-1] - 1].size[1]:
            bott -= self.page_images[pn[-1] - 1].size[1] / ZM
            pn.append(pn[-1] + 1)
            if pn[-1] - 1 >= page_images_cnt:
                return ""

        return "@@{}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}##" \
            .format("-".join([str(p) for p in pn]),
                    bx["x0"], bx["x1"], top, bott)

    def __filterout_scraps(self, boxes, ZM):

        def width(b):
            return b["x1"] - b["x0"]

        def height(b):
            return b["bottom"] - b["top"]

        def usefull(b):
            if b.get("layout_type"):
                return True
            if width(
                    b) > self.page_images[b["page_number"] - 1].size[0] / ZM / 3:
                return True
            if b["bottom"] - b["top"] > self.mean_height[b["page_number"] - 1]:
                return True
            return False

        res = []
        while boxes:
            lines = []
            widths = []
            pw = self.page_images[boxes[0]["page_number"] - 1].size[0] / ZM
            mh = self.mean_height[boxes[0]["page_number"] - 1]
            mj = self.proj_match(
                boxes[0]["text"]) or boxes[0].get(
                "layout_type",
                "") == "title"

            def dfs(line, st):
                nonlocal mh, pw, lines, widths
                lines.append(line)
                widths.append(width(line))
                width_mean = np.mean(widths)
                mmj = self.proj_match(
                    line["text"]) or line.get(
                    "layout_type",
                    "") == "title"
                for i in range(st + 1, min(st + 20, len(boxes))):
                    if (boxes[i]["page_number"] - line["page_number"]) > 0:
                        break
                    if not mmj and self._y_dis(
                            line, boxes[i]) >= 3 * mh and height(line) < 1.5 * mh:
                        break

                    if not usefull(boxes[i]):
                        continue
                    if mmj or \
                            (self._x_dis(boxes[i], line) < pw / 10): \
                            # and abs(width(boxes[i])-width_mean)/max(width(boxes[i]),width_mean)<0.5):
                        # concat following
                        dfs(boxes[i], i)
                        boxes.pop(i)
                        break

            try:
                if usefull(boxes[0]):
                    dfs(boxes[0], 0)
                else:
                    logging.debug("WASTE: " + boxes[0]["text"])
            except Exception as e:
                pass
            boxes.pop(0)
            mw = np.mean(widths)
            if mj or mw / pw >= 0.35 or mw > 200:
                res.append(
                    "\n".join([c["text"] + self._line_tag(c, ZM) for c in lines]))
            else:
                logging.debug("REMOVED: " +
                              "<<".join([c["text"] for c in lines]))

        return "\n\n".join(res)

    @staticmethod
    def total_page_number(fnm, binary=None):
        try:
            pdf = pdfplumber.open(
                fnm) if not binary else pdfplumber.open(BytesIO(binary))
            return len(pdf.pages)
        except Exception as e:
            logging.error(str(e))

    def __images__(self, fnm, zoomin=3, page_from=0,
                   page_to=299, callback=None):
        self.lefted_chars = []
        self.mean_height = []
        self.mean_width = []
        self.boxes = []
        self.garbages = {}
        self.page_cum_height = [0]
        self.page_layout = []
        self.page_from = page_from
        st = timer()
        try:
            self.pdf = pdfplumber.open(fnm) if isinstance(
                fnm, str) else pdfplumber.open(BytesIO(fnm))
            self.page_images = [p.to_image(resolution=72 * zoomin).annotated for i, p in
                                enumerate(self.pdf.pages[page_from:page_to])]
            self.page_chars = [[c for c in page.chars if self._has_color(c)] for page in
                               self.pdf.pages[page_from:page_to]]
            self.total_page = len(self.pdf.pages)
        except Exception as e:
            logging.error(str(e))

        self.outlines = []
        try:
            self.pdf = pdf2_read(fnm if isinstance(fnm, str) else BytesIO(fnm))
            outlines = self.pdf.outline

            def dfs(arr, depth):
                for a in arr:
                    if isinstance(a, dict):
                        self.outlines.append((a["/Title"], depth))
                        continue
                    dfs(a, depth + 1)

            dfs(outlines, 0)
        except Exception as e:
            logging.warning(f"Outlines exception: {e}")
        if not self.outlines:
            logging.warning(f"Miss outlines")

        logging.info("Images converted.")
        self.is_english = [re.search(r"[a-zA-Z0-9,/¸;:'\[\]\(\)!@#$%^&*\"?<>._-]{30,}", "".join(
            random.choices([c["text"] for c in self.page_chars[i]], k=min(100, len(self.page_chars[i]))))) for i in
                           range(len(self.page_chars))]
        if sum([1 if e else 0 for e in self.is_english]) > len(
                self.page_images) / 2:
            self.is_english = True
        else:
            self.is_english = False
        self.is_english = False

        st = timer()
        for i, img in enumerate(self.page_images):
            chars = self.page_chars[i] if not self.is_english else []
            self.mean_height.append(
                np.median(sorted([c["height"] for c in chars])) if chars else 0
            )
            self.mean_width.append(
                np.median(sorted([c["width"] for c in chars])) if chars else 8
            )
            self.page_cum_height.append(img.size[1] / zoomin)
            j = 0
            while j + 1 < len(chars):
                if chars[j]["text"] and chars[j + 1]["text"] \
                        and re.match(r"[0-9a-zA-Z,.:;!%]+", chars[j]["text"] + chars[j + 1]["text"]) \
                        and chars[j + 1]["x0"] - chars[j]["x1"] >= min(chars[j + 1]["width"],
                                                                       chars[j]["width"]) / 2:
                    chars[j]["text"] += " "
                j += 1

            self.__ocr(i + 1, img, chars, zoomin)
            if callback and i % 6 == 5:
                callback(prog=(i + 1) * 0.6 / len(self.page_images), msg="")
        # print("OCR:", timer()-st)

        if not self.is_english and not any(
                [c for c in self.page_chars]) and self.boxes:
            bxes = [b for bxs in self.boxes for b in bxs]
            self.is_english = re.search(r"[\na-zA-Z0-9,/¸;:'\[\]\(\)!@#$%^&*\"?<>._-]{30,}",
                                        "".join([b["text"] for b in random.choices(bxes, k=min(30, len(bxes)))]))

        logging.info("Is it English:", self.is_english)

        self.page_cum_height = np.cumsum(self.page_cum_height)
        assert len(self.page_cum_height) == len(self.page_images) + 1
        if len(self.boxes) == 0 and zoomin < 9: self.__images__(fnm, zoomin * 3, page_from,
                                                                page_to, callback)

    def __call__(self, fnm, need_image=True, zoomin=3, return_html=False):
        self.__images__(fnm, zoomin)
        self._layouts_rec(zoomin)
        self._table_transformer_job(zoomin)
        self._text_merge()
        self._concat_downward()
        self._filter_forpages()
        tbls = self._extract_table_figure(
            need_image, zoomin, return_html, False)
        return self.__filterout_scraps(deepcopy(self.boxes), zoomin), tbls

    def remove_tag(self, txt):
        return re.sub(r"@@[\t0-9.-]+?##", "", txt)

    def crop(self, text, ZM=3, need_position=False):
        imgs = []
        poss = []
        for tag in re.findall(r"@@[0-9-]+\t[0-9.\t]+##", text):
            pn, left, right, top, bottom = tag.strip(
                "#").strip("@").split("\t")
            left, right, top, bottom = float(left), float(
                right), float(top), float(bottom)
            poss.append(([int(p) - 1 for p in pn.split("-")],
                         left, right, top, bottom))
        if not poss:
            if need_position:
                return None, None
            return

        max_width = max(
            np.max([right - left for (_, left, right, _, _) in poss]), 6)
        GAP = 6
        pos = poss[0]
        poss.insert(0, ([pos[0][0]], pos[1], pos[2], max(
            0, pos[3] - 120), max(pos[3] - GAP, 0)))
        pos = poss[-1]
        poss.append(([pos[0][-1]], pos[1], pos[2], min(self.page_images[pos[0][-1]].size[1] / ZM, pos[4] + GAP),
                     min(self.page_images[pos[0][-1]].size[1] / ZM, pos[4] + 120)))

        positions = []
        for ii, (pns, left, right, top, bottom) in enumerate(poss):
            right = left + max_width
            bottom *= ZM
            for pn in pns[1:]:
                bottom += self.page_images[pn - 1].size[1]
            imgs.append(
                self.page_images[pns[0]].crop((left * ZM, top * ZM,
                                               right *
                                               ZM, min(
                    bottom, self.page_images[pns[0]].size[1])
                                               ))
            )
            if 0 < ii < len(poss) - 1:
                positions.append((pns[0] + self.page_from, left, right, top, min(
                    bottom, self.page_images[pns[0]].size[1]) / ZM))
            bottom -= self.page_images[pns[0]].size[1]
            for pn in pns[1:]:
                imgs.append(
                    self.page_images[pn].crop((left * ZM, 0,
                                               right * ZM,
                                               min(bottom,
                                                   self.page_images[pn].size[1])
                                               ))
                )
                if 0 < ii < len(poss) - 1:
                    positions.append((pn + self.page_from, left, right, 0, min(
                        bottom, self.page_images[pn].size[1]) / ZM))
                bottom -= self.page_images[pn].size[1]

        if not imgs:
            if need_position:
                return None, None
            return
        height = 0
        for img in imgs:
            height += img.size[1] + GAP
        height = int(height)
        width = int(np.max([i.size[0] for i in imgs]))
        pic = Image.new("RGB",
                        (width, height),
                        (245, 245, 245))
        height = 0
        for ii, img in enumerate(imgs):
            if ii == 0 or ii + 1 == len(imgs):
                img = img.convert('RGBA')
                overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
                overlay.putalpha(128)
                img = Image.alpha_composite(img, overlay).convert("RGB")
            pic.paste(img, (0, int(height)))
            height += img.size[1] + GAP

        if need_position:
            return pic, positions
        return pic

    def get_position(self, bx, ZM):
        poss = []
        pn = bx["page_number"]
        top = bx["top"] - self.page_cum_height[pn - 1]
        bott = bx["bottom"] - self.page_cum_height[pn - 1]
        poss.append((pn, bx["x0"], bx["x1"], top, min(
            bott, self.page_images[pn - 1].size[1] / ZM)))
        while bott * ZM > self.page_images[pn - 1].size[1]:
            bott -= self.page_images[pn - 1].size[1] / ZM
            top = 0
            pn += 1
            poss.append((pn, bx["x0"], bx["x1"], top, min(
                bott, self.page_images[pn - 1].size[1] / ZM)))
        return poss


class PlainParser(object):
    def __call__(self, filename, from_page=0, to_page=100000, **kwargs):
        self.outlines = []
        lines = []
        try:
            self.pdf = pdf2_read(
                filename if isinstance(
                    filename, str) else BytesIO(filename))
            for page in self.pdf.pages[from_page:to_page]:
                lines.extend([t for t in page.extract_text().split("\n")])

            outlines = self.pdf.outline

            def dfs(arr, depth):
                for a in arr:
                    if isinstance(a, dict):
                        self.outlines.append((a["/Title"], depth))
                        continue
                    dfs(a, depth + 1)

            dfs(outlines, 0)
        except Exception as e:
            logging.warning(f"Outlines exception: {e}")
        if not self.outlines:
            logging.warning(f"Miss outlines")

        return [(l, "") for l in lines], []

    def crop(self, ck, need_position):
        raise NotImplementedError

    @staticmethod
    def remove_tag(txt):
        raise NotImplementedError


if __name__ == "__main__":
    pass

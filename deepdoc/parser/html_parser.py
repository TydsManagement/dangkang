from rag.nlp import find_codec
import readability
import html_text
import chardet

def get_encoding(file):
    """
    通过chardet库检测文件的编码方式。

    参数:
    file: 字符串，文件的路径。

    返回:
    字符串，文件的编码方式。
    """
    with open(file,'rb') as f:
        tmp = chardet.detect(f.read())
        return tmp['encoding']


class RAGFlowHtmlParser:
    """
    RAGFlowHtmlParser类用于解析HTML文件或二进制数据，提取标题和内容。
    该类实现了可调用接口，可以通过实例直接调用，传入文件名或二进制数据。
    """

    def __call__(self, fnm, binary=None):
        """
        解析HTML文件或二进制数据，提取标题和内容。

        参数:
        fnm: 字符串，HTML文件的路径。如果传入二进制数据，则该参数应为None。
        binary: 二进制数据，包含HTML内容。如果为None，则从文件中读取内容。

        返回:
        列表，包含标题和内容的字符串。
        """
        txt = ""
        if binary:
            # 如果传入的是二进制数据，检测并解码其编码方式
            encoding = find_codec(binary)
            txt = binary.decode(encoding, errors="ignore")
        else:
            # 如果传入的是文件名，打开文件并读取内容
            with open(fnm, "r",encoding=get_encoding(fnm)) as f:
                txt = f.read()

        # 使用readability库解析HTML内容，提取标题和摘要
        html_doc = readability.Document(txt)
        title = html_doc.title()
        content = html_text.extract_text(html_doc.summary(html_partial=True))
        txt = f'{title}\n{content}'
        # 将标题和内容分割成列表返回
        sections = txt.split("\n")
        return sections

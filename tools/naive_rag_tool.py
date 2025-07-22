# 从langchain.tools.retriever模块中导入create_retriever_tool函数，用于创建检索工具
from langchain.tools.retriever import create_retriever_tool
# 导入os模块，用于处理文件和目录路径
import os
# 从utils模块中导入get_embedding_model函数，用于获取嵌入模型
from utils import get_embedding_model


# 工具函数：返回原始 retriever
def get_naive_rag_tool(vectorstore_name: str, as_tool: bool = False):
    import os
    from langchain_chroma import Chroma
    from langchain.tools.retriever import create_retriever_tool
    from utils import get_embedding_model  # 请根据你项目实际路径导入

    # 拼接向量数据库路径
    persist_directory = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "kb", vectorstore_name, "vectorstore"
    )

    # 初始化 Chroma 向量数据库
    vectorstore = Chroma(
        collection_name=vectorstore_name,
        embedding_function=get_embedding_model(platform_type="Ollama"),
        persist_directory=persist_directory,
    )

    # 打印调试信息：检查是否加载成功
    print(f"📦 向量库 `{vectorstore_name}` 加载成功，包含文档数量：{vectorstore._collection.count()}")

    # 创建检索器（建议先不加过滤）
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )

    # 返回原始 retriever，用于拼 prompt
    if not as_tool:
        return retriever

    # 否则封装成 tool
    retriever_tool = create_retriever_tool(
        retriever=retriever,
        name=f"{vectorstore_name}_knowledge_base_tool",
        description=f"search and return information about {vectorstore_name}",
    )
    retriever_tool.response_format = "content"

    # 添加工具的自定义返回格式（dict）
    retriever_tool.func = lambda query: {
        f"已知内容 {i+1}": doc.page_content
        for i, doc in enumerate(retriever.get_relevant_documents(query))
    }

    return retriever_tool


# 如果当前模块是主程序入口
if __name__ == "__main__":
    # 调用get_naive_rag_tool函数，传入"personal_information"作为参数，获取检索工具
    retriever_tool = get_naive_rag_tool("personal_information")
    # 打印检索工具对查询"刘虔"的响应结果
    print(retriever_tool.invoke("刘虔"))
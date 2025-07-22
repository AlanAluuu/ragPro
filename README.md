# 智能客服系統
构建基于 RAG 的问答系统，结合 LangChain 框架、Streamlit 前端界面、Chroma 向量数据库、多知识库检索和聊天交互功能。



### 1. RAG（Retrieval-Augmented Generation）

**核心思想：检索 + 生成**

* **检索模块**：通过用户的问题去知识库中寻找相关文档。
* **生成模块**：将检索到的内容嵌入 prompt 中，由大语言模型（如 qwen3）生成自然语言答案。


### 2. LangChain 框架

* 构建对话流程
* 工具调用调度
* 消息状态管理（`MessagesState`）
* 检索器封装为 Tool（LangChain Tool API）


### 3. Streamlit 前端框架

Streamlit 用于构建 Web 应用界面，负责：

* 知识库创建，多知识库选择
* 聊天输入/输出


### 4. 嵌入模型（Embedding Model）

使用 nomic-embed-text 模型进行嵌入，RecursiveCharacterTextSplitter 进行切分

### 5. 向量数据库

使用 langchain_chroma.Chroma 来管理每个知识库的向量表示，并完成文档检索：

### 6. 检索策略（召回策略）

* 使用向量相似度检索
* Top-K 策略（召回前5条相关内容）


### 7. 多知识库支持

RAG 图中支持一次查询多个知识库，循环拼接上下文。


### 8. 工具调用（ToolNode + ToolMessage）

LangChain 中的每个知识库封装为 `Tool`，用于在 LLM 推理过程中自动调用。

## 运行环境
建议使用 Python>=3.10
可参考如下命令进行环境创建
```commandline
conda create -n agent python=3.10 -y
conda activate agent
```
安装依赖
```commandline
pip install -r requirements.txt
```

使用以下命令行运行webui
```bash
streamlit run rag.py
```


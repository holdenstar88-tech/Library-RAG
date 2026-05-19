# 智能校园图书 RAG 问答系统

这是一个基于 `LangChain + Milvus + DeepSeek API` 的校园图书问答项目。
系统流程是：文档解析 -> 文本切分 -> SQLite 元数据记录 -> Milvus 向量存储 -> BM25 + 向量检索 -> RRF 重排 -> 大模型生成回答。

## 功能

- Web 对话式问答
- Milvus 向量检索
- SQLite 元数据表，支持增量更新
- BM25 + 向量混合召回
- RRF 融合排序
- DeepSeek 生成回答
- Docker 本地部署

## 目录

- `app/api`：Web API
- `app/services`：检索、索引、向量库、元数据服务
- `app/ingestion`：文档加载与切分
- `app/retrieval`：BM25、RRF、过滤逻辑
- `app/generation`：回答生成
- `app/static`：前端页面
- `data/raw`：知识库原始文档
- `data/processed`：SQLite 元数据文件

## 运行方式

1. 配置环境变量

```bash
copy .env.example .env
```

2. 启动 Milvus 和应用

```bash
docker compose up -d --build
```

3. 手动同步知识库

```bash
python -m app.cli sync
```

4. 启动本地服务

```bash
uvicorn app.api.main:app --reload
```

5. 打开页面

```text
http://127.0.0.1:8000
```

## 增量更新规则

- 新增文件：入库新版本，写入 SQLite 元数据表和 Milvus
- 修改文件：旧版本标记为 `superseded`，新版本重新写入
- 删除文件：旧版本标记为 `deleted`，同时删除对应 Milvus 向量
- 检索时只使用 `status=active` 的数据

## 配置项

建议在 `.env` 中配置：

- `API_KEY`
- `EMBEDDING_API_KEY`
- `EMBEDDING_BASE_URL`
- `EMBEDDING_MODEL`
- `DEEPSEEK_API_KEY`
- `DEEPSEEK_BASE_URL`
- `DEEPSEEK_MODEL`
- `MILVUS_URI`
- `MILVUS_COLLECTION`
- `METADATA_DB_PATH`

## 备注

- 召回采用 `BM25 + 向量检索`
- 单路召回 `Top-3`
- 重排使用 `RRF`
- 低置信度时会返回澄清式回答，而不是强行编造

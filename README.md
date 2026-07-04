# 智能校园图书 RAG 问答系统

这是一个对标图书馆 OPAC/馆藏查询体验的校园图书 RAG 项目。系统支持精确馆藏编号查询、ISBN/索书号检索、通俗分类浏览、书架行列定位，以及基于主角、主题词和书籍大意的语义检索。

系统流程：数据导入校验 -> 文档切分 -> SQLite 元数据记录 -> Milvus 向量存储 -> 精确匹配 + BM25 + 向量检索 -> RRF 融合排序 -> DeepSeek 生成回答。

## 功能

- 馆藏查询界面：自然语言搜索、分类筛选、高级检索、详情面板
- 通俗分类：文学、历史、科幻、计算机、艺术、哲学、社科、教育、医学、自然科学、经济管理
- 精确馆藏字段：馆藏编号、ISBN、索书号、书架号、行、列、楼层、区域
- 内容语义检索：支持通过主角名、主题词、剧情/书籍大意寻找书籍
- 混合检索：BM25 + Milvus 向量检索 + RRF 融合排序
- 幻觉控制：低置信度时返回澄清式回答
- 批量导入：支持 JSON 和 CSV，缺少关键字段时拒绝入库

## 数据字段

每条图书记录建议包含以下字段：

| 字段 | 含义 | 是否必填 |
| --- | --- | --- |
| `book_id` | 精确馆藏编号或条码号 | 是 |
| `title` | 书名 | 是 |
| `author` | 作者 | 否 |
| `isbn` | ISBN | 否 |
| `call_number` | 索书号/分类排架号 | 否 |
| `category` | 通俗分类 | 是 |
| `subjects` | 主题词，JSON 可用数组，CSV 用分号分隔 | 否 |
| `main_characters` | 主角/人物，JSON 可用数组，CSV 用分号分隔 | 否 |
| `plot_summary` | 书籍大意/内容概述 | 否 |
| `shelf_code` | 书架号，如 `F` | 是 |
| `shelf_row` | 书架第几行 | 是 |
| `shelf_col` | 书架第几列 | 是 |
| `floor` | 楼层 | 否 |
| `area` | 馆内区域 | 否 |
| `copy_count` | 馆藏册数 | 否 |
| `available_count` | 可借册数 | 否 |
| `availability` | 馆藏状态 | 否 |
| `borrow_rule` | 借阅规则 | 否 |
| `open_time` | 开放时间 | 否 |

CSV 模板见 `data/templates/book_import_template.csv`。JSON 示例见 `data/raw/sample_books.json`。

位置编码示例：`shelf_code=F`、`shelf_row=1`、`shelf_col=2` 会展示为 `F书架 第1行 第2列`。

## 查询示例

- `馆藏编号 LIB-2026-0001`
- `ISBN 9787536692930`
- `索书号 I247.55/L63`
- `主角 孙悟空`
- `讲外星文明的科幻书`
- `关于明朝历史的可借图书`
- `分类:计算机 Python 数据分析`

## API

- `GET /api/health`：服务状态
- `POST /api/catalog/search`：馆藏检索
- `POST /api/chat`：RAG 智能问答
- `POST /api/reindex`：重新同步知识库

`/api/catalog/search` 请求示例：

```json
{
  "query": "主角 孙悟空",
  "category": "文学",
  "available_only": true,
  "limit": 12
}
```

## 运行方式

### Docker 一键运行

1. 配置环境变量

```bash
copy .env.example .env
```

当前默认使用阿里云 DashScope Embedding，`.env` 至少需要配置：

```env
EMBEDDING_BACKEND=dashscope
EMBEDDING_API_KEY=你的阿里云DashScope Key
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

DEEPSEEK_API_KEY=你的DeepSeek Key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

2. 构建并启动全部服务

```bash
docker compose up -d --build
```

启动后应包含以下容器：

```bash
docker compose ps
```

正常会看到 `rag-etcd`、`rag-minio`、`rag-milvus`、`rag-app` 均为运行状态。

3. 同步知识库

```bash
docker compose exec app python -m app.cli sync
```

4. 检查健康状态

```bash
curl http://127.0.0.1:8000/api/health
```

期望看到 `vector_store_ready=true`，并且 `documents_loaded` 大于 0。

5. 打开页面

```text
http://127.0.0.1:8000
```

### 本地运行后端

如果只用 Docker 启动 Milvus 依赖，后端也可以在本地运行：

```bash
docker compose up -d etcd minio milvus
pip install -r requirements.txt
python -m app.cli sync
uvicorn app.api.main:app --reload
```

### 常见问题

- `rag-app Exited (1)`：先执行 `docker compose logs --tail=160 app` 查看 Python 报错。
- 只有 `rag-etcd`、`rag-minio`、`rag-milvus`，没有 `rag-app`：通常是 app 镜像构建失败，执行 `docker compose build app --progress=plain` 看具体错误。
- Docker 构建时下载很慢：本项目默认走 DashScope Embedding，已不需要 `sentence-transformers` 和 `torch`；确认 `.env` 中存在 `EMBEDDING_BACKEND=dashscope`。
- PowerShell 中中文响应乱码：通常是控制台编码问题，浏览器页面使用 UTF-8 会正常显示。
- 重新导入数据后页面没刷新：执行 `docker compose exec app python -m app.cli sync` 后，再调用 `POST /api/reindex` 或重启 `rag-app`。

## 增量更新规则

- 新增文件：写入 SQLite 元数据表和 Milvus
- 修改文件：旧版本标记为 `superseded`，新版本重新写入
- 删除文件：旧版本标记为 `deleted`，同时删除对应 Milvus 向量
- 检索时只使用 `status=active` 的数据

## 对标方向

本项目借鉴真实图书馆系统常见能力：Koha 的 OPAC/编目/流通思路、LOC 分类体系中的索书号概念、VuFind 的分面检索体验、Evergreen 的馆藏与状态管理、WorldCat 的统一书目检索体验。当前版本聚焦馆藏查询和 RAG 问答，不包含登录、借还书、预约、罚款等流通业务。

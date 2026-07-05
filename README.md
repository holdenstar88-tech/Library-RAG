# Library RAG

一个面向校园图书馆场景的 RAG 检索与问答项目，提供接近 OPAC / 馆藏检索系统的体验：  
支持馆藏编号、ISBN、索书号等精确查询，也支持按主题词、主角、内容概述进行语义检索，并结合 DeepSeek 输出中文回答。

## Features

- 馆藏检索：自然语言搜索、分类筛选、高级字段检索、详情面板
- 混合检索：BM25 + Milvus 向量召回 + RRF 融合排序
- 智能问答：围绕馆藏信息、借阅规则、开放时间生成中文回答
- 低置信度兜底：证据不足时返回澄清式答案，减少幻觉
- 批量导入：支持 JSON / CSV 数据源
- 前端交互：亮暗模式、右下角悬浮助手、浮层聊天、可拖拽助手
- 分页：默认每页 10 条，支持 10 / 20 / 50 条切换

## Architecture

主流程如下：

`数据导入校验 -> 文档切分 -> SQLite 元数据记录 -> Milvus 向量存储 -> 精确匹配 + BM25 + 向量检索 -> RRF 融合排序 -> DeepSeek 生成回答`

### 模型职责

- `DeepSeek`：负责 `/api/chat` 的回答生成
- `Milvus`：只负责向量检索与召回，不负责回答生成
- `BM25`：提供关键词匹配补充召回
- `DashScope Embedding`（默认）：负责文本向量化

换句话说，AI 助手回答问题走的是 DeepSeek，向量模型只负责帮系统找到候选上下文。

## UI Highlights

- 默认浅色主题，可切换亮暗模式
- 右下角悬浮 AI 助手
- 助手支持打开浮层聊天
- 助手支持拖拽，位置可记忆
- 搜索结果与详情区域分栏展示

## Data Schema

建议每条图书记录包含以下字段：

| 字段 | 含义 | 必填 |
| --- | --- | --- |
| `book_id` | 精确馆藏编号或条码号 | 是 |
| `title` | 书名 | 是 |
| `author` | 作者 | 否 |
| `isbn` | ISBN | 否 |
| `call_number` | 索书号 / 分类排架号 | 否 |
| `category` | 通俗分类 | 是 |
| `subjects` | 主题词 | 否 |
| `main_characters` | 主角 / 人物 | 否 |
| `plot_summary` | 内容概述 | 否 |
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

参考文件：

- CSV 模板：`data/templates/book_import_template.csv`
- JSON 示例：`data/raw/sample_books.json`

位置编码示例：`shelf_code=F`、`shelf_row=1`、`shelf_col=2` 会展示为 `F书架 第1行 第2列`。

## Query Examples

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

`/api/catalog/search` 示例：

```json
{
  "query": "主角 孙悟空",
  "category": "文学",
  "available_only": true,
  "page": 1,
  "limit": 10
}
```

响应会返回：

- `total`
- `page`
- `limit`
- `total_pages`
- `has_prev`
- `has_next`

## Quick Start

### Docker

复制环境变量模板：

```bash
copy .env.example .env
```

至少需要配置：

```env
EMBEDDING_BACKEND=dashscope
EMBEDDING_API_KEY=your_dashscope_key
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

DEEPSEEK_API_KEY=your_deepseek_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

启动服务：

```bash
docker compose up -d --build
```

查看容器状态：

```bash
docker compose ps
```

同步知识库：

```bash
docker compose exec app python -m app.cli sync
```

检查健康状态：

```bash
curl http://127.0.0.1:8000/api/health
```

访问页面：

```text
http://127.0.0.1:8000
```

### Local Backend

如果只用 Docker 启动 Milvus 相关依赖，也可以本地运行后端：

```bash
docker compose up -d etcd minio milvus
pip install -r requirements.txt
python -m app.cli sync
uvicorn app.api.main:app --reload
```

## Development Notes

- 如果修改了 `app/static` 下的前端资源，而页面跑在 Docker 容器里，需要重建 `rag-app`：

```bash
docker compose up -d --build --force-recreate app
```

- 浏览器建议执行一次 `Ctrl+F5`

## FAQ

- `rag-app Exited (1)`：执行 `docker compose logs --tail=160 app`
- 没有 `rag-app`：执行 `docker compose build app --progress=plain`
- Docker 构建很慢：确认 `.env` 中使用的是 DashScope Embedding
- PowerShell 中文乱码：通常是控制台编码问题，浏览器页面仍应正常显示 UTF-8
- 导入后页面没变化：先执行 `sync`，再调用 `/api/reindex` 或重启 `rag-app`

## Privacy

- `.env`、`.env.*` 已被忽略，不应提交到仓库
- `DEEPSEEK_API_KEY`、`EMBEDDING_API_KEY` 只应保留在本地环境
- 提交到 GitHub 时只提交代码、文档和示例数据

## Reference Direction

项目参考了真实图书馆系统常见能力与体验：

- Koha
- LOC
- VuFind
- Evergreen
- WorldCat

当前版本聚焦馆藏查询与 RAG 问答，不包含登录、借还书、预约、罚款等流通业务。

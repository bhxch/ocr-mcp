# ocr-mcp

一个 stdio 模式的 OCR MCP server。多个 MCP client 共享**同一个常驻 PaddleOCR（PP-OCRv6）后端**——重量级模型只加载一次，推理串行复用。每个 MCP client 启动一个**瘦前端**（绝不 import paddle），前端们通过 Unix domain socket 与唯一的长驻 daemon 通信。

基于 **mcp 2.0**（`from mcp.server import MCPServer`）。

## 为什么这样设计

PaddleOCR 模型加载占用大量内存（CPU 推理几百 MB + 秒级加载）。标准做法是每个 MCP client 各自启动一个 server、各自加载模型，内存浪费严重。本项目让所有 client 共享一个常驻后端：

```
MCP client A ─┐
MCP client B ─┼─ stdio ─▶ 瘦前端(每 client 一个,轻量) ─┐
MCP client C ─┘                                          ├─ Unix socket ─▶ 常驻 daemon(唯一,只加载一次模型)
                                                         ┘
```

- daemon 懒启动（首个请求时拉起），引用计数 + 空闲超时（默认 600s）自动退出释放内存。
- 前端进程不加载 paddle，保持轻量。

## 安装

    uv sync

Python 3.11–3.12。依赖：`mcp[cli]>=2.0`、`paddleocr>=3.0`、`paddlepaddle>=3.0`、`platformdirs>=4.0`。

### 预下载模型（可选）

模型缓存在 `~/.paddlex/official_models/`。首次使用会从百度 bcebos 自动下载（需外网）。内网/离线机器可先在联网机器上跑一次：

    uv run ocr-mcp download

或通过 `OCR_MCP_MODEL_DIR` 指定已下载的模型目录。

## 配置 MCP client

任何支持 MCP 的 client（Claude Desktop、Claude Code、Cursor 等）通过 `uv run` 接入：

```json
{
  "mcpServers": {
    "ocr": {
      "command": "uv",
      "args": ["--directory", "/abs/path/ocr-mcp", "run", "ocr-mcp"]
    }
  }
}
```

（`ocr-mcp` 不带子命令默认走 `stdio`。）首个 `ocr_image` 调用会懒启动后端 daemon（每用户一个）。

也可以直接从 git 仓库运行（无需本地 clone，uv 会自动拉取指定分支）：

```json
{
  "mcpServers": {
    "ocr": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/bhxch/ocr-mcp.git@main", "ocr-mcp"]
    }
  }
}
```

`@main` 指定分支（需先将该分支 push 到 remote），可换成 `@<tag>` 或 `@<commit>`；私有仓库配置好 git 凭据即可。包 `ocr-mcp` 位于该仓库根，无需 `#subdirectory`。

## 工具

- **`ocr_image(image_path, return_polys=false, return_scores=false)`** —— 识别本地图片文件。返回一个 **markdown table**：表头 `| text |` + 每个文本行一行。需要坐标 / 置信度时传 `return_polys=true` / `return_scores=true`，表头会相应增加 `poly`（四点坐标）/ `score`（置信度）列。错误时返回 `is_error` 结果，消息形如 `ERROR: DAEMON_UNAVAILABLE: ...` 或 `ERROR: <CODE>: <message>`。

### 真实示例

默认调用（不传 `return_polys` / `return_scores`）返回一个 markdown table，只有 `text` 列：

```
| text |
| --- |
| 订单20260729金额99.0 |
```

需要置信度 / 坐标时传 `return_scores=true` / `return_polys=true`，表头会相应增加列：

```
| text | score | poly |
| --- | --- | --- |
| 订单20260729金额99.0 | 0.9978 | [[0, 22], [558, 24], [558, 76], [0, 74]] |
```

## 快速验证

    uv run pytest -q            # 单元测试（paddle 已 mock）
    uv run pytest -q -m slow    # 端到端（真实模型，需先 download）

手动走一遍真实 MCP 协议（stdio → daemon → OCR），下面这条管道把 `initialize` / `tools/call` 喂给 server：

    printf '%s\n' \
      '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"t","version":"0"}}}' \
      '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
      '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"ocr_image","arguments":{"image_path":"/abs/path/img.png"}}}' \
    | uv run ocr-mcp

> 注意：真实 MCP client 会保持 stdin 打开直到收到响应；上面的管道会在请求发出后立即 EOF，可能看到 `Connection closed`。它只用于冒烟（能看到 `initialize` 的 `serverInfo` 响应即说明 server 正常）。完整验证请用真实 client 或 `pytest -m slow`。

## 配置（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `OCR_MCP_MODEL_DET` | `PP-OCRv6_medium_det` | 检测模型 |
| `OCR_MCP_MODEL_REC` | `PP-OCRv6_medium_rec` | 识别模型 |
| `OCR_MCP_MODEL_DIR` | 未设（paddle 默认 `~/.paddlex`） | 自定义模型缓存目录（离线部署，会设置 `PADDLEX_HOME`）|
| `OCR_MCP_DATA_DIR` | 平台 user-data 目录 | daemon 的 lock / socket / log 位置 |
| `OCR_MCP_IDLE_TIMEOUT` | `600` | daemon 空闲退出秒数 |
| `OCR_MCP_REQUEST_TIMEOUT` | `60` | 单次 OCR 请求超时秒数 |
| `OCR_MCP_LOG_LEVEL` | `INFO` | 日志级别 |

## 平台支持

支持 **Linux、macOS、Windows 10 1803+（2018-04）** —— 三平台统一用 `AF_UNIX` Unix domain socket。

不支持老 Windows（< 10 1803）：缺少 `AF_UNIX`，且 `transport.named_pipe` 是刻意的 `NotImplementedError` stub（asyncio 没有 Windows 命名管道 server 的公开 API）。没有命名管道回退。

## 架构

双层进程模型：

- **瘦 stdio MCP server**（每 MCP client 进程一个）—— 注册 `ocr_image` 工具，不 import paddle。
- **常驻 OCR daemon**（每用户一个）—— 加载一次 PaddleOCR，串行推理（单实例非线程安全）。按需启动、引用计数、空闲自退出、token 鉴权。

两者通过 Unix socket 上的 NDJSON 帧通信（见 `src/ocr_mcp/protocol.py`）。完整设计见 `docs/superpowers/specs/2026-07-29-ocr-mcp-server-design.md`。

## 开发

    uv run pytest -q            # 单元测试（paddle mock）—— 沙箱内可跑
    uv run pytest -q -m slow    # 端到端（真实模型）
    uv run ruff check src tests

**沙箱说明**：默认套件的纯单元测试在沙箱内可跑。依赖 socket / paddle 的测试（`test_transport_unix`、`test_worker`、`test_lifecycle`、`-m slow` 集成测试）会创建 `AF_UNIX` socket 并加载真实模型，需**禁用沙箱**（否则 `Operation not permitted`）。

## 许可证

MIT，见 [LICENSE](LICENSE)。

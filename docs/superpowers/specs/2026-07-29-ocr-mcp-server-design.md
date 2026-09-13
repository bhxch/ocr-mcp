# OCR MCP Server 设计文档

- 日期：2026-07-29
- 状态：已确认，待实现
- 分支：`feat-mcp-ocr`

## 1. 背景与目标

前一会话（devops 项目 `tool/img-ocr/`）已验证 PaddleOCR PP-OCRv6 在纯 CPU 下可正确识别中文图片，但产物是**一次性 CLI 脚本**：每次 `uv run` 都重新加载模型，且每个调用方各自起进程、各自加载模型，内存浪费严重。

本项目的目标是构建一个 **uv 管理的 Python OCR MCP server**，核心解决「多 MCP client 共享单一 OCR 后端」的问题：

- 支持 **stdio 模式**（MCP 标准本地接入），stdio 下只接受本地图片文件路径。
- 多个 MCP client（Claude Code、Cursor 等）启动各自 stdio server 时，**探测本机是否已有常驻 OCR 后端**；已有则复用，没有则拉起一个。模型只加载一份。
- 设计考量：**并发正确、节省内存、复用后端、跨平台兼容**（Linux / macOS / Windows）。

### 非目标（YAGNI）

- 不做版面分析、表格识别、PDF 输入（本轮仅文本行 + 置信度 + 坐标）。
- 不做批量识别 tool（仅单张 `ocr_image`）。
- 不做 SSE / streamable-http 等 stdio 之外的 transport（后续可扩展）。
- 不做 GPU 推理（本轮纯 CPU；`paddlepaddle` 默认 CPU 轮子）。

## 2. 关键决策记录

| 决策点 | 结论 |
|---|---|
| OCR 引擎 | PaddleOCR `3.7.0` + paddlepaddle `3.3.1` + paddlex `3.7.2`，模型 **PP-OCRv6_medium**（v6 最大推理模型） |
| daemon 生命周期 | **懒启动 + 空闲超时自动退出**（最后一个 client 断开后空闲 N 分钟释放内存）|
| 功能范围 | 文本行 + 置信度 + 坐标（`dt_polys`）；不做版面/表格/PDF |
| 模型获取 | 自动下载 + `ocr-mcp download` 预下载子命令 + `OCR_MCP_MODEL_DIR` 自定义目录（覆盖联网/内网/离线）|
| IPC 传输 | **AF_UNIX（POSIX + Win10 1803+）+ lock file 寻址**；老 Windows (<10 1803) 不支持（`named_pipe` 为 `NotImplementedError` stub；执行期决策方案 A）|
| 并发模型 | 单 predictor 实例 + 串行推理队列（paddlepaddle 单实例非线程安全；只占一份模型内存）|
| 跨平台路径 | 引入 `platformdirs` 依赖 |
| MCP SDK | **mcp 2.0 `MCPServer`**（`FastMCP` 在 mcp 2.0 已移除；执行期从 brief 的 1.x 升级）|
| Python 版本 | `>=3.11,<3.13`（paddlepaddle 约束）|

## 3. 架构总览：双层进程模型

```
┌─────────────────────────────────────────────────────────────────┐
│  本机（单用户）                                                   │
│                                                                  │
│   MCP client A          MCP client B          MCP client C        │
│       │ stdio               │ stdio               │ stdio         │
│       ▼                     ▼                     ▼               │
│   ┌──────────┐         ┌──────────┐         ┌──────────┐         │
│   │ stdio    │         │ stdio    │         │ stdio    │  瘦前端  │
│   │ server A │         │ server B │         │ server C │  (每client│
│   │ (轻量)   │         │ (轻量)   │         │ (轻量)   │   一个)   │
│   └────┬─────┘         └────┬─────┘         └────┬─────┘         │
│        │ Unix socket/        │                   │                │
│        │ named pipe          │                   │                │
│        └──────────┬──────────┴───────────────────┘                │
│                   ▼                                              │
│          ┌─────────────────────┐    常驻后端 daemon（全局唯一）    │
│          │   OCR backend        │                                 │
│          │   ┌───────────────┐  │    • 只加载一次 PP-OCRv6 模型    │
│          │   │ PaddleOCR 单例│  │    • 串行推理队列（线程安全）     │
│          │   └───────┬───────┘  │    • client 引用计数 + 空闲超时   │
│          │           │ predict  │    • lock file 寻址 + 原子拉起    │
│          └───────────┴──────────┘                                 │
└─────────────────────────────────────────────────────────────────┘
```

**核心思想：** 模型只在 daemon 里加载一份；所有 stdio server 都是瘦进程，只做 MCP ↔ socket 协议转换，不碰 paddle。从而彻底避免「每个 MCP client 各自加载后端」。

## 4. 组件划分（src 布局）

```
src/ocr_mcp/
├── __main__.py              # CLI 入口
├── cli.py                   # 子命令路由: stdio | serve | download
├── config.py                # 环境变量/配置(模型档位/超时/路径/自定义模型目录)
├── paths.py                 # 跨平台 lock/socket/log 文件位置(用 platformdirs)
├── protocol.py              # 请求/响应 JSON schema + 换行分隔序列化
├── server.py                # MCP server 定义 + tool 注册(stdio 前端, 不 import paddle)
├── transport/
│   ├── base.py              # Transport 抽象接口(connect/listen/send/recv)
│   ├── unix_sock.py         # AF_UNIX(POSIX + Win10 1803+ 统一实现)
│   └── named_pipe.py        # NotImplementedError stub(老 Windows 不支持,方案 A)
└── backend/
    ├── engine.py            # PaddleOCR 封装(单例/predict/结果标准化; 延迟 import paddle)
    ├── worker.py            # daemon 主体:监听/串行队列/引用计数/空闲超时
    └── lifecycle.py         # 探测/拉起/lock file/陈旧清理/竞争仲裁
```

每个单元职责单一、可独立测试：`engine` 只管模型，`transport` 只管字节搬运，`protocol` 只管消息格式，`backend.lifecycle` 只管进程编排，`server` 只管 MCP 协议。

### 🔑 关键架构约束

**stdio server 进程绝不能 `import paddleocr/paddlepaddle`。** 否则瘦前端也会被拖进几百 MB。`engine.py` 对 paddle 的 import 必须延迟到 daemon worker 内部；`server.py` 只经 transport 调 daemon，完全不碰 `engine`。这是「省内存」的第二道保险（第一道是共享后端）。

## 5. 数据流（一次 `ocr_image` 调用）

1. MCP client spawn `ocr-mcp stdio`，走标准 MCP stdio 协议握手。
2. client 调 tool → stdio server 调 `lifecycle.ensure_daemon()`：
   - 读 lock file（pid + socket 路径 + token）→ pid 存活且 socket 可连 → 复用。
   - 否则 → **detached spawn** `ocr-mcp serve` → 等 socket 就绪 → 连接。
3. server 经 transport 发一行 JSON 请求（见 §6）。
4. daemon worker 从串行队列取该请求 → `engine.predict(path)` → 标准化为 `{texts, scores, polys}` → 回写一行 JSON。
5. server 收响应 → 转 MCP `TextContent` → 回 client。
6. server 存活期间 daemon 引用计数 +1；server 退出 −1；归零后启动空闲计时器，到点（默认 10 分钟）daemon 自退出，释放全部模型内存。

## 6. 通信协议（stdio server ↔ daemon）

换行分隔 JSON（NDJSON）：每条消息是一行 UTF-8 JSON 后跟 `\n`。JSON 序列化时内部换行自动转义为 `\n` 字符串，不破坏消息边界。

### 6.1 消息信封

**请求（server → daemon）**

```json
{
  "v": 1,
  "id": "9b3f1a2c...",
  "op": "ocr",
  "token": "a1b2c3d4...",
  "payload": { ... }
}
```

**响应成功（daemon → server）**

```json
{ "v": 1, "id": "9b3f1a2c...", "ok": true, "result": { ... } }
```

**响应失败（daemon → server）**

```json
{ "v": 1, "id": "9b3f1a2c...", "ok": false,
  "error": { "code": "IMAGE_NOT_FOUND", "message": "image file not found: /abs/path/x.png" } }
```

每个请求必有 `id`，daemon 对每个 `id` 恰好回一个响应。

### 6.2 `op: "ocr"` —— 图片识别

请求 payload：

```json
{
  "image_path": "/abs/path/to/img.png",
  "options": {
    "use_textline_orientation": false,
    "return_polys": true,
    "return_scores": true
  }
}
```

- `image_path`：必填，本机可访问的绝对路径。daemon 与 server 同机同用户，路径直接共享。
- `return_polys` / `return_scores`：按需索取，关掉可减小响应体积。

响应 result：

```json
{
  "image_path": "/abs/path/to/img.png",
  "width": 1742,
  "height": 395,
  "lines": [
    { "text": "借：660114 销售费用_销售运费", "score": 0.97,
      "poly": [[12, 34], [180, 34], [180, 58], [12, 58]] }
  ],
  "text": "借：660114 销售费用_销售运费\n第二行...\n第三行...",
  "elapsed_ms": 1234
}
```

- `lines[].poly`：透传 PaddleOCR `dt_polys`（N×2 数组），通常 4 点矩形，弯曲文本可能更多点，不强制 4 点。
- `text`：daemon 预拼接纯文本，方便直接喂给 LLM。

`ocr` 专属错误码：

| code | 触发条件 |
|---|---|
| `IMAGE_NOT_FOUND` | `image_path` 不存在 |
| `IMAGE_UNREADABLE` | 存在但无法读取（权限/损坏）|
| `UNSUPPORTED_FORMAT` | 非支持的图片格式 |
| `INFERENCE_FAILED` | paddle 推理抛异常（`message` 含原始错误）|

### 6.3 `op: "ping"` —— 健康检查 / 保活

请求 payload：`{}`

响应 result：

```json
{ "status": "ready", "pid": 12345, "uptime_s": 3600, "clients": 2,
  "model": "PP-OCRv6_medium_det + PP-OCRv6_medium_rec" }
```

用于 `lifecycle.ensure_daemon()` 探测 daemon 是否健康、模型是否加载完成。

### 6.4 横切约定

- **鉴权（token）**：daemon 启动生成 `secrets.token_hex(16)` 随机 token 写入 lock file；server 每请求带上；daemon 校验，不符返回 `error.code=UNAUTHORIZED`。配合 socket 文件权限 `600`（POSIX）/ 命名管道 ACL（Windows），双重保险。
- **引用计数 + 空闲超时**：server 与 daemon 保持长连接（一个 stdio server 进程一个持久连接）。daemon accept 时 `clients += 1`，断开时 `clients -= 1`；`clients == 0` 启动空闲计时器（默认 600s），期间有新连接则取消，到点退出。
- **串行队列**：daemon 内专用推理协程/线程从 FIFO 队列取请求串行 `predict`；server 端每请求带超时（默认 60s），超时返回 `error.code=TIMEOUT`。
- **消息边界**：NDJSON 行式协议；不引入二进制长度前缀（YAGNI）。

## 7. MCP tool 接口

只暴露一个 tool `ocr_image`，参数与协议 `options` 1:1 映射：

```jsonc
{
  "image_path":   "string (必填), 本地图片绝对路径",
  "return_polys": "bool (可选, 默认 true)",
  "return_scores":"bool (可选, 默认 true)"
}
```

返回单个 `TextContent`，内容是 daemon `result` 的 JSON（含 `text` + `lines` + `elapsed_ms`）。失败时返回 MCP `isError: true` + 人类可读错误说明。

## 8. CLI 子命令与 client 配置

| 子命令 | 作用 |
|---|---|
| `stdio`（默认，无参也走它）| 启动瘦 stdio MCP server 前端 |
| `serve` | 启动后端 daemon（一般由 stdio 自动拉起，也可手动跑用于调试/强制常驻）|
| `download` | 仅预下载模型到缓存，不启动 daemon（纳入部署/CI）|

MCP client 配置示例：

```jsonc
{ "mcpServers": { "ocr": {
  "command": "uv",
  "args": ["--directory", "/abs/path/ocr-mcp", "run", "ocr-mcp"]
} } }
```

## 9. 配置项（环境变量，`config.py`）

| 变量 | 默认 | 说明 |
|---|---|---|
| `OCR_MCP_MODEL_DET` | `PP-OCRv6_medium_det` | 检测模型名 |
| `OCR_MCP_MODEL_REC` | `PP-OCRv6_medium_rec` | 识别模型名 |
| `OCR_MCP_MODEL_DIR` | `~/.paddlex` | 自定义模型缓存目录（离线/内网部署）|
| `OCR_MCP_DATA_DIR` | `platformdirs.user_data_dir("ocr-mcp")` | lock/socket/log 文件目录 |
| `OCR_MCP_IDLE_TIMEOUT` | `600` | daemon 空闲退出秒数 |
| `OCR_MCP_REQUEST_TIMEOUT` | `60` | 单次 OCR 请求超时秒数 |
| `OCR_MCP_LOG_LEVEL` | `INFO` | 日志级别 |

> **实现时验证：** `OCR_MCP_MODEL_DIR` 如何透传给 paddleocr 3.x（可能为环境变量 `PADDLEX_HOME` 或初始化参数）。默认沿用 `~/.paddlex/official_models`，自定义机制在实现阶段确认。

## 10. 错误处理（分层）

- **tool 层**：参数校验失败 → MCP error。
- **lifecycle 层**：拉起失败重试一次→仍失败 `DAEMON_UNAVAILABLE`；lock file 陈旧（pid 已死）→ 清理 stale lock + socket 后重 spawn；并发竞争 → `O_CREAT|O_EXCL` 原子建 lock 仲裁，落败方等待重探。
- **transport 层**：断连 → server 重新 `ensure_daemon` 一次。
- **engine 层**：模型加载/下载失败 → daemon 退出码非 0 + 日志，server 报 `MODEL_LOAD_FAILED`（message 提示跑 `ocr-mcp download` 或设 `OCR_MCP_MODEL_DIR`）；推理异常 → `INFERENCE_FAILED`（含原始错误）。
- **图片层**：`IMAGE_NOT_FOUND` / `IMAGE_UNREADABLE` / `UNSUPPORTED_FORMAT`。

## 11. 测试策略（分层单测为主，不依赖模型/网络）

- `test_protocol`：序列化 round-trip、非法 JSON 拒绝、必填字段缺失校验。
- `test_transport`：loopback 测 unix socket 连接/收发/断连/残留 socket 清理。
- `test_engine`：**mock `PaddleOCR`**，测结果标准化（`rec_texts/rec_scores/dt_polys`→`lines`）、单例只 init 一次、`enable_mkldnn=False` 正确透传。
- `test_lifecycle`：mock subprocess + 假 lock file，测探测命中/未命中、陈旧清理、原子竞争、等就绪。
- `test_worker`：mock engine，测串行队列顺序、引用计数增减、空闲超时退出、token 拒绝。
- `test_server`：用 mcp SDK test client 测 tool 注册、参数校验、成功/失败响应转换。
- 集成测试（标记 `slow`，需模型）：起真 daemon 对 `tests/fixtures/*.png` 跑端到端。

## 12. 跨平台与依赖

**socket transport**：POSIX + Win10 1803+ 统一用 `socket.AF_UNIX`（文件 `data_dir/daemon.sock`）；残留 socket 文件 connect 失败时删除重建。**老 Windows (<10 1803) 不支持**（执行期决策方案 A）：asyncio 无 Windows 命名管道 server API，且 AF_UNIX 已覆盖 Win10 1803+；`named_pipe` 模块保留为 `NotImplementedError` stub。

**detached spawn**：POSIX `start_new_session=True`；Windows `CREATE_NEW_PROCESS_GROUP|DETACHED_PROCESS` + 隐藏窗口。daemon 的 stdout/stderr **重定向到日志文件**（绝不能写继承的 stdout，避免污染）。

**依赖清单（pyproject.toml）**：

- 运行：`mcp[cli]>=2.0`、`paddleocr>=3.0`（验证过 3.7.0）、`paddlepaddle>=3.0`（验证过 3.3.1）、`platformdirs>=4.0`
- dev：`pytest`、`pytest-asyncio`、`ruff`
- 构建：`hatchling`，src 布局，`requires-python = ">=3.11,<3.13"`
- entry point：`ocr-mcp = "ocr_mcp.__main__:main"`

## 13. 已知技术坑（来自前一会话验证）

1. **PP-OCRv6 模型命名**：v6 用 `medium/small/tiny`（无 `server/mobile`），最大推理模型是 `PP-OCRv6_medium_det` / `PP-OCRv6_medium_rec`。
2. **CPU 必须 `enable_mkldnn=False`**：paddlepaddle 3.3 默认 PIR + onednn，PP-OCRv6 medium 的 PIR 属性 `ConvertPirAttribute2RuntimeAttribute` 不支持，会崩。关 onednn 走纯 CPU。
3. **用 `predict()` 而非废弃的 `ocr()`**：`ocr()` 在 paddleocr 3.x 已废弃且返回结构变了。
4. **结果用 dict 取值**：`OCRResult` 继承 dict，`rec_texts/rec_scores/dt_polys` 是 dict key 而非对象属性，用 `res.get("rec_texts")`，不是 `getattr`。
5. **显式指定模型名时 `lang` 无效**：语言能力由 PP-OCRv6 模型本身决定（ch 模型支持中英数字）。
6. **模型下载源（百度 bcebos）需外网**：首次下载约 det 16s / rec 23s，缓存于 `~/.paddlex/official_models/`。

## 14. 未来扩展（记录，本轮不做）

- SSE / streamable-http transport（远程访问）。
- 版面分析 / 表格识别 / PDF 输入（需加载额外模型，内存显著增加）。
- 批量识别 tool。
- GPU 推理（换 `paddlepaddle-gpu`）。
- 多 predictor 实例池（提升吞吐，代价是内存 ×N）。

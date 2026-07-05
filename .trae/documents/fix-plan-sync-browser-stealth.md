# 修复计划：plan 进度同步 + 浏览器反检测 + OpenManus 设计移植

## 背景

三个问题需要同时解决：
1. 深思熟虑式任务完成后，最后一个 plan 步骤未打勾显示 completed
2. 向 OpenManus 移植成熟设计（click_element/input_text、refresh、web_search 改进）
3. 浏览器无法打开携程等网站（缺少反自动化检测措施）

---

## 修改 1：observe_node — 终止时标记当前步骤为 completed + 严格 is_finished 检查

**文件**: `agent/nodes/observe.py`

### 1.1 终止时当前步骤标 completed，后续才标 skipped

当前逻辑：terminate 时从 `current_step_index` 开始全标 `skipped`，导致最后完成的步骤也显示未完成。

修改：terminate 时先将 `current_step_index` 标为 `completed`，再将从 `current_step_index + 1` 开始标 `skipped`。

### 1.2 "无工具调用"路径必须所有步骤都完成才设 is_finished

当前逻辑：当前步骤完成就设 `is_finished=True`，没检查全部步骤。

修改：遍历 plan 检查所有步骤 status，只有全部 completed/done/skipped 才设 `is_finished`。

---

## 修改 2：browser_use.py — 反检测 + click/input + refresh + web_search 改进

**文件**: `tools/browser_use.py`

### 2.1 添加反自动化检测参数（解决携程等问题）

在 `_ensure_browser_context()` 中：

- `BrowserConfig` 添加 `extra_chromium_args`:
  - `--disable-blink-features=AutomationControlled` — 移除 Chrome 自动化标志
  - `--no-sandbox`
  - `--disable-infobars`
  - `--disable-dev-shm-usage`

- context 创建后注入 `add_init_script()` 反检测 JS：
  - `navigator.webdriver = undefined`
  - 移除 `cdc_` 开头属性
  - 伪造 `chrome.runtime`
  - 伪造 `navigator.plugins`（非空数组）
  - 伪造 `navigator.languages = ['zh-CN', 'zh', 'en']`

### 2.2 实现 click_element 和 input_text

参照 OpenManus，在 `_run_browser_action` 中添加：

- `click_element`: `context.get_dom_element_by_index(index)` → `context._click_element_node(element)` → `page.wait_for_load_state(timeout=5000)` → `asyncio.sleep(1)` → `_safe_page_summary`
- `input_text`: `context.get_dom_element_by_index(index)` → `context._input_text_element_node(element, text)` → `_safe_page_summary`

### 2.3 添加 refresh 操作

- `_VALID_ACTIONS` 添加 `"refresh"`
- `_run_browser_action` 添加 `refresh` 分支：`await context.refresh_page()`
- 更新 `browser_use.description` 添加 `refresh`

### 2.4 改进 web_search 操作

当前硬编码百度 URL，改为：
1. 先调用 `tools.web_search._search(query)` 获取搜索结果
2. 导航到第一个结果的 URL
3. 如果搜索失败，回退到百度搜索

---

## 修改 3：think.py — 修复 browser_use fallback 使用 web_search 而非 go_to_url

**文件**: `agent/nodes/think.py`

当前 `_extract_fallback_tool_call` 中 browser_use 的 fallback 构造 `{"action": "go_to_url", "url": query_value}`，但 `query_value` 是搜索词不是 URL。

修改：改为 `{"action": "web_search", "query": query_value}`，让浏览器先搜索再导航。

---

## 移植的 OpenManus 设计清单

| # | 设计 | 来源 | 应用到 |
|---|------|------|--------|
| 1 | click_element 实现（get_dom_element_by_index + _click_element_node + 等待加载） | OpenManus browser_use_tool.py 第272-341行 | browser_use.py |
| 2 | input_text 实现（get_dom_element_by_index + _input_text_element_node） | OpenManus browser_use_tool.py 第343-354行 | browser_use.py |
| 3 | refresh 操作 | OpenManus browser_use_tool.py 第248-250行 | browser_use.py |
| 4 | web_search 先搜后导航（而非硬编码百度） | OpenManus browser_use_tool.py 第252-269行 | browser_use.py |
| 5 | extra_chromium_args 反检测 | OpenManus browser_use_tool.py 第147-171行（配置支持） | browser_use.py |
| 6 | add_init_script 反检测 JS 注入 | OpenManus 风格的浏览器隐身最佳实践 | browser_use.py |
| 7 | click 后等待动态加载（wait_for_load_state + sleep） | OpenManus browser_use_tool.py 第285-286行 | browser_use.py |

---

## 验证方式

1. `pytest tests -q` — 全量测试通过
2. 手动测试：问"帮我写一份小米集团2024年的简要分析报告" → 所有 plan 步骤在回答前都显示 completed
3. 手动测试：问"去携程查询7月1日从上海到北京的机票" → 浏览器能成功打开携程并返回结果
4. 手动测试：refresh、click_element、input_text 操作正常工作

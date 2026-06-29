import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./index.css";

const rootEl = document.getElementById("root");
if (!rootEl) {
  document.body.innerHTML = '<div style="padding:20px;color:red">Error: root element not found</div>';
} else {
  try {
    createRoot(rootEl).render(
      <StrictMode>
        <App />
      </StrictMode>,
    );
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    rootEl.innerHTML = `<div style="padding:20px;color:red;font-family:monospace;white-space:pre-wrap">React render error:<br/>${message}<br/><br/>${err instanceof Error ? err.stack : ""}</div>`;
  }

  // 3 秒后如果 root 仍为空，显示诊断提示
  setTimeout(() => {
    if (rootEl && rootEl.innerHTML.trim() === "") {
      rootEl.innerHTML = '<div style="padding:20px;color:#666">页面加载超时，请按 F12 查看控制台错误信息。</div>';
    }
  }, 3000);
}

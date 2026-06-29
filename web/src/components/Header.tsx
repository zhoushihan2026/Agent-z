// Agent-z Header 组件

interface HeaderProps {
  sidebarCollapsed: boolean;
  onToggleSidebar: () => void;
}

export function Header({ sidebarCollapsed, onToggleSidebar }: HeaderProps) {
  return (
    <header className="flex items-center h-12 bg-white border-b border-[#F1F5F9] shrink-0 px-4">
      <button
        onClick={onToggleSidebar}
        className="p-1.5 text-[#64748B] hover:text-[#1E293B] hover:bg-[#F1F5F9] rounded-lg transition-colors mr-3"
        title={sidebarCollapsed ? "展开侧边栏" : "折叠侧边栏"}
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M3 12h18M3 6h18M3 18h18" />
        </svg>
      </button>
      <h1 className="text-[14px] font-medium text-[#64748B]">
        Agent-z 深度研报分析
      </h1>
    </header>
  );
}

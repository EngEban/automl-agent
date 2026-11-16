"use client";

import { useId, useRef, useState } from "react";
import type { KeyboardEvent, ReactNode } from "react";

export interface TabItem {
  id: string;
  label: string;
  content: ReactNode;
}

export function Tabs({ items, label = "Content sections" }: { items: TabItem[]; label?: string }) {
  const instanceId = useId();
  const [activeId, setActiveId] = useState(items[0]?.id ?? "");
  const tabRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const activeTab = items.find((item) => item.id === activeId) ?? items[0];

  if (items.length === 0) return null;

  function moveFocus(event: KeyboardEvent<HTMLButtonElement>, currentIndex: number) {
    const direction = event.key === "ArrowRight" ? 1 : event.key === "ArrowLeft" ? -1 : 0;
    if (!direction) return;
    event.preventDefault();
    const nextIndex = (currentIndex + direction + items.length) % items.length;
    setActiveId(items[nextIndex].id);
    tabRefs.current[nextIndex]?.focus();
  }

  return (
    <div className="tabs">
      <div className="tab-list" role="tablist" aria-label={label}>
        {items.map((item, index) => (
          <button
            key={item.id}
            ref={(node) => { tabRefs.current[index] = node; }}
            className="tab"
            id={`${instanceId}-tab-${item.id}`}
            type="button"
            role="tab"
            aria-selected={activeTab?.id === item.id}
            aria-controls={`${instanceId}-panel`}
            tabIndex={activeTab?.id === item.id ? 0 : -1}
            onClick={() => setActiveId(item.id)}
            onKeyDown={(event) => moveFocus(event, index)}
          >
            {item.label}
          </button>
        ))}
      </div>
      {activeTab && (
        <div className="tab-panel" id={`${instanceId}-panel`} role="tabpanel" aria-labelledby={`${instanceId}-tab-${activeTab.id}`} tabIndex={0}>
          {activeTab.content}
        </div>
      )}
    </div>
  );
}

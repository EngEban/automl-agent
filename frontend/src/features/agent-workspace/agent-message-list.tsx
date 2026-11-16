"use client";

import { Bot, UserRound, Wrench } from "lucide-react";
import type { AgentMessageRead } from "@/types/api";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function toolNames(value: Record<string, unknown> | null | undefined): string[] {
  if (!value || !Array.isArray(value.calls)) return [];
  return value.calls.flatMap((call) => {
    if (!isRecord(call) || !isRecord(call.function)) return [];
    return typeof call.function.name === "string" ? [call.function.name] : [];
  });
}

function toolResultName(value: Record<string, unknown> | null | undefined): string {
  return value && typeof value.tool === "string" ? value.tool : "Agent tool";
}

function timestampLabel(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? ""
    : new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit" }).format(date);
}

export function AgentMessageList({ messages }: { messages: AgentMessageRead[] }) {
  return (
    <div className="agent-message-list" aria-live="polite">
      {messages.filter((message) => message.role !== "system").map((message) => {
        if (message.role === "tool") {
          return (
            <details className="agent-tool-result" key={message.id}>
              <summary><Wrench size={14} aria-hidden="true" /> Tool result · {toolResultName(message.tool_results)}</summary>
              <pre>{message.content}</pre>
              {timestampLabel(message.created_at) && <time dateTime={message.created_at}>{timestampLabel(message.created_at)}</time>}
            </details>
          );
        }

        const isUser = message.role === "user";
        const calls = toolNames(message.tool_calls);
        return (
          <article className={`agent-message agent-message-${isUser ? "user" : "assistant"}`} key={message.id}>
            <span className="agent-message-avatar" aria-hidden="true">
              {isUser ? <UserRound size={16} /> : <Bot size={16} />}
            </span>
            <div className="agent-message-body">
              <div className="agent-message-heading">
                <strong>{isUser ? "You" : "Agent"}</strong>
                {timestampLabel(message.created_at) && <time dateTime={message.created_at}>{timestampLabel(message.created_at)}</time>}
              </div>
              {message.content && <p className="agent-message-content">{message.content}</p>}
              {calls.length > 0 && (
                <div className="agent-tool-call-list" aria-label="Agent tool calls">
                  {calls.map((name, index) => <span key={`${message.id}-${name}-${index}`}><Wrench size={12} aria-hidden="true" />{name}</span>)}
                </div>
              )}
              {!message.content && calls.length === 0 && !isUser && <p className="agent-message-content">The Agent returned an empty message.</p>}
            </div>
          </article>
        );
      })}
    </div>
  );
}

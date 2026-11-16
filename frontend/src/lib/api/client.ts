import type { ApiErrorBody } from "@/types/api";

const configuredOrigin = process.env.NEXT_PUBLIC_API_BASE_URL?.trim();
const backendOrigin = (configuredOrigin || "http://localhost:8100").replace(/\/+$/, "");

/** Backend origin plus the prefix mounted by app.main:app. */
export const API_BASE_URL = backendOrigin.endsWith("/api")
  ? backendOrigin
  : `${backendOrigin}/api`;

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
    public readonly body: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export class ApiTimeoutError extends Error {
  constructor(timeoutMs: number) {
    super(`The API request exceeded its ${timeoutMs} ms timeout.`);
    this.name = "ApiTimeoutError";
  }
}

export interface ApiRequestOptions<TBody = never>
  extends Omit<RequestInit, "body" | "headers" | "signal"> {
  body?: TBody | FormData;
  /** The FastAPI backend uses bearer tokens; tokens are passed in memory per request. */
  token?: string;
  headers?: HeadersInit;
  signal?: AbortSignal;
  timeoutMs?: number;
}

async function readError(response: Response): Promise<ApiError> {
  const rawBody = await response.text().catch(() => "");
  let body: unknown = rawBody || undefined;

  if (rawBody) {
    try {
      body = JSON.parse(rawBody) as unknown;
    } catch {
      body = rawBody;
    }
  }

  const detail = (body as ApiErrorBody | undefined)?.detail;
  let message = response.statusText || `The API request failed (${response.status}).`;
  if (typeof detail === "string") {
    message = detail;
  } else if (Array.isArray(detail)) {
    const firstIssue = detail[0];
    if (firstIssue && typeof firstIssue === "object" && "msg" in firstIssue && typeof firstIssue.msg === "string") {
      message = firstIssue.msg;
    }
  } else if (detail !== null && typeof detail === "object") {
    if (typeof detail.message === "string") message = detail.message;
    else if (typeof detail.msg === "string") message = detail.msg;
  }

  return new ApiError(message, response.status, body);
}

/** Typed JSON request helper for the existing FastAPI /api routes. */
export async function apiRequest<TResponse, TBody = never>(
  path: string,
  options: ApiRequestOptions<TBody> = {},
): Promise<TResponse> {
  const {
    body,
    token,
    headers: suppliedHeaders,
    signal: externalSignal,
    timeoutMs = 15_000,
    ...requestInit
  } = options;

  const controller = new AbortController();
  let didTimeout = false;
  const timeout = setTimeout(() => {
    didTimeout = true;
    controller.abort();
  }, timeoutMs);
  const forwardAbort = () => controller.abort(externalSignal?.reason);

  if (externalSignal?.aborted) {
    forwardAbort();
  } else {
    externalSignal?.addEventListener("abort", forwardAbort, { once: true });
  }

  const headers = new Headers(suppliedHeaders);
  headers.set("Accept", "application/json");
  if (token) headers.set("Authorization", `Bearer ${token}`);

  let requestBody: BodyInit | undefined;
  const routePath = path.replace(/^\/+/, "").replace(/^api(?:\/|$)/, "");
  if (body !== undefined) {
    if (typeof FormData !== "undefined" && body instanceof FormData) {
      headers.delete("Content-Type");
      requestBody = body;
    } else {
      headers.set("Content-Type", "application/json");
      requestBody = JSON.stringify(body);
    }
  }

  try {
    const response = await fetch(
      `${API_BASE_URL}/${routePath}`,
      {
        ...requestInit,
        headers,
        body: requestBody,
        signal: controller.signal,
      },
    );

    if (!response.ok) throw await readError(response);
    if (response.status === 204) return undefined as TResponse;
    return (await response.json()) as TResponse;
  } catch (error) {
    if (didTimeout) throw new ApiTimeoutError(timeoutMs);
    throw error;
  } finally {
    clearTimeout(timeout);
    externalSignal?.removeEventListener("abort", forwardAbort);
  }
}

export interface ApiDownloadFile {
  blob: Blob;
  filename: string | null;
}

function responseFilename(contentDisposition: string | null): string | null {
  if (!contentDisposition) return null;

  const encoded = contentDisposition.match(/filename\*=UTF-8''([^;]+)/i)?.[1];
  const quoted = contentDisposition.match(/filename\s*=\s*"([^"]+)"/i)?.[1];
  const plain = contentDisposition.match(/filename\s*=\s*([^;]+)/i)?.[1]?.trim();
  let candidate = encoded ?? quoted ?? plain;
  if (!candidate) return null;

  try {
    if (encoded) candidate = decodeURIComponent(candidate);
  } catch {
    return null;
  }

  const safeName = candidate
    .replace(/\\/g, "/")
    .split("/")
    .pop()
    ?.replace(/[\x00-\x1f\x7f]/g, "")
    .trim();
  return safeName && safeName !== "." && safeName !== ".." ? safeName.slice(0, 180) : null;
}

/** Authenticated binary download and safe filename metadata from the response. */
export async function apiDownloadFile(
  path: string,
  options: { token: string; signal?: AbortSignal; timeoutMs?: number },
): Promise<ApiDownloadFile> {
  const { token, signal: externalSignal, timeoutMs = 30_000 } = options;
  const controller = new AbortController();
  let didTimeout = false;
  const timeout = setTimeout(() => {
    didTimeout = true;
    controller.abort();
  }, timeoutMs);
  const forwardAbort = () => controller.abort(externalSignal?.reason);

  if (externalSignal?.aborted) forwardAbort();
  else externalSignal?.addEventListener("abort", forwardAbort, { once: true });

  const headers = new Headers({ Accept: "application/octet-stream" });
  headers.set("Authorization", `Bearer ${token}`);
  const routePath = path.replace(/^\/+/, "").replace(/^api(?:\/|$)/, "");

  try {
    const response = await fetch(`${API_BASE_URL}/${routePath}`, {
      headers,
      signal: controller.signal,
    });
    if (!response.ok) throw await readError(response);
    return {
      blob: await response.blob(),
      filename: responseFilename(response.headers.get("Content-Disposition")),
    };
  } catch (error) {
    if (didTimeout) throw new ApiTimeoutError(timeoutMs);
    throw error;
  } finally {
    clearTimeout(timeout);
    externalSignal?.removeEventListener("abort", forwardAbort);
  }
}

/** Authenticated binary download through the same API origin and error handling. */
export async function apiDownload(
  path: string,
  options: { token: string; signal?: AbortSignal; timeoutMs?: number },
): Promise<Blob> {
  return (await apiDownloadFile(path, options)).blob;
}

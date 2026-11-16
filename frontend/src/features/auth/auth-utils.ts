export function safeAuthDestination(candidate: string | null | undefined): string {
  if (!candidate || !candidate.startsWith("/") || candidate.startsWith("//") || candidate.includes("\\")) {
    return "/workspace";
  }

  try {
    const parsed = new URL(candidate, "http://automl-agent.local");
    if (parsed.origin !== "http://automl-agent.local") return "/workspace";
    if (["/login", "/register"].includes(parsed.pathname)) return "/workspace";
    return `${parsed.pathname}${parsed.search}${parsed.hash}`;
  } catch {
    return "/workspace";
  }
}

export function firstAuthSearchParam(value: string | string[] | undefined): string | null {
  return Array.isArray(value) ? value[0] ?? null : value ?? null;
}

export function authErrorMessage(error: unknown, action: "login" | "register"): string {
  if (error instanceof Error && error.name === "ApiTimeoutError") {
    return "The request took too long. Please check your connection and try again.";
  }

  if (error instanceof Error && error.name === "ApiError" && "status" in error) {
    const status = (error as Error & { status: number }).status;
    if (action === "login" && status === 401) return "The email or password is incorrect.";
    if (action === "login" && status === 403) return "Please verify your email before signing in.";
    if (action === "register" && status === 400) return "An account with this email already exists.";
    if (status === 422) return "Please check the information and try again.";
    if (status >= 500) return "The service is temporarily unavailable. Please try again shortly.";
    return action === "login"
      ? "We could not sign you in. Please check your details and try again."
      : "We could not create your account. Please check your details and try again.";
  }

  if (error instanceof TypeError) {
    return "We could not reach the service. Check your connection and try again.";
  }

  return action === "login"
    ? "We could not sign you in. Please try again."
    : "We could not create your account. Please try again.";
}
